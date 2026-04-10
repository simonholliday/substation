"""Tests for PSD calculation, noise floor estimation, and detection logic."""

import numpy
import pytest

import substation.constants
import substation.scanner

import iq_generators


class TestPSDCalculation:

	def test_noise_flat_psd (self, scanner_instance):
		"""Gaussian noise should produce a roughly flat PSD."""
		iq = iq_generators.generate_noise_iq(scanner_instance.sample_rate, 0.1)
		# Trim to expected slice size
		iq = iq[:scanner_instance.samples_per_slice]
		psd_db, _ = scanner_instance._calculate_psd_data(iq, include_segment_psd=False)
		assert len(psd_db) == scanner_instance.fft_size
		# PSD should be roughly flat: std of dB values should be moderate
		assert numpy.std(psd_db) < 15  # dB

	def test_tone_peak_in_psd (self, scanner_instance):
		"""A tone should produce a clear peak in the PSD."""
		sc = scanner_instance
		# Place a tone at the first channel frequency
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 0.1, amplitude=1.0)
		noise = iq_generators.generate_noise_iq(sc.sample_rate, 0.1, power_db=-50)
		iq = (tone + noise)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		# The peak should be significantly above the median
		assert numpy.max(psd_db) > numpy.median(psd_db) + 10


class TestNoiseFloorEstimation:

	def test_with_noise_mask (self, scanner_instance):
		"""Noise floor should be estimated from gap bins."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		nf = sc._estimate_noise_floor(psd_db)
		assert isinstance(nf, (float, numpy.floating))

	def test_fallback_without_mask (self, scanner_instance):
		"""Without noise mask, falls back to 25th percentile."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		# Temporarily remove noise mask
		saved_mask = sc.noise_mask
		sc.noise_mask = None
		nf = sc._estimate_noise_floor(psd_db)
		sc.noise_mask = saved_mask
		assert isinstance(nf, (float, numpy.floating))


class TestEMASmoothing:

	def test_ema_converges (self, scanner_instance):
		"""EMA noise floor should converge over multiple slices."""
		sc = scanner_instance
		sc._noise_floor_ema = None
		sc._warmup_remaining = 0  # skip warmup for this test

		floors = []
		for _ in range(20):
			iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
			psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
			raw_nf = sc._estimate_noise_floor(psd_db)

			if sc._noise_floor_ema is None:
				sc._noise_floor_ema = raw_nf
			else:
				alpha = substation.constants.NOISE_FLOOR_EMA_ALPHA
				sc._noise_floor_ema = alpha * raw_nf + (1 - alpha) * sc._noise_floor_ema
			floors.append(sc._noise_floor_ema)

		# Later values should have less variance than early ones
		early_var = numpy.var(floors[:5])
		late_var = numpy.var(floors[-5:])
		assert late_var <= early_var + 1.0  # allow small tolerance


class TestChannelPower:

	def test_tone_on_channel (self, scanner_instance):
		"""Injecting a tone at a channel frequency should produce high channel power."""
		sc = scanner_instance
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 0.1, amplitude=1.0)
		noise = iq_generators.generate_noise_iq(sc.sample_rate, 0.1, power_db=-50)
		iq = (tone + noise)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		nf = sc._estimate_noise_floor(psd_db)
		powers = sc._get_channel_powers(psd_db)
		snr_ch0 = powers[0] - nf
		# Should show clear signal presence (well above noise)
		# Note: exact SNR depends on FFT size, windowing, and tone/bin alignment
		assert snr_ch0 > 5.0

	def test_noise_only_below_threshold (self, scanner_instance):
		"""With only noise, all channels should be below threshold."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1, power_db=-20)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		nf = sc._estimate_noise_floor(psd_db)
		powers = sc._get_channel_powers(psd_db)
		snrs = powers - nf
		assert numpy.all(snrs < sc.snr_threshold_db)

	def test_vectorized_matches_individual (self, scanner_instance):
		"""Vectorized channel powers should match individual computation."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		batch_powers = sc._get_channel_powers(psd_db)
		for i, ch_freq in enumerate(sc.channels):
			individual = sc._get_channel_power(psd_db, ch_freq)
			assert batch_powers[i] == pytest.approx(individual, abs=0.1)


class TestSegmentPowerVariance:

	def test_variance_low_for_stationary_noise (self, scanner_instance):

		"""
		Stationary Gaussian noise should produce low temporal variance across
		segment PSDs — this is the discriminator that lets us reject noise-only
		channel triggers.
		"""

		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.5)[:sc.samples_per_slice]
		_, segment_psds = sc._calculate_psd_data(iq, include_segment_psd=True)
		assert segment_psds is not None and len(segment_psds) >= 2

		# Test variance for a representative channel near the centre
		ch_freq = sc.channels[len(sc.channels) // 2]
		stddev = sc._segment_power_variance(ch_freq, segment_psds)

		# Stationary noise should have variance close to the natural sampling
		# variance of an 8-segment Welch PSD — well below 3 dB.
		assert stddev < 3.0, f"Expected stationary noise variance < 3 dB, got {stddev:.2f} dB"

	def test_variance_high_for_modulated_signal (self, scanner_instance):

		"""
		An amplitude-modulated signal should produce high temporal variance
		across segment PSDs as its envelope rises and falls.
		"""

		sc = scanner_instance

		# Build an AM-modulated carrier at a target channel frequency.
		# Choose a modulation rate slow enough that segments capture the
		# envelope at clearly different points (one full cycle across the slice).
		n = sc.samples_per_slice
		t = numpy.arange(n) / sc.sample_rate
		duration_s = n / sc.sample_rate
		audio_freq = 2.0 / duration_s  # 2 cycles across the slice
		mod_depth = 0.95
		envelope = 1.0 + mod_depth * numpy.sin(2.0 * numpy.pi * audio_freq * t)

		# Place at the first channel offset from centre to avoid DC mask
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		carrier = numpy.exp(2j * numpy.pi * offset_hz * t)
		iq = (envelope * carrier).astype(numpy.complex64)

		_, segment_psds = sc._calculate_psd_data(iq, include_segment_psd=True)
		assert segment_psds is not None and len(segment_psds) >= 2

		stddev = sc._segment_power_variance(ch_freq, segment_psds)

		# Heavy AM should produce variance well above the 3 dB discriminator
		assert stddev > 3.0, f"Expected modulated signal variance > 3 dB, got {stddev:.2f} dB"

	def test_variance_zero_for_too_few_segments (self, scanner_instance):

		"""With fewer than 2 segments, variance should return 0.0 (cannot compute)."""

		sc = scanner_instance
		ch_freq = sc.channels[0]

		assert sc._segment_power_variance(ch_freq, []) == 0.0
		assert sc._segment_power_variance(ch_freq, None) == 0.0
