"""Tests for AM and NFM demodulation with synthetic IQ."""

import time

import numpy
import pytest
import scipy.fft
import scipy.signal

import substation.constants
import substation.dsp.demodulation
import substation.dsp.filters

import iq_generators


def _dominant_freq (audio: numpy.ndarray, sample_rate: int) -> float:
	"""Return the dominant frequency in the audio signal via FFT."""
	spectrum = numpy.abs(scipy.fft.rfft(audio))
	freqs = scipy.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
	# Ignore DC bin
	spectrum[0] = 0
	return float(freqs[numpy.argmax(spectrum)])


class TestNFMDemodulation:

	def test_recovers_tone (self):
		"""Demodulate a 1 kHz FM signal and verify the tone is present."""
		sr = 1_024_000
		audio_rate = 16000
		audio_freq = 1000.0
		deviation = 2500.0
		iq = iq_generators.generate_fm_iq(audio_freq, deviation, sr, 0.1)
		audio, state = substation.dsp.demodulation.demodulate_nfm(iq, sr, audio_rate)
		assert len(audio) > 0
		# Skip the first 20% to avoid filter transients
		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], audio_rate)
		assert abs(dominant - audio_freq) < 200  # within 200 Hz

	def test_state_continuity (self):
		"""Two consecutive blocks produce exactly what one pass over both produces.

		A threshold on the step at the join passed or failed by where the
		join fell on the waveform; comparing with one pass does not.
		"""
		sr = 1_024_000
		audio_rate = 16000
		iq = iq_generators.generate_fm_iq(1000.0, 2500.0, sr, 0.2)
		half = len(iq) // 2

		state = None
		audio_a, state = substation.dsp.demodulation.demodulate_nfm(iq[:half], sr, audio_rate, state=state)
		audio_b, state = substation.dsp.demodulation.demodulate_nfm(iq[half:], sr, audio_rate, state=state)
		joined = numpy.concatenate([audio_a, audio_b])

		whole, _ = substation.dsp.demodulation.demodulate_nfm(iq, sr, audio_rate, state=None)

		numpy.testing.assert_allclose(joined, whole[:len(joined)], atol=1e-5)

	def test_empty_input (self):
		audio, state = substation.dsp.demodulation.demodulate_nfm(
			numpy.array([], dtype=numpy.complex64), 1_024_000, 16000
		)
		assert len(audio) == 0

	def test_output_dtype (self):
		iq = iq_generators.generate_fm_iq(1000.0, 2500.0, 1_024_000, 0.05)
		audio, _ = substation.dsp.demodulation.demodulate_nfm(iq, 1_024_000, 16000)
		assert audio.dtype == numpy.float32


class TestAMDemodulation:

	def test_recovers_tone (self):
		"""Demodulate a 1 kHz AM signal and verify the tone is present."""
		sr = 1_024_000
		audio_rate = 16000
		audio_freq = 1000.0
		iq = iq_generators.generate_am_iq(audio_freq, 0.8, sr, 0.1)
		audio, state = substation.dsp.demodulation.demodulate_am(iq, sr, audio_rate)
		assert len(audio) > 0
		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], audio_rate)
		assert abs(dominant - audio_freq) < 200

	def test_empty_input (self):
		audio, state = substation.dsp.demodulation.demodulate_am(
			numpy.array([], dtype=numpy.complex64), 1_024_000, 16000
		)
		assert len(audio) == 0

	def test_output_range (self):
		"""AM output should be within [-1, 1] after AGC and clipping."""
		iq = iq_generators.generate_am_iq(1000.0, 0.8, 1_024_000, 0.1)
		audio, _ = substation.dsp.demodulation.demodulate_am(iq, 1_024_000, 16000)
		assert numpy.all(audio >= -1.0)
		assert numpy.all(audio <= 1.0)


class TestPickIfDecimation:

	"""
	Tests for the _pick_if_decimation helper.

	Regression tests for two separate issues:

	1. AirSpy R2 hang: with sample_rate=2500000 and audio_sample_rate=16000,
	   the previous naive round(sample_rate / target_if_rate) produced
	   if_decimation=39 → if_rate=64103 (coprime with 2500000) → a 50-million-
	   tap rational resampling filter that locked up the system.  The helper
	   must always pick a value that exactly divides sample_rate so the IF
	   decimation step uses fast integer downsampling.

	2. AirSpy HF+ block-boundary click: with sample_rate=912000 and
	   audio_sample_rate=16000, the earlier helper picked if_decimation=15
	   → if_rate=60800, which is an integer divisor of 912000 but NOT a
	   multiple of 16000.  That forced the downstream decimate_audio call
	   into its rational resample_poly path, which emits a short tail
	   transient at every block boundary and produces an audible ~5 Hz click
	   in recordings.  The helper must prefer "clean chain" candidates where
	   *both* sample_rate % d == 0 AND (sample_rate // d) % audio_sample_rate
	   == 0, so the audio stage stays on its stateful integer path.
	"""

	def test_airspy_r2_2_5mhz (self):
		"""
		AirSpy R2 native rate must produce a clean integer divisor.

		No clean-chain candidate exists for 2_500_000 → 16_000 (prime
		factorisations have incompatible power-of-two factors), so the
		helper falls back to any integer divisor of 2500000 closest to
		the ideal — which is still d=40.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(2_500_000, 16_000, 4.0)
		assert 2_500_000 % dec == 0, f"if_decimation {dec} must evenly divide 2500000"
		assert dec == 40

	def test_rtlsdr_1024khz_unchanged (self):
		"""
		RTL-SDR PMR config must still pick if_decimation=16 — 16 is both
		a divisor of 1024000 AND yields a clean 64000 → 16000 audio stage,
		so it's the clean-chain winner and matches the pre-change behaviour.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(1_024_000, 16_000, 4.0)
		assert dec == 16
		assert (1_024_000 // dec) % 16_000 == 0

	def test_hackrf_2_4mhz (self):
		"""
		HackRF 2.4 MHz path prefers d=30 (clean chain) over d=40 (dirty
		chain).  At ideal=38, clean candidates in the window are
		{25, 30, 50, 75}; d=30 is closest.  Yields if_rate=80000 which
		divides cleanly into 16000 (factor 5).
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(2_400_000, 16_000, 4.0)
		assert 2_400_000 % dec == 0
		assert (2_400_000 // dec) % 16_000 == 0, \
			f"if_rate {2_400_000 // dec} must be a multiple of 16000 (clean chain)"
		assert dec == 30

	def test_hackrf_dmr_12_5mhz (self):
		"""
		HackRF DMR wide-band path picks a clean integer divisor.

		No clean-chain candidate exists for 12_500_000 → 16_000 (same
		power-of-two mismatch as AirSpy R2), so the helper falls back to
		d=200 — still the closest integer divisor to the ideal.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(12_500_000, 16_000, 4.0)
		assert 12_500_000 % dec == 0
		assert dec == 200

	def test_airspy_hf_912khz_clean_chain (self):
		"""
		AirSpy HF+ Discovery native rate picks d=19, yielding if_rate=48000
		which is exactly 3 × 16000.  This is the fix for the block-boundary
		click on the air_civil_bristol_airspyhf band: before the clean-chain
		preference was added, the helper picked d=15 (closer to the ideal
		14 but NOT a clean chain), and the audio decimation stage fell into
		the rational resample_poly path with its tail transient.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(912_000, 16_000, 4.0)
		assert dec == 19
		assert 912_000 % dec == 0
		if_rate = 912_000 // dec
		assert if_rate == 48_000
		assert if_rate % 16_000 == 0, \
			"if_rate must be a multiple of audio_sample_rate so the audio stage stays on the integer path"

	def test_clean_chain_is_preferred_over_closer_dirty_chain (self):
		"""
		Proves the clean-chain tier genuinely wins over the nearest-divisor
		tier when both are available.  For 912_000 → 16_000, the absolute
		closest integer divisor to ideal=14 is d=15 (distance 1), but d=15
		gives the dirty if_rate=60800 (not a multiple of 16000).  The
		clean-chain candidate d=19 is further from the ideal (distance 5)
		but gives the clean if_rate=48000.  The helper must pick d=19.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(912_000, 16_000, 4.0)
		# If the clean-chain preference were absent, we'd expect 15 here.
		assert dec != 15, "clean-chain preference must override nearest-divisor"
		assert dec == 19

	def test_returns_at_least_one (self):
		"""For very low sample rates, the helper must never return 0."""
		dec = substation.dsp.demodulation._pick_if_decimation(48_000, 16_000, 4.0)
		assert dec >= 1


class TestNFMAt2_5MHz:

	"""
	Regression tests proving the AirSpy R2 NFM hang is fixed.  Each test
	caps execution time at 1 second — the previously-broken path tried to
	allocate a 50-million-tap filter (~400 MB) inside scipy.signal.resample_poly
	and could lock up a 2 GB Pi for minutes (or OOM-kill it).
	"""

	def test_nfm_2_5mhz_completes_quickly (self):
		"""demodulate_nfm at 2.5 MHz must process a 200ms slice in well under 1s."""
		sr = 2_500_000
		asr = 16_000
		duration_s = 0.2

		# Generate a simple FM-modulated 1 kHz tone
		n = int(sr * duration_s)
		t = numpy.arange(n, dtype=numpy.float64) / sr
		audio_freq = 1000.0
		deviation = 2500.0
		phase = 2 * numpy.pi * deviation * numpy.cumsum(numpy.sin(2 * numpy.pi * audio_freq * t)) / sr
		iq = numpy.exp(1j * phase).astype(numpy.complex64)

		start = time.perf_counter()
		audio, _ = substation.dsp.demodulation.demodulate_nfm(iq, sr, asr)
		elapsed = time.perf_counter() - start

		assert elapsed < 1.0, f"demodulate_nfm took {elapsed:.2f}s — should be well under 1s"
		assert len(audio) > 0
		# Sanity check the recovered tone is in the right ballpark
		dominant = _dominant_freq(audio[len(audio) // 5:], asr)
		assert abs(dominant - audio_freq) < 200

	def test_am_2_5mhz_completes_quickly (self):
		"""demodulate_am at 2.5 MHz must also be fast (now uses the same IF helper)."""
		sr = 2_500_000
		asr = 16_000
		iq = iq_generators.generate_am_iq(1000.0, 0.8, sr, 0.2)

		start = time.perf_counter()
		audio, _ = substation.dsp.demodulation.demodulate_am(iq, sr, asr)
		elapsed = time.perf_counter() - start

		assert elapsed < 1.0, f"demodulate_am took {elapsed:.2f}s — should be well under 1s"
		assert len(audio) > 0


class TestRationalResampleContinuity:

	"""
	Verify the streaming polyphase resampler produces seamless audio across
	block boundaries.  The old resample_poly overlap-save approach drifted
	by ~0.74 output samples per block (cumulative phase error of ~27°/block
	at 1 kHz) and had a 3 dB SNR vs. whole-signal processing.
	"""

	IF_RATE = 62500
	AUDIO_RATE = 16000
	BLOCK_SIZE = 13107

	def test_blockwise_matches_whole_signal (self):
		"""Block-by-block output must be bit-identical to whole-signal."""
		n_blocks = 10
		total = self.BLOCK_SIZE * n_blocks
		t = numpy.arange(total) / self.IF_RATE
		signal = numpy.sin(2 * numpy.pi * 1000 * t).astype(numpy.float32)

		state = {}
		blocks = []
		for b in range(n_blocks):
			blk = signal[b * self.BLOCK_SIZE:(b + 1) * self.BLOCK_SIZE]
			out, state = substation.dsp.filters.decimate_audio(blk, self.IF_RATE, self.AUDIO_RATE, state)
			blocks.append(out)
		blockwise = numpy.concatenate(blocks)

		whole_state: dict = {}
		whole, _ = substation.dsp.filters.decimate_audio(signal, self.IF_RATE, self.AUDIO_RATE, whole_state)

		n = min(len(blockwise), len(whole))
		numpy.testing.assert_array_equal(blockwise[:n], whole[:n])

	def test_phase_continuity_at_boundaries (self):
		"""Phase jumps at block boundaries must be < 1 degree."""
		n_blocks = 10
		total = self.BLOCK_SIZE * n_blocks
		t = numpy.arange(total) / self.IF_RATE
		signal = numpy.sin(2 * numpy.pi * 1000 * t).astype(numpy.float32)

		state = {}
		blocks = []
		for b in range(n_blocks):
			blk = signal[b * self.BLOCK_SIZE:(b + 1) * self.BLOCK_SIZE]
			out, state = substation.dsp.filters.decimate_audio(blk, self.IF_RATE, self.AUDIO_RATE, state)
			blocks.append(out)
		audio = numpy.concatenate(blocks)

		boundaries = numpy.cumsum([len(b) for b in blocks[:-1]])
		for b in boundaries:
			if b < 50 or b + 50 >= len(audio):
				continue
			seg = audio[b - 50:b + 50]
			analytic = scipy.signal.hilbert(seg)
			phases = numpy.unwrap(numpy.angle(analytic))
			jump = abs(phases[51] - phases[50] - (phases[50] - phases[49]))
			assert jump < numpy.radians(1.0), (
				f"Phase jump {numpy.degrees(jump):.1f}° at boundary sample {b} exceeds 1°"
			)

	def test_unity_passband_gain (self):
		"""DC and voice-band signals must pass at unity gain."""
		dc = numpy.ones(50000, dtype=numpy.float32)
		state: dict = {}
		dc_out, _ = substation.dsp.filters.decimate_audio(dc, self.IF_RATE, self.AUDIO_RATE, state)
		assert abs(dc_out[200:].mean() - 1.0) < 0.01

	def test_complex_signal_supported (self):
		"""The rational path must handle complex64 IQ (used by decimate_iq)."""
		t = numpy.arange(self.BLOCK_SIZE * 3) / self.IF_RATE
		iq = numpy.exp(1j * 2 * numpy.pi * 500 * t).astype(numpy.complex64)
		state: dict = {}
		blocks = []
		for b in range(3):
			blk = iq[b * self.BLOCK_SIZE:(b + 1) * self.BLOCK_SIZE]
			out, state = substation.dsp.filters.decimate_iq(blk, self.IF_RATE, self.AUDIO_RATE, state)
			blocks.append(out)
		result = numpy.concatenate(blocks)
		assert numpy.iscomplexobj(result)
		assert len(result) > 0


class TestDemodulatorsDict:

	def test_keys (self):
		assert "NFM" in substation.dsp.demodulation.DEMODULATORS
		assert "AM" in substation.dsp.demodulation.DEMODULATORS
		assert "USB" in substation.dsp.demodulation.DEMODULATORS
		assert "LSB" in substation.dsp.demodulation.DEMODULATORS

	def test_callable (self):
		for key, func in substation.dsp.demodulation.DEMODULATORS.items():
			assert callable(func)


def _ssb_iq_tone (audio_freq: float, sample_rate: int, duration_s: float) -> numpy.ndarray:

	"""
	Synthesise the IQ baseband for a single SSB tone.

	A USB transmission containing a tone at +audio_freq Hz produces a
	complex sinusoid at +audio_freq in the IQ baseband (the carrier is
	implicit at 0 Hz).  An LSB transmission of the same tone produces
	a complex sinusoid at -audio_freq.
	"""

	t = numpy.arange(int(sample_rate * duration_s), dtype=numpy.float64) / sample_rate
	return numpy.exp(1j * 2 * numpy.pi * audio_freq * t).astype(numpy.complex64)


class TestSSBDemodulation:

	"""Tests for the Weaver-method SSB demodulator."""

	def test_usb_recovers_audio_tone (self):

		"""USB demodulation of a +1 kHz IQ tone yields audio at 1 kHz."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(+1000.0, sr, 1.0)
		audio, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)

		# Skip the filter transient at the start
		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], asr)

		assert abs(dominant - 1000.0) < 50, f"Expected ~1000 Hz, got {dominant} Hz"

	def test_lsb_recovers_audio_tone (self):

		"""LSB demodulation of a -1 kHz IQ tone yields audio at 1 kHz."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(-1000.0, sr, 1.0)
		audio, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq, sr, asr)

		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], asr)

		assert abs(dominant - 1000.0) < 50, f"Expected ~1000 Hz, got {dominant} Hz"

	def test_usb_rejects_lsb_tone (self):

		"""Demodulating a +1 kHz IQ tone (USB content) as LSB should
		produce strongly attenuated audio compared to demodulating as USB.

		The test adds a small amount of background noise to the input.
		With pure tones the post-AGC FFT bin ratio is meaningless because
		the AGC normalises any residual to fill the dynamic range; with
		realistic noise the right-sideband demod produces a clean tone
		while the wrong-sideband demod produces noise-dominated output.
		"""

		sr = 192_000
		asr = 16_000
		numpy.random.seed(42)

		iq = _ssb_iq_tone(+1000.0, sr, 1.0)
		noise = (numpy.random.randn(len(iq)) + 1j * numpy.random.randn(len(iq))).astype(numpy.complex64) * 0.01
		iq_noisy = (iq + noise).astype(numpy.complex64)

		audio_right, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq_noisy, sr, asr)
		audio_wrong, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq_noisy, sr, asr)

		# Use the steady-state second half to avoid filter transients
		half = len(audio_right) // 2
		right_ss = audio_right[half:]
		wrong_ss = audio_wrong[half:]

		# Tone-to-rest ratio: dominant bin power vs everything else
		def tone_ratio_db (audio, target_hz, asr):
			fft = numpy.abs(scipy.fft.rfft(audio))
			freqs = scipy.fft.rfftfreq(len(audio), 1.0 / asr)
			bin_idx = int(numpy.argmin(numpy.abs(freqs - target_hz)))
			tone = fft[bin_idx] ** 2
			rest = numpy.sum(fft ** 2) - tone
			return 10 * numpy.log10(tone / max(rest, 1e-30))

		right_db = tone_ratio_db(right_ss, 1000.0, asr)
		wrong_db = tone_ratio_db(wrong_ss, 1000.0, asr)

		# Right sideband should have a much cleaner tone than wrong
		assert right_db > wrong_db + 15.0, (
			f"Sideband rejection too weak: right={right_db:.1f} dB, wrong={wrong_db:.1f} dB"
		)

	def test_lsb_rejects_usb_tone (self):

		"""Symmetric: a -1 kHz LSB tone demodulated as USB should be
		much weaker than the same tone demodulated as LSB."""

		sr = 192_000
		asr = 16_000
		numpy.random.seed(43)

		iq = _ssb_iq_tone(-1000.0, sr, 1.0)
		noise = (numpy.random.randn(len(iq)) + 1j * numpy.random.randn(len(iq))).astype(numpy.complex64) * 0.01
		iq_noisy = (iq + noise).astype(numpy.complex64)

		audio_right, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq_noisy, sr, asr)
		audio_wrong, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq_noisy, sr, asr)

		half = len(audio_right) // 2
		right_ss = audio_right[half:]
		wrong_ss = audio_wrong[half:]

		def tone_ratio_db (audio, target_hz, asr):
			fft = numpy.abs(scipy.fft.rfft(audio))
			freqs = scipy.fft.rfftfreq(len(audio), 1.0 / asr)
			bin_idx = int(numpy.argmin(numpy.abs(freqs - target_hz)))
			tone = fft[bin_idx] ** 2
			rest = numpy.sum(fft ** 2) - tone
			return 10 * numpy.log10(tone / max(rest, 1e-30))

		right_db = tone_ratio_db(right_ss, 1000.0, asr)
		wrong_db = tone_ratio_db(wrong_ss, 1000.0, asr)

		assert right_db > wrong_db + 15.0, (
			f"Sideband rejection too weak: right={right_db:.1f} dB, wrong={wrong_db:.1f} dB"
		)

	def test_state_continuity_across_blocks (self):

		"""Feeding a long signal in one chunk vs two halves should
		produce equivalent audio after the filter transient — proves
		that the per-block oscillator phase and filter state are
		preserved correctly."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(+1500.0, sr, 0.5)

		# Single-shot
		one_shot, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)

		# Two-half: feed first half, then second half, share state
		half = len(iq) // 2
		first, state = substation.dsp.demodulation.DEMODULATORS['USB'](iq[:half], sr, asr)
		second, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq[half:], sr, asr, state=state)
		concat = numpy.concatenate([first, second])

		# Both should be the same length
		assert len(one_shot) == len(concat), f"Lengths differ: {len(one_shot)} vs {len(concat)}"

		# After the initial transient, the two should agree closely.
		# Skip the first 20% to bypass the filter warm-up region.
		settle = len(one_shot) // 5
		diff_rms = numpy.sqrt(numpy.mean((one_shot[settle:] - concat[settle:]) ** 2))
		one_shot_rms = numpy.sqrt(numpy.mean(one_shot[settle:] ** 2))

		# Allow up to 5% RMS divergence — looser than NFM's continuity
		# test because the AGC level estimate carries some lag across
		# the block boundary, but tight enough to catch real bugs like
		# a missing phase update or filter state reset.
		assert diff_rms < 0.05 * one_shot_rms, (
			f"State continuity violated: diff RMS {diff_rms:.4f} vs signal RMS {one_shot_rms:.4f}"
		)

	def test_zero_input_returns_zero (self):

		"""Empty IQ → empty audio, no exceptions."""

		sr = 192_000
		asr = 16_000
		iq = numpy.array([], dtype=numpy.complex64)

		audio_usb, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)
		audio_lsb, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq, sr, asr)

		assert audio_usb.size == 0
		assert audio_lsb.size == 0

	def test_silent_input_returns_silent (self):

		"""Zero-amplitude IQ → zero-amplitude audio, no NaNs."""

		sr = 192_000
		asr = 16_000
		iq = numpy.zeros(int(sr * 0.1), dtype=numpy.complex64)

		audio, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)

		assert not numpy.any(numpy.isnan(audio))
		# AGC has a non-zero floor so the output is bounded but small
		assert numpy.max(numpy.abs(audio)) < 0.5

	def test_invalid_sideband_raises (self):

		"""demodulate_ssb with an invalid sideband string raises ValueError."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(+1000.0, sr, 0.1)

		with pytest.raises(ValueError, match="sideband"):
			substation.dsp.demodulation.demodulate_ssb(iq, sr, asr, sideband='WSB')


class TestHampelBlanker:

	def test_clean_signal_unchanged (self):
		"""A smooth signal with no outliers passes through unmodified, half a window late."""
		hw = substation.dsp.demodulation._BLANKER_HALF_WIN
		signal = numpy.sin(numpy.linspace(0, 10 * numpy.pi, 1000)).astype(numpy.float32)
		state: dict = {}
		result = substation.dsp.demodulation._blanker_hampel(signal, state)
		assert len(result) == len(signal)
		numpy.testing.assert_allclose(result[hw:], signal[:-hw], atol=1e-6)
		numpy.testing.assert_allclose(result[:hw], 0.0)

	def test_spikes_removed (self):
		"""Injected impulse spikes should be replaced with local median."""
		signal = numpy.sin(numpy.linspace(0, 10 * numpy.pi, 1000)).astype(numpy.float32)
		spiked = signal.copy()
		spike_positions = [100, 300, 500, 700]
		for pos in spike_positions:
			spiked[pos] = 5.0  # huge outlier vs ~1.0 amplitude
		state: dict = {}
		hw = substation.dsp.demodulation._BLANKER_HALF_WIN
		result = substation.dsp.demodulation._blanker_hampel(spiked, state)[hw:]
		# Spikes should be suppressed — result should be close to original
		for pos in spike_positions:
			assert abs(result[pos]) < 2.0, f"Spike at {pos} not suppressed: {result[pos]}"
		# Non-spike samples should be unchanged
		mask = numpy.ones(len(result), dtype=bool)
		for pos in spike_positions:
			mask[max(0, pos-1):pos+2] = False
		numpy.testing.assert_allclose(result[mask], signal[:len(result)][mask], atol=1e-6)

	def test_glitch_at_the_end_of_a_block_is_caught (self):
		"""Regression: a two-sample glitch in a block's last three samples passed the blanker.

		They were judged at once, with mirrored padding for the neighbours
		still to come, so the glitch counted as its own neighbours.
		"""
		signal = (0.3 * numpy.sin(numpy.linspace(0, 20 * numpy.pi, 800))).astype(numpy.float32)
		signal[398:400] = 3.0
		state: dict = {}

		out = numpy.concatenate([
			substation.dsp.demodulation._blanker_hampel(signal[:400], state),
			substation.dsp.demodulation._blanker_hampel(signal[400:], state),
		])

		assert numpy.max(numpy.abs(out[390:410])) < 1.0

	def test_blocks_give_what_one_pass_gives (self):
		"""Blanking in blocks gives the same output as blanking the whole signal at once."""
		rng = numpy.random.default_rng(2)
		signal = (0.3 * numpy.sin(numpy.linspace(0, 40 * numpy.pi, 2000)) + 0.01 * rng.standard_normal(2000)).astype(numpy.float32)
		signal[rng.choice(2000, 30, replace=False)] = 4.0
		# Two-sample glitches at block ends, where mirrored padding let them through
		for end in (333, 666, 999):
			signal[end - 2:end] = 4.0

		whole = substation.dsp.demodulation._blanker_hampel(signal, {})
		state: dict = {}
		pieces = numpy.concatenate([substation.dsp.demodulation._blanker_hampel(signal[i:i + 333], state) for i in range(0, 2000, 333)])

		numpy.testing.assert_allclose(pieces, whole)


class TestCTCSSDetection:

	def test_detects_known_tone (self):
		"""A clean CTCSS tone should be detected correctly."""
		sr = 16000
		duration = 0.3  # 300ms of audio
		t = numpy.arange(int(sr * duration)) / sr
		# 88.5 Hz CTCSS tone mixed with voice-band content
		audio = (
			numpy.sin(2 * numpy.pi * 88.5 * t) * 0.1 +
			numpy.sin(2 * numpy.pi * 1000 * t) * 0.3
		).astype(numpy.float32)
		result = substation.dsp.demodulation.detect_ctcss(audio, sr)
		assert result == 88.5

	def test_no_tone_returns_none (self):
		"""Voice-only audio (no subaudible tone) should return None."""
		sr = 16000
		t = numpy.arange(int(sr * 0.3)) / sr
		audio = (numpy.sin(2 * numpy.pi * 1000 * t) * 0.5).astype(numpy.float32)
		result = substation.dsp.demodulation.detect_ctcss(audio, sr)
		assert result is None

	def test_noise_returns_none (self):
		"""Random noise should not trigger false CTCSS detection."""
		sr = 16000
		audio = numpy.random.RandomState(42).randn(int(sr * 0.3)).astype(numpy.float32) * 0.1
		result = substation.dsp.demodulation.detect_ctcss(audio, sr)
		assert result is None

	def test_distinguishes_adjacent_tones (self):
		"""Should distinguish 67.0 Hz from 69.3 Hz (closest pair, 2.3 Hz apart)."""
		sr = 16000
		t = numpy.arange(int(sr * 0.3)) / sr
		for freq in (67.0, 69.3):
			audio = (numpy.sin(2 * numpy.pi * freq * t) * 0.1).astype(numpy.float32)
			result = substation.dsp.demodulation.detect_ctcss(audio, sr)
			assert result == freq, f"Expected {freq}, got {result}"


# The Golay(23,12) generator DCS uses, x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1,
# as its exponents.  The tests build DCS words with their own long division
# rather than the decoder's constant, so a wrong decoder cannot also make the
# signals it is tested with.
GOLAY_GENERATOR_EXPONENTS = (11, 10, 6, 5, 4, 2, 0)


def _gf2_remainder (coefficients: list[int]) -> list[int]:

	"""Remainder of a polynomial over GF(2), coefficients[i] being that of x^i, divided by the Golay generator."""

	degree = max(GOLAY_GENERATOR_EXPONENTS)
	remainder = list(coefficients)

	for power in range(len(remainder) - 1, degree - 1, -1):
		if remainder[power]:
			for exponent in GOLAY_GENERATOR_EXPONENTS:
				remainder[power - degree + exponent] ^= 1

	return remainder[:degree]


def dcs_word (code: int) -> int:

	"""
	The 23-bit DCS word for a 9-bit code, in detect_dcs's window layout.

	The code, then the 100 filler, fill bits 0-11 (the first sent), and the
	parity fills bits 12-22, so that the word is a multiple of the generator.
	With x^23 = 1 for this code, that parity is the data times x^11, modulo
	the generator.
	"""

	data = (code & 0x1FF) | (0b100 << 9)
	data_bits = [(data >> i) & 1 for i in range(12)]
	parity_bits = _gf2_remainder([0] * 11 + data_bits)
	word = data | sum(bit << (12 + i) for i, bit in enumerate(parity_bits))

	assert not any(_gf2_remainder([(word >> i) & 1 for i in range(23)]))
	return word


def dcs_audio (code: int, sr: int = 16000, duration: float = 1.0) -> numpy.typing.NDArray[numpy.float32]:

	"""A DCS bitstream at 134.3 bps, repeating the code's word, sent from bit 0, as NRZ audio."""

	word = dcs_word(code)
	samples_per_bit = sr / substation.constants.DCS_BITRATE
	bit_positions = (numpy.arange(int(sr * duration)) / samples_per_bit).astype(int) % 23
	bits = numpy.array([(word >> i) & 1 for i in range(23)])[bit_positions]
	return numpy.where(bits == 1, 0.1, -0.1).astype(numpy.float32)


def _rotate (word: int, places: int) -> int:

	"""Rotate a 23-bit word, as a receiver sees the endless stream from another starting bit."""

	return ((word << places) | (word >> (23 - places))) & 0x7FFFFF


class TestDCSDetection:

	def test_words_match_the_textbook_equivalents (self):
		"""The test's own encoder agrees with the standard: DCS 023, 340 and 766 are rotations of one word."""
		word = dcs_word(0o023)
		aligned = {_rotate(word, k) & 0x1FF for k in range(23) if (_rotate(word, k) >> 9) & 0x07 == 4}

		assert aligned == {0o023, 0o340, 0o766}

	def test_every_standard_code_decodes (self):
		"""Regression: fed real DCS words, the decoder got 19 of the 104 standard codes right.

		Its parity table was not the Golay code DCS uses, and the old tests
		built their signals from that same table, so they could not see it.
		"""
		for code in substation.constants.DCS_STANDARD_CODES:
			assert substation.dsp.demodulation._golay2312_decode(dcs_word(code)) == code, f"{code:03o}"

	def test_up_to_three_flipped_bits_are_corrected (self):
		"""Golay(23,12) corrects any three bit errors, in the data or the parity."""
		rng = numpy.random.default_rng(7)

		for code in (0o023, 0o125, 0o631, 0o754):
			for flips in (1, 2, 3):
				for _ in range(20):
					positions = rng.choice(23, size=flips, replace=False)
					corrupted = dcs_word(code) ^ sum(1 << int(position) for position in positions)
					assert substation.dsp.demodulation._golay2312_decode(corrupted) == code

	def test_noise_returns_none (self):
		"""Random noise should not trigger false DCS detection."""
		sr = 16000

		for seed in range(20):
			audio = numpy.random.RandomState(seed).randn(int(sr * 1.0)).astype(numpy.float32) * 0.1
			assert substation.dsp.demodulation.detect_dcs(audio, sr) is None

	def test_detects_known_codes (self):
		"""End-to-end: a real DCS bitstream decodes to the transmitted code."""
		codes = sorted(substation.constants.DCS_STANDARD_CODES)

		for code in codes[::9]:
			result = substation.dsp.demodulation.detect_dcs(dcs_audio(code), 16000)
			assert result == code, f"Expected {code:03o}, got {result if result is None else format(result, '03o')}"

	def test_detects_code_under_voice (self):
		"""DCS detection must survive voice content above the subaudible band."""
		sr = 16000
		t = numpy.arange(sr) / sr
		voice = (0.3 * numpy.sin(2 * numpy.pi * 1000 * t)).astype(numpy.float32)

		assert substation.dsp.demodulation.detect_dcs(dcs_audio(0o023) + voice, sr) == 0o023

	def test_equivalent_code_reports_its_standard_code (self):
		"""A radio sending 340, which is 023's word read from another bit, is reported as 023.

		The stream has no frame marker, so a receiver cannot tell the two
		apart; the standard list holds one code of each rotation class.
		"""
		assert 0o340 not in substation.constants.DCS_STANDARD_CODES

		assert substation.dsp.demodulation.detect_dcs(dcs_audio(0o340), 16000) == 0o023


class TestNfmCarrier:

	@pytest.mark.parametrize("offset_hz", [0.0, 2.0, 100.0])
	def test_unmodulated_carrier_demodulates_to_silence (self, offset_hz):
		"""Regression: a carrier on the radio channel's frequency demodulated to loud noise.

		The demodulator subtracted each block's mean from the radio channel's
		IQ, which removes a carrier within a few Hz of the frequency, so the
		audio silence timeout could not end the recording of a keyed but
		silent transmitter.
		"""
		if_rate = 256000
		t = numpy.arange(if_rate * 2) / if_rate
		rng = numpy.random.default_rng(0)
		noise = 0.001 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
		iq = (0.3 * numpy.exp(2j * numpy.pi * offset_hz * t) + noise).astype(numpy.complex64)

		state = None
		blocks = []
		for start in range(0, len(iq), len(iq) // 4):
			audio, state = substation.dsp.demodulation.demodulate_nfm(iq[start:start + len(iq) // 4], if_rate, 16000, state)
			blocks.append(audio)

		audio = numpy.concatenate(blocks[1:])
		assert numpy.sqrt(numpy.mean(audio ** 2)) < substation.constants.AUDIO_SILENCE_RMS_THRESHOLD


class TestVoiceAgc:

	@staticmethod
	def _syllabic_audio (seconds: float = 3.0, sr: int = 16000) -> numpy.typing.NDArray[numpy.float32]:
		"""Band-limited noise with a 4 Hz syllable envelope, the case that showed the old AGC's steps."""
		rng = numpy.random.default_rng(5)
		n = int(seconds * sr)
		noise = scipy.signal.sosfilt(scipy.signal.butter(4, [300, 3000], btype='bandpass', fs=sr, output='sos'), rng.standard_normal(n))
		envelope = 0.55 + 0.45 * numpy.sin(2 * numpy.pi * 4 * numpy.arange(n) / sr)
		return (0.2 * noise * envelope).astype(numpy.float32)

	def test_blocks_give_what_one_pass_gives (self):
		"""Regression: the AGC recomputed centred windows per block, so its gain stepped by 2-8 dB at every block join.

		Processed in 256 ms blocks, as the scanner hands them over, the
		output must equal processing the whole recording at once.
		"""
		audio = self._syllabic_audio()
		whole = substation.dsp.demodulation._apply_voice_agc(audio.copy(), 16000, {}, 'test_')

		state: dict = {}
		block = 4096
		pieces = [substation.dsp.demodulation._apply_voice_agc(audio[i:i + block].copy(), 16000, state, 'test_') for i in range(0, len(audio), block)]

		numpy.testing.assert_allclose(numpy.concatenate(pieces), whole, rtol=1e-5, atol=1e-7)

	def test_gain_does_not_drop_ahead_of_a_loud_onset (self):
		"""Regression: the centred windows looked ahead, so the gain ducked before a loud onset."""
		sr = 16000
		quiet = (0.01 * numpy.sin(2 * numpy.pi * 500 * numpy.arange(sr) / sr)).astype(numpy.float32)
		with_onset = quiet.copy()
		with_onset[sr // 2:] *= 50.0

		before = substation.dsp.demodulation._apply_voice_agc(quiet.copy(), sr, {}, 'test_')
		after = substation.dsp.demodulation._apply_voice_agc(with_onset.copy(), sr, {}, 'test_')

		numpy.testing.assert_allclose(after[:sr // 2], before[:sr // 2], rtol=1e-6)

	def test_output_never_exceeds_the_output_gain (self):
		"""The level rises at once to every new peak, so the output cannot overshoot."""
		audio = self._syllabic_audio()
		audio[20000:20100] *= 20.0

		out = substation.dsp.demodulation._apply_voice_agc(audio, 16000, {}, 'test_')

		assert numpy.max(numpy.abs(out)) <= substation.constants.AM_OUTPUT_GAIN + 1e-6


class TestVoiceBandpass:

	def test_ctcss_tone_removed (self):
		"""NFM demodulation removes a low CTCSS tone from the audio it returns, through its own voice band-pass."""
		if_rate = 256000
		audio_rate = 16000
		block = if_rate // 2
		t = numpy.arange(3 * block) / if_rate
		message = 0.3 * numpy.sin(2 * numpy.pi * 1000 * t) + 0.1 * numpy.sin(2 * numpy.pi * 88.5 * t)
		phase = 2 * numpy.pi * (2500.0 / 0.4) * numpy.cumsum(message) / if_rate
		iq = numpy.exp(1j * phase).astype(numpy.complex64)

		state = None
		blocks = []
		for start in range(0, 3 * block, block):
			audio, state = substation.dsp.demodulation.demodulate_nfm(iq[start:start + block], if_rate, audio_rate, state)
			blocks.append(audio)

		# The first block is where tones are detected, before the band-pass settles
		filtered = numpy.concatenate(blocks[1:])
		spectrum = numpy.abs(scipy.fft.rfft(filtered * numpy.hanning(len(filtered))))
		freqs = scipy.fft.rfftfreq(len(filtered), d=1.0 / audio_rate)
		ctcss_bin = numpy.argmin(numpy.abs(freqs - 88.5))
		voice_bin = numpy.argmin(numpy.abs(freqs - 1000))

		# CTCSS should be at least 20 dB below the voice tone
		assert spectrum[voice_bin] > spectrum[ctcss_bin] * 10

	def test_state_continuity_across_blocks (self):
		"""Spikes at block boundaries should be detected using cross-block state."""
		signal = numpy.zeros(200, dtype=numpy.float32)
		# Spike right at the start of block 2
		block1 = signal[:100].copy()
		block2 = signal[100:].copy()
		block2[0] = 5.0  # spike at first sample of block 2
		state: dict = {}
		hw = substation.dsp.demodulation._BLANKER_HALF_WIN
		substation.dsp.demodulation._blanker_hampel(block1, state)
		result2 = substation.dsp.demodulation._blanker_hampel(block2, state)
		assert abs(result2[hw]) < 1.0, f"Spike at block boundary not suppressed: {result2[hw]}"

	def test_empty_signal (self):
		"""Empty input should return empty output without error."""
		state: dict = {}
		result = substation.dsp.demodulation._blanker_hampel(numpy.array([], dtype=numpy.float32), state)
		assert len(result) == 0
