"""Tests for transition detection and sample-level trimming."""

import numpy

import substation.constants
import substation.scanner


class TestRefineTrimOnAudio:

	def test_turn_on_trims_silence (self):
		"""Turn-ON should trim leading silence and keep padding around the first above-threshold sample."""
		audio = numpy.zeros(2000, dtype=numpy.float32)
		audio[500:1500] = 0.5  # signal starts at 500
		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)

		assert len(trimmed) < len(audio)
		# First above-threshold sample is at 500; up to TRIM_PRE_SAMPLES of pad is kept.
		# So the returned length is at most 1500 (2000 - (500 - TRIM_PRE_SAMPLES))
		# and at least 1500 (the 0.5 run itself).
		assert len(trimmed) >= 1500

	def test_turn_off_trims_trailing (self):
		"""Turn-OFF should trim trailing silence."""
		audio = numpy.zeros(2000, dtype=numpy.float32)
		audio[200:800] = 0.5  # signal ends at 800
		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=False)

		assert len(trimmed) < len(audio)
		assert len(trimmed) >= 800  # at least up to end of signal

	def test_all_silence (self):
		"""All-silence input returns the full audio unchanged."""
		audio = numpy.zeros(1000, dtype=numpy.float32)
		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		numpy.testing.assert_array_equal(trimmed, audio)

	def test_all_signal (self):
		"""All-signal input returns the full audio."""
		audio = numpy.ones(1000, dtype=numpy.float32) * 0.5
		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		assert len(trimmed) == len(audio)

	def test_short_padding_clamped (self):
		"""When signal starts near the beginning, padding is clamped to what's available."""
		audio = numpy.zeros(1000, dtype=numpy.float32)
		audio[10:500] = 0.5  # starts at sample 10

		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		# Pad is clamped to 10 (can't go before index 0), so trimmed starts at 0.
		assert len(trimmed) == 1000

	def test_empty_input (self):
		audio = numpy.array([], dtype=numpy.float32)
		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		assert len(trimmed) == 0

	def test_signal_at_exact_threshold (self):
		"""Signal exactly at threshold should be detected and the result trimmed."""
		threshold = substation.constants.TRIM_AMPLITUDE_THRESHOLD
		audio = numpy.zeros(1000, dtype=numpy.float32)
		audio[300:700] = threshold

		trimmed = substation.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		assert len(trimmed) < len(audio)


class TestFindTransitionIndex:

	def test_no_segment_psd_turn_on (self, scanner_instance):
		"""Without segment PSD, turn-ON returns 0 (use entire buffer)."""
		sc = scanner_instance
		iq = numpy.zeros(sc.samples_per_slice, dtype=numpy.complex64)
		idx = sc._find_transition_index(iq, sc.channels[0], turning_on=True, segment_psd=None, segment_noise_floors=None)
		assert idx == 0

	def test_no_segment_psd_turn_off (self, scanner_instance):
		"""Without segment PSD, turn-OFF returns len(samples)."""
		sc = scanner_instance
		iq = numpy.zeros(sc.samples_per_slice, dtype=numpy.complex64)
		idx = sc._find_transition_index(iq, sc.channels[0], turning_on=False, segment_psd=None, segment_noise_floors=None)
		assert idx == len(iq)
