"""Tests for transition detection and sample-level trimming."""

import numpy
import pytest

import sdr_scanner.constants
import sdr_scanner.scanner


class TestRefineTrimOnAudio:

	def test_turn_on_trims_silence (self):
		"""Turn-ON should trim leading silence and preserve padding."""
		audio = numpy.zeros(2000, dtype=numpy.float32)
		audio[500:1500] = 0.5  # signal starts at 500
		trimmed, pad = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		# Should have trimmed some leading silence
		assert len(trimmed) < len(audio)
		# Pad should be at most TRIM_PRE_SAMPLES
		assert pad <= sdr_scanner.constants.TRIM_PRE_SAMPLES
		assert pad > 0

	def test_turn_off_trims_trailing (self):
		"""Turn-OFF should trim trailing silence and preserve padding."""
		audio = numpy.zeros(2000, dtype=numpy.float32)
		audio[200:800] = 0.5  # signal ends at 800
		trimmed, pad = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=False)
		assert len(trimmed) < len(audio)
		assert pad <= sdr_scanner.constants.TRIM_POST_SAMPLES

	def test_all_silence (self):
		"""All-silence input returns the full audio with pad=0."""
		audio = numpy.zeros(1000, dtype=numpy.float32)
		trimmed, pad = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		numpy.testing.assert_array_equal(trimmed, audio)
		assert pad == 0

	def test_all_signal (self):
		"""All-signal input returns the full audio."""
		audio = numpy.ones(1000, dtype=numpy.float32) * 0.5
		trimmed_on, pad_on = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		assert len(trimmed_on) == len(audio)

	def test_short_padding_clamped (self):
		"""When signal starts near beginning, padding is clamped."""
		audio = numpy.zeros(1000, dtype=numpy.float32)
		audio[10:500] = 0.5  # starts at sample 10
		trimmed, pad = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		assert pad == 10  # can only pad 10 samples back

	def test_empty_input (self):
		audio = numpy.array([], dtype=numpy.float32)
		trimmed, pad = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		assert len(trimmed) == 0
		assert pad == 0

	def test_signal_at_exact_threshold (self):
		"""Signal exactly at threshold should be detected."""
		threshold = sdr_scanner.constants.TRIM_AMPLITUDE_THRESHOLD
		audio = numpy.zeros(1000, dtype=numpy.float32)
		audio[300:700] = threshold  # exactly at threshold
		trimmed, pad = sdr_scanner.scanner.RadioScanner._refine_trim_on_audio(audio, turning_on=True)
		# Should detect the signal
		assert len(trimmed) < len(audio) or pad > 0


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
