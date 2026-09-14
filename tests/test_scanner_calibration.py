"""Tests for frequency calibration: when a measured PPM correction is applied, and when it is refused."""

import time
import unittest.mock

import numpy
import pytest

import substation.constants
import substation.scanner


class FakeCalibrationSdr:

	"""A receiver that supplies synthetic IQ samples at the calibration frequency.

	With station_offset_ppm set it delivers a clean carrier that many PPM from
	where it should be, standing in for a station seen through a crystal error.
	Its correction behaves as measured on an RTL-SDR Blog V4: each PPM of
	correction moves the station one PPM further in the same direction.
	Without a station the samples are noise only, standing in for a
	calibration frequency with nothing on it.
	"""

	def __init__ (self, seed: int, station_offset_ppm: float | None = None, fail_after_reads: int | None = None) -> None:

		"""Set the starting tuning and choose what the receiver will deliver."""

		self.center_freq = 446.1e6
		self.sample_rate = 2.4e6
		self.freq_correction = 0

		self._rng = numpy.random.default_rng(seed)
		self._station_offset_ppm = station_offset_ppm
		self._fail_after_reads = fail_after_reads
		self._reads = 0
		self._sample_index = 0

	def read_samples (self, n: int) -> numpy.ndarray:

		"""Return n IQ samples of noise, plus the carrier when one is configured."""

		if self._fail_after_reads is not None and self._reads >= self._fail_after_reads:
			raise OSError("USB read failed")

		self._reads += 1

		samples = (self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)) / numpy.sqrt(2)

		if self._station_offset_ppm is not None:
			apparent_offset_hz = self.center_freq * (self._station_offset_ppm + self.freq_correction) * 1e-6
			t = (numpy.arange(n) + self._sample_index) / self.sample_rate
			samples += 10 * numpy.exp(2j * numpy.pi * apparent_offset_hz * t)

		self._sample_index += n

		return samples.astype(numpy.complex64)


def _calibrate (sdr: FakeCalibrationSdr, known_freq: float = 93.7e6) -> None:

	"""Run the real calibration method against a fake receiver, without its settling delays."""

	scanner = substation.scanner.RadioScanner.__new__(substation.scanner.RadioScanner)
	scanner.sdr = sdr

	with unittest.mock.patch.object(time, "sleep"):
		scanner._calibrate_sdr(known_freq)


class TestEvaluateCalibration:

	def test_consistent_strong_measurements_are_applied (self):
		"""A station that appears 30 PPM high, consistently, needs a correction of -30."""
		correction, reason = substation.scanner.RadioScanner._evaluate_calibration([29.6, 30.1, 30.0, 30.4, 29.9], 35.0)

		assert correction == -30
		assert reason == ""

	def test_a_minority_of_outliers_does_not_block_the_correction (self):
		"""Two wild readings among ten are removed before the spread is judged."""
		measurements = [30.0, 30.2, 29.8, 30.1, 29.9, 30.0, 30.3, 29.7, 410.0, -350.0]

		correction, reason = substation.scanner.RadioScanner._evaluate_calibration(measurements, 35.0)

		assert correction == -30
		assert reason == ""

	def test_scattered_measurements_are_refused (self):
		"""Readings scattered across hundreds of PPM, as noise produces, are not applied."""
		measurements = list(numpy.random.default_rng(0).uniform(-530.0, 530.0, 10))

		correction, reason = substation.scanner.RadioScanner._evaluate_calibration(measurements, 16.0)

		assert correction is None
		assert "disagree" in reason

	def test_weak_signal_is_refused (self):
		"""Consistent readings below the signal strength limit are not applied."""
		weak_db = substation.constants.CALIBRATION_MIN_SIGNAL_DB - 1.0

		correction, reason = substation.scanner.RadioScanner._evaluate_calibration([12.0, 12.1, 11.9], weak_db)

		assert correction is None
		assert "too weak" in reason

	def test_implausibly_large_correction_is_refused (self):
		"""A consistent correction beyond what any working receiver needs is not applied."""
		too_large = float(substation.constants.CALIBRATION_MAX_CORRECTION_PPM + 50)

		correction, reason = substation.scanner.RadioScanner._evaluate_calibration([too_large] * 5, 35.0)

		assert correction is None
		assert "larger than" in reason

	def test_no_measurements_is_refused (self):
		"""With nothing measured there is nothing to apply."""
		correction, reason = substation.scanner.RadioScanner._evaluate_calibration([], 0.0)

		assert correction is None
		assert "no measurements" in reason


class TestCalibrateSdr:

	def test_no_station_leaves_correction_unchanged (self, caplog):
		"""Regression: with only noise at the calibration frequency, the old code applied a random correction.

		Measured before the fix with these synthetic receivers, it applied 17, 25,
		-4 and 77 PPM across four seeds - at PMR446, 77 PPM is about five radio
		channels.  Now the correction must stay as it was, the warning must say
		calibration was skipped, and the tuning must be restored.
		"""
		sdr = FakeCalibrationSdr(seed=0)

		_calibrate(sdr)

		assert sdr.freq_correction == 0
		assert sdr.center_freq == 446.1e6
		assert sdr.sample_rate == 2.4e6
		assert "calibration at 93.700 MHz skipped" in caplog.text

	def test_existing_correction_survives_a_failed_calibration (self):
		"""A correction set before calibration is kept, not reset, when calibration is refused."""
		sdr = FakeCalibrationSdr(seed=3)
		sdr.freq_correction = 12

		_calibrate(sdr)

		assert sdr.freq_correction == 12

	def test_station_offset_is_cancelled (self):
		"""Regression: a station 30 PPM high must get a -30 correction, not +30.

		The old code applied the offset with its own sign.  Measured on an
		RTL-SDR Blog V4 against a broadcast station at 93.7 MHz, that doubled
		the error: the station sat at +3.4 PPM uncorrected, +6.4 PPM after the
		old +3 correction, and +0.4 PPM after the fixed -3.
		"""
		sdr = FakeCalibrationSdr(seed=1, station_offset_ppm=30.0)

		_calibrate(sdr)

		assert sdr.freq_correction == -30
		assert sdr.center_freq == 446.1e6
		assert sdr.sample_rate == 2.4e6

	def test_existing_correction_is_adjusted_not_replaced (self):
		"""Measurements are taken with the current correction in effect, so the result adds to it.

		A station 30 PPM high, seen through an existing -10 correction, measures
		20 PPM high.  Replacing the correction with -20 would leave the receiver
		10 PPM off; adjusting it gives -30.
		"""
		sdr = FakeCalibrationSdr(seed=2, station_offset_ppm=30.0)
		sdr.freq_correction = -10

		_calibrate(sdr)

		assert sdr.freq_correction == -30

	def test_calibrating_again_keeps_a_good_correction (self):
		"""Once corrected, the station measures on frequency, so a second calibration changes nothing."""
		sdr = FakeCalibrationSdr(seed=2, station_offset_ppm=30.0)

		_calibrate(sdr)
		_calibrate(sdr)

		assert sdr.freq_correction == -30

	def test_read_failure_restores_tuning (self):
		"""If a read fails part way, the receiver is returned to its scan tuning before the error propagates."""
		sdr = FakeCalibrationSdr(seed=0, fail_after_reads=5)

		with pytest.raises(OSError):
			_calibrate(sdr)

		assert sdr.center_freq == 446.1e6
		assert sdr.sample_rate == 2.4e6
		assert sdr.freq_correction == 0
