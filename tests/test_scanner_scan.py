"""Tests that run RadioScanner.scan() from start to finish, with fake receivers and IQ files instead of hardware."""

import asyncio
import datetime
import threading
import unittest.mock

import numpy
import pytest
import soundfile
import yaml

import substation.cli
import substation.config
import substation.devices
import substation.devices.base
import substation.scanner


class FakeLiveDevice (substation.devices.base.BaseDevice):

	"""A live receiver that streams a few slices of weak noise and then stops.

	With error set it stops as an unplugged RTL-SDR does, by raising from its
	blocking read.  Without one it stops as a HackRF or SoapySDR device does,
	by reporting the end of its stream from its own thread.
	"""

	def __init__ (self, blocks: int, error: Exception | None = None) -> None:

		"""Choose how many slices to deliver and how to stop."""

		self._sample_rate: float | None = None
		self._center_freq: float | None = None
		self._gain: float | str | None = None
		self._blocks = blocks
		self._error = error
		self.closed = False

	@property
	def sample_rate (self) -> float | None:
		"""The sample rate last set."""
		return self._sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:
		"""Set the sample rate."""
		self._sample_rate = value

	@property
	def center_freq (self) -> float | None:
		"""The centre frequency last set."""
		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		"""Tune."""
		self._center_freq = value

	@property
	def gain (self) -> float | str | None:
		"""The gain last set."""
		return self._gain

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		"""Set the gain."""
		self._gain = value

	def _deliver (self, callback, num_samples: int) -> None:

		"""Pass on the slices of noise."""

		rng = numpy.random.default_rng(0)

		for _ in range(self._blocks):
			noise = 0.01 * (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples))
			callback(noise.astype(numpy.complex64), None)

	def read_samples_async (self, callback, num_samples: int) -> None:

		"""Stream, then stop in the chosen way."""

		if self._error is not None:
			self._deliver(callback, num_samples)
			raise self._error

		def stream () -> None:
			self._deliver(callback, num_samples)
			callback(None, None)

		threading.Thread(target=stream, daemon=True).start()

	def cancel_read_async (self) -> None:
		"""Nothing to cancel once the stream has stopped."""

	def close (self) -> None:
		"""Record that the scan closed the device."""
		self.closed = True


def _write_iq_wav (path, sample_rate: float, seconds: float) -> None:

	"""Write a stereo PCM_16 IQ file of weak noise, the format FileDevice plays."""

	rng = numpy.random.default_rng(1)
	frames = int(sample_rate * seconds)
	iq = 0.01 * rng.standard_normal((frames, 2))
	soundfile.write(str(path), iq, int(sample_rate), subtype="PCM_16")


class TestScanEnds:

	def test_live_device_failure_is_raised_after_cleanup (self, app_config, monkeypatch):
		"""Regression: a live device that failed mid-scan ended scan() normally, so the CLI exited 0.

		An unplugged RTL-SDR raises from its blocking read.  scan() must close
		the device, then raise that error.
		"""
		device = FakeLiveDevice(blocks=2, error=OSError("LIBUSB_ERROR_NO_DEVICE"))
		monkeypatch.setattr(substation.devices, "create_device", lambda *args, **kwargs: device)
		scanner = substation.scanner.RadioScanner(config=app_config, band_name="test_nfm", device_type="rtlsdr")

		with pytest.raises(OSError, match="LIBUSB_ERROR_NO_DEVICE"):
			asyncio.run(scanner.scan())

		assert device.closed

	def test_live_stream_that_ends_is_an_error (self, app_config, monkeypatch):
		"""A live device that reports the end of its stream has failed, because only the scan ends a live stream."""
		device = FakeLiveDevice(blocks=2)
		monkeypatch.setattr(substation.devices, "create_device", lambda *args, **kwargs: device)
		scanner = substation.scanner.RadioScanner(config=app_config, band_name="test_nfm", device_type="hackrf")

		with pytest.raises(RuntimeError, match="stopped streaming"):
			asyncio.run(scanner.scan())

		assert device.closed

	def test_device_that_cannot_open_is_raised (self, app_config, monkeypatch):
		"""A device that fails to open stops the scan with its error."""
		def refuse (*args, **kwargs):
			raise OSError("Could not open SDR")

		monkeypatch.setattr(substation.devices, "create_device", refuse)
		scanner = substation.scanner.RadioScanner(config=app_config, band_name="test_nfm", device_type="rtlsdr")

		with pytest.raises(OSError, match="Could not open SDR"):
			asyncio.run(scanner.scan())

	def test_file_played_to_its_end_returns_normally (self, app_config, tmp_path):
		"""Reaching the end of an IQ file is how playback finishes, so it is not an error."""
		band = app_config.bands["test_nfm"]
		iq_path = tmp_path / "noise.wav"
		_write_iq_wav(iq_path, band.sample_rate, seconds=1.0)
		clock = substation.scanner.VirtualClock(datetime.datetime(2000, 1, 1), band.sample_rate)

		scanner = substation.scanner.RadioScanner(
			config=app_config,
			band_name="test_nfm",
			device_type="file",
			clock=clock,
			device_kwargs={"file_path": str(iq_path), "center_freq": (band.freq_start + band.freq_end) / 2},
		)

		asyncio.run(scanner.scan())

		assert clock.samples_delivered > 0


class TestCliExitStatus:

	def test_scan_that_fails_exits_1 (self, tmp_path, minimal_config_dict, monkeypatch):
		"""Regression: every failure inside a scan exited 0, so service managers never restarted the scanner."""
		config_path = tmp_path / "config.yaml"
		config_path.write_text(yaml.dump(minimal_config_dict))
		monkeypatch.chdir(tmp_path)

		with unittest.mock.patch("sys.argv", ["substation", "--band", "test_nfm", "--device-type", "no-such-device", "-c", str(config_path)]):
			with pytest.raises(SystemExit) as exc_info:
				substation.cli.main()

		assert exc_info.value.code == 1
