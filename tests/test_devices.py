"""Tests for the device factory, mock protocol, and FileDevice."""

import datetime
import sys
import threading
import types
import unittest.mock

import numpy
import pytest
import soundfile

import substation.devices
import substation.scanner


def _mock_create_device (alias, device_type, mock_class_name):
	"""Helper: mock the submodule and call create_device."""
	mock_device = unittest.mock.MagicMock()
	mock_cls = unittest.mock.MagicMock(return_value=mock_device)
	fake_module = types.ModuleType(f"substation.devices.{device_type}")
	setattr(fake_module, mock_class_name, mock_cls)

	patches = {
		f"substation.devices.{device_type}": fake_module,
	}
	with unittest.mock.patch.dict(sys.modules, patches):
		# Also set the attribute on the parent module so `substation.devices.rtlsdr` resolves
		setattr(substation.devices, device_type, fake_module)
		try:
			device = substation.devices.create_device(alias, device_index=0)
		finally:
			delattr(substation.devices, device_type)
	return mock_cls


class TestCreateDevice:

	def test_unknown_type_raises (self):
		with pytest.raises(ValueError, match="Unsupported"):
			substation.devices.create_device("unknown_sdr")

	def test_rtlsdr_aliases (self):
		for alias in ("rtl", "rtlsdr", "rtl-sdr"):
			mock_cls = _mock_create_device(alias, "rtlsdr", "RtlSdrDevice")
			mock_cls.assert_called_once_with(0)

	def test_hackrf_aliases (self):
		for alias in ("hackrf", "hackrf-one", "hackrfone"):
			mock_cls = _mock_create_device(alias, "hackrf", "HackRfDevice")
			mock_cls.assert_called_once_with(0)

	def test_airspy_aliases (self):
		for alias in ("airspy", "airspy-r2", "airspyr2"):
			mock_cls = _mock_create_device(alias, "soapysdr", "SoapySdrDevice")
			mock_cls.assert_called_once_with('airspy', 0)

	def test_airspyhf_aliases (self):
		for alias in ("airspyhf", "airspy-hf", "airspyhf+"):
			mock_cls = _mock_create_device(alias, "soapysdr", "SoapySdrDevice")
			mock_cls.assert_called_once_with('airspyhf', 0)

	def test_soapy_generic_passthrough (self):
		mock_cls = _mock_create_device("soapy:lime", "soapysdr", "SoapySdrDevice")
		mock_cls.assert_called_once_with('lime', 0)

	def test_case_insensitive (self):
		mock_cls = _mock_create_device("RTLSDR", "rtlsdr", "RtlSdrDevice")
		mock_cls.assert_called_once()


class TestFileDevice:

	def _make_iq_wav (self, tmp_path, n_frames=16000, sample_rate=2000000):
		"""Create a small 2-channel IQ WAV file for testing."""
		path = str(tmp_path / "test_iq.wav")
		t = numpy.arange(n_frames) / sample_rate
		# FM-modulated tone at +50 kHz offset
		iq = numpy.exp(2j * numpy.pi * 50000 * t).astype(numpy.complex64)
		# Write as 2-channel float32 (I, Q)
		stereo = numpy.column_stack([iq.real, iq.imag])
		soundfile.write(path, stereo, sample_rate)
		return path

	def test_reads_sample_rate (self, tmp_path):
		path = self._make_iq_wav(tmp_path, sample_rate=2000000)
		dev = substation.devices.create_device('file', file_path=path, center_freq=446e6)
		assert dev.sample_rate == 2000000

	def test_center_freq (self, tmp_path):
		path = self._make_iq_wav(tmp_path)
		dev = substation.devices.create_device('file', file_path=path, center_freq=446.059e6)
		assert dev.center_freq == 446.059e6
		# Setter is a no-op — getter still returns the original
		dev.center_freq = 500e6
		assert dev.center_freq == 446.059e6

	def test_streams_samples (self, tmp_path):
		"""FileDevice delivers IQ samples via callback."""
		n_frames = 65536
		path = self._make_iq_wav(tmp_path, n_frames=n_frames, sample_rate=1000000)
		dev = substation.devices.create_device('file', file_path=path, center_freq=446e6)

		received = []
		done = threading.Event()

		def cb (samples, _ctx):
			received.append(samples.copy())
			if sum(len(s) for s in received) >= n_frames:
				done.set()

		chunk_size = 16384
		dev.read_samples_async(cb, chunk_size)
		done.wait(timeout=5.0)
		dev.close()

		total = sum(len(s) for s in received)
		assert total >= n_frames - chunk_size  # may lose last partial chunk
		assert all(s.dtype == numpy.complex64 for s in received)
		# All chunks except possibly the last must be exactly chunk_size
		assert all(len(s) == chunk_size for s in received[:-1])
		assert len(received[-1]) <= chunk_size

	def test_rejects_mono_file (self, tmp_path):
		"""A mono WAV file should be rejected."""
		path = str(tmp_path / "mono.wav")
		soundfile.write(path, numpy.zeros(1000, dtype=numpy.float32), 16000)
		with pytest.raises(ValueError, match="2 channels"):
			substation.devices.create_device('file', file_path=path, center_freq=446e6)

	def test_rejects_non_wav (self, tmp_path):
		"""A file without a RIFF header should be rejected."""
		path = str(tmp_path / "bad.wav")
		with open(path, 'wb') as f:
			f.write(b'\x00' * 1024)
		with pytest.raises(ValueError, match="RIFF"):
			substation.devices.create_device('file', file_path=path, center_freq=446e6)

	def test_rejects_missing_wave_marker (self, tmp_path):
		"""A RIFF file without WAVE marker should be rejected."""
		path = str(tmp_path / "bad.wav")
		with open(path, 'wb') as f:
			f.write(b'RIFF')
			f.write(b'\x00\x00\x00\x00')  # file size
			f.write(b'NOPE')  # not WAVE
		with pytest.raises(ValueError, match="WAVE"):
			substation.devices.create_device('file', file_path=path, center_freq=446e6)

	def test_calibrate_iq_scale_normal_amplitude (self, tmp_path):
		"""IQ scale returns 1.0 when amplitude is already in a sensible range."""
		path = self._make_iq_wav(tmp_path, n_frames=65536, sample_rate=1000000)
		dev = substation.devices.create_device('file', file_path=path, center_freq=446e6)
		# Default test signal has significant amplitude, so no normalisation needed
		scale = dev._calibrate_iq_scale()
		assert scale == 1.0

	def test_calibrate_iq_scale_weak_signal (self, tmp_path):
		"""IQ scale applies normalisation for very weak signals."""
		path = str(tmp_path / "weak_iq.wav")
		n_frames = 65536
		# Create a very weak signal (amplitude 0.0001)
		t = numpy.arange(n_frames) / 1000000
		iq = (numpy.exp(2j * numpy.pi * 50000 * t) * 0.0001).astype(numpy.complex64)
		stereo = numpy.column_stack([iq.real, iq.imag])
		soundfile.write(path, stereo, 1000000)
		dev = substation.devices.create_device('file', file_path=path, center_freq=446e6)
		scale = dev._calibrate_iq_scale()
		# Weak signal should get a scale factor > 1.0
		assert scale > 1.0


class TestVirtualClock:

	def test_initial_time (self):
		start = datetime.datetime(2025, 3, 16, 16, 13, 20)
		clock = substation.scanner.VirtualClock(start, sample_rate=2000000)
		assert clock.now() == start
		assert clock.time() == pytest.approx(start.timestamp())

	def test_advance (self):
		start = datetime.datetime(2025, 3, 16, 16, 13, 20)
		clock = substation.scanner.VirtualClock(start, sample_rate=2000000)
		# Advance by 2 million samples = 1 second
		clock.advance(2000000)
		expected = start + datetime.timedelta(seconds=1)
		assert clock.now() == expected
		assert clock.time() == pytest.approx(expected.timestamp())

	def test_fractional_seconds (self):
		start = datetime.datetime(2000, 1, 1)
		clock = substation.scanner.VirtualClock(start, sample_rate=16000)
		clock.advance(8000)  # 0.5 seconds
		assert clock.time() == pytest.approx(start.timestamp() + 0.5)
