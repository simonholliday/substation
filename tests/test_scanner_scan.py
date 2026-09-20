"""Tests that run RadioScanner.scan() from start to finish, with fake receivers and IQ files instead of hardware."""

import asyncio
import concurrent.futures
import datetime
import json
import pathlib
import threading
import time
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

import iq_generators

EVENTS = ('channel_state', 'recording_started', 'recording_saved', 'recording_discarded', 'noise_floor', 'channel_snr')


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


def _play_transmissions (config_dict, tmp_path, seconds, transmissions, handlers=(), sample_rate=256e3):

	"""
	Play weak noise with bursty FM transmissions on chosen radio channels through a real scan.

	transmissions holds (position in scanner.channels, start seconds, stop
	seconds) for each one.  handlers holds (event, handler) pairs registered
	before the scan starts.  Returns the scanner once scan() has returned.
	"""

	config_dict["bands"]["test_nfm"]["sample_rate"] = sample_rate
	config = substation.config.validate_config(config_dict)
	probe = substation.scanner.RadioScanner(config=config, band_name="test_nfm", device_type="file")

	n = int(sample_rate * seconds)
	rng = numpy.random.default_rng(3)
	iq = (0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(numpy.complex64)

	for position, start_s, stop_s in transmissions:
		first, last = int(start_s * sample_rate), int(stop_s * sample_rate)
		offset = probe.channels[position] - probe.center_freq
		iq[first:last] += iq_generators.generate_bursty_fm_iq(offset, sample_rate, last - first, start_index=first)

	path = tmp_path / "playback.wav"
	iq_generators.write_iq_wav(path, iq, sample_rate)

	scanner = substation.scanner.RadioScanner(
		config=config,
		band_name="test_nfm",
		device_type="file",
		clock=substation.scanner.VirtualClock(datetime.datetime(2000, 1, 1), sample_rate),
		device_kwargs={"file_path": str(path), "center_freq": probe.center_freq},
	)

	for event, handler in handlers:
		scanner.on(event, handler)

	asyncio.run(scanner.scan())
	return scanner


def _recorder (events):

	"""Handlers that append (event, payload) to events, one per event name."""

	return [(name, lambda _name=name, **payload: events.append((_name, payload))) for name in EVENTS]


class TestEventsDuringAScan:

	def test_payloads_are_plain_python_types (self, minimal_config_dict, tmp_path):
		"""Regression: is_active was a numpy bool on ON events, so json.dumps of a payload raised."""
		events = []
		_play_transmissions(minimal_config_dict, tmp_path, 6.0, [(3, 1.5, 4.0)], handlers=_recorder(events))

		states = [payload for name, payload in events if name == 'channel_state']
		assert [payload['is_active'] for payload in states] == [True, False]

		for name, payload in events:
			json.dumps(payload)

			if name == 'channel_state':
				assert type(payload['is_active']) is bool
				assert type(payload['snr_db']) is float

	def test_activation_events_arrive_in_the_documented_order (self, minimal_config_dict, tmp_path):
		"""One transmission gives recording_started, channel_state ON, channel_state OFF, then recording_saved."""
		events = []
		_play_transmissions(minimal_config_dict, tmp_path, 6.0, [(3, 1.5, 4.0)], handlers=_recorder(events))

		sequence = [
			name if name != 'channel_state' else ('ON' if payload['is_active'] else 'OFF')
			for name, payload in events
			if name not in ('noise_floor', 'channel_snr')
		]
		assert sequence == ['recording_started', 'ON', 'OFF', 'recording_saved']

	def test_async_handler_receives_events_emitted_without_a_loop (self, minimal_config_dict, tmp_path):
		"""Regression: async handlers on recording_saved, noise_floor and channel_snr were silently dropped."""
		received = []

		async def on_saved (**payload):
			received.append(payload['file_path'])

		async def on_noise_floor (**payload):
			received.append('noise_floor')

		_play_transmissions(minimal_config_dict, tmp_path, 6.0, [(3, 1.5, 4.0)], handlers=[('recording_saved', on_saved), ('noise_floor', on_noise_floor)])

		assert received.count('noise_floor') > 0
		assert [path for path in received if path != 'noise_floor'][0].endswith('.wav')

	def test_radio_channel_on_when_the_scan_ends_gets_its_off (self, minimal_config_dict, tmp_path):
		"""Regression: a radio channel still ON at the end of a scan never got an OFF, so OSC consumers showed it active."""
		events = []
		async_states = []

		async def on_state (**payload):
			async_states.append(payload['is_active'])

		_play_transmissions(minimal_config_dict, tmp_path, 5.0, [(3, 2.0, 5.0)], handlers=_recorder(events) + [('channel_state', on_state)])

		states = [payload['is_active'] for name, payload in events if name == 'channel_state']
		assert states == [True, False]
		assert async_states == [True, False]


def _wav_chunks (path) -> list[bytes]:

	"""The chunk IDs of a RIFF WAV file, in order."""

	data = pathlib.Path(path).read_bytes()
	chunks = []
	position = 12

	while position + 8 <= len(data):
		chunk_id = data[position:position + 4]
		size = int.from_bytes(data[position + 4:position + 8], "little")
		chunks.append(chunk_id)
		position += 8 + size + (size % 2)

	return chunks


class TestEndOfScan:

	@pytest.mark.parametrize("stop_s", [5.0, 5.2, 5.4])
	def test_recording_that_ends_near_the_end_of_a_file_is_finished (self, minimal_config_dict, tmp_path, stop_s):
		"""Regression: a recording whose transmission ended just before the file did was cancelled mid-close.

		Its stop ran as a fire-and-forget coroutine that the end of the scan
		cancelled, so it had no BEXT chunk, the post-recording checks never
		ran, and recording_saved never fired.
		"""
		events = []
		_play_transmissions(minimal_config_dict, tmp_path, 5.8, [(3, 1.5, stop_s)], handlers=_recorder(events))

		saved = [payload['file_path'] for name, payload in events if name == 'recording_saved']
		assert len(saved) == 1
		assert b"bext" in _wav_chunks(saved[0])

	def test_cleanup_waits_for_the_slice_being_processed (self, scanner_instance, monkeypatch):
		"""Regression: cleanup raced a slice still being processed after a cancel, and a KeyError left recordings open.

		The slice turns one radio channel off, taking its recorder, while
		cleanup is part way through closing the others.
		"""
		stopped = []
		scanner_instance.channel_recorders = {1.0: "recorder 1", 2.0: "recorder 2"}

		async def fake_stop (channel_freq, recorder, tone):
			await asyncio.sleep(0.2)
			stopped.append(recorder)

		def slice_turning_channel_2_off ():
			time.sleep(0.05)
			scanner_instance.channel_recorders.pop(2.0, None)

		monkeypatch.setattr(scanner_instance, "_stop_channel_recording", fake_stop)
		device = FakeLiveDevice(blocks=0)
		scanner_instance.sdr = device

		async def run ():
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
				scanner_instance._processing_future = executor.submit(slice_turning_channel_2_off)
				await scanner_instance._cleanup_sdr()

		asyncio.run(run())

		assert stopped == ["recorder 1"]
		assert device.closed
