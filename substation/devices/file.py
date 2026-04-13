"""
IQ file playback device.

Streams a 2-channel WAV file (I/Q) through the scanner pipeline as if
it were a live SDR device.  Runs at full speed (no real-time pacing) —
the scanner processes samples as fast as it can.

Handles WAV files larger than 4 GB where the header's 32-bit size
fields have overflowed — the true frame count is computed from the
actual file size, not from the WAV header.

Supported formats: WAV with 2 channels (I and Q), PCM_16, any sample rate.

Usage:
    substation --band pmr --iq-file recording.wav --center-freq 446059313
"""

import logging
import os
import struct
import threading
import typing

import numpy
import numpy.typing

import substation.devices.base

logger = logging.getLogger(__name__)


def _parse_wav_header (file_path: str) -> tuple[int, int, int, int]:
	"""Parse a WAV header and return (sample_rate, channels, bits_per_sample, data_offset).

	Reads only the header — does not load sample data.  Works with
	files larger than 4 GB where the RIFF/data size fields have
	overflowed.
	"""

	with open(file_path, 'rb') as f:
		riff = f.read(4)
		if riff != b'RIFF':
			raise ValueError(f"Not a WAV file (missing RIFF header): {file_path}")
		f.read(4)  # file size (may be overflowed — ignore)
		wave = f.read(4)
		if wave != b'WAVE':
			raise ValueError(f"Not a WAV file (missing WAVE marker): {file_path}")

		sample_rate = 0
		channels = 0
		bits_per_sample = 0
		data_offset = 0

		while True:
			chunk_id = f.read(4)
			if len(chunk_id) < 4:
				break
			chunk_size = struct.unpack('<I', f.read(4))[0]

			if chunk_id == b'fmt ':
				fmt_data = f.read(chunk_size)
				audio_fmt, channels, sample_rate, _, _, bits_per_sample = struct.unpack('<HHIIHH', fmt_data[:16])
				if audio_fmt != 1:
					raise ValueError(f"Unsupported WAV format {audio_fmt} (only PCM supported)")
			elif chunk_id == b'data':
				data_offset = f.tell()
				break
			else:
				f.seek(chunk_size, 1)

	if data_offset == 0 or sample_rate == 0:
		raise ValueError(f"Invalid WAV file (missing fmt or data chunk): {file_path}")

	return sample_rate, channels, bits_per_sample, data_offset


class FileDevice (substation.devices.base.BaseDevice):

	"""Stream IQ samples from a WAV file.

	The file must have exactly 2 channels (I and Q), PCM_16 format.
	The sample rate is read from the WAV header.  The center frequency
	is provided by the caller.

	Handles files larger than 4 GB by computing the true frame count
	from the file size rather than trusting the WAV header's 32-bit
	size fields.

	Samples are delivered at full speed via the callback interface,
	identically to a live SDR device.
	"""

	def __init__ (self, file_path: str, center_freq: float) -> None:

		self._file_path = file_path
		self._center_freq = center_freq
		self._stop_event = threading.Event()
		self._reader_thread: threading.Thread | None = None

		# Parse WAV header (works even if size fields overflowed)
		sample_rate, channels, bits_per_sample, data_offset = _parse_wav_header(file_path)

		if channels != 2:
			raise ValueError(
				f"IQ file must have exactly 2 channels (I and Q), "
				f"got {channels}: {file_path}"
			)
		if bits_per_sample != 16:
			raise ValueError(
				f"Only PCM_16 WAV files are supported, "
				f"got {bits_per_sample}-bit: {file_path}"
			)

		self._sample_rate = float(sample_rate)
		self._data_offset = data_offset
		self._bytes_per_frame = channels * (bits_per_sample // 8)  # 4 bytes
		self._gain: float | str | None = None

		# True frame count from file size (not from WAV header which overflows at 4 GB)
		file_size = os.path.getsize(file_path)
		data_bytes = file_size - data_offset
		self._frames = data_bytes // self._bytes_per_frame

		duration = self._frames / self._sample_rate
		logger.info(
			f"IQ file: {file_path} — {self._sample_rate/1e6:.3f} MHz, "
			f"{self._frames} frames ({duration:.0f}s / {duration/3600:.1f}h), "
			f"center {self._center_freq/1e6:.6f} MHz"
		)

	@property
	def sample_rate (self) -> float | None:
		return self._sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:
		if abs(value - self._sample_rate) > 1.0:
			logger.warning(
				f"IQ file sample rate is {self._sample_rate:.0f} Hz, "
				f"ignoring request to set {value:.0f} Hz"
			)

	@property
	def center_freq (self) -> float | None:
		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		pass

	@property
	def gain (self) -> float | str | None:
		return self._gain

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		self._gain = value

	def _calibrate_iq_scale (self) -> float:
		"""Measure IQ amplitude and return a normalisation factor.

		Reads a few initial chunks from the raw file, measures median
		RMS, and returns a scale factor that brings the noise floor to
		~0.01 RMS if the signal is very weak.  Returns 1.0 if the
		amplitude is already in a sensible range.
		"""

		rms_values = []
		read_frames = 65536
		read_bytes = read_frames * self._bytes_per_frame

		with open(self._file_path, 'rb') as f:
			f.seek(self._data_offset)
			for _ in range(20):
				raw = f.read(read_bytes)
				if len(raw) < self._bytes_per_frame:
					break
				n_frames = len(raw) // self._bytes_per_frame
				samples = numpy.frombuffer(raw[:n_frames * self._bytes_per_frame], dtype=numpy.int16)
				iq = (samples[0::2] + 1j * samples[1::2]).astype(numpy.complex64) / 32768.0
				block_rms = float(numpy.sqrt(numpy.mean(numpy.abs(iq) ** 2)))
				rms_values.append(block_rms)
				if len(rms_values) >= 10:
					break

		if not rms_values:
			return 1.0

		median_rms = float(numpy.median(rms_values))

		if median_rms > 0.001:
			logger.debug(f"IQ scale: no normalisation needed (median RMS {median_rms:.6f})")
			return 1.0

		if median_rms < 1e-10:
			logger.warning("IQ calibration: signal too weak, using scale 1.0")
			return 1.0

		target_rms = 0.01
		scale = target_rms / median_rms
		logger.info(f"IQ scale: median RMS {median_rms:.6f} — applying {scale:.1f}x normalisation")
		return scale

	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:
		"""Stream IQ samples from the WAV file at full speed.

		Reads raw PCM_16 bytes directly (bypassing libsndfile) to
		handle files larger than 4 GB where the WAV header has
		overflowed.  Converts int16 I/Q pairs to complex64.
		"""

		self._stop_event.clear()

		iq_scale = self._calibrate_iq_scale()
		self.iq_scale = iq_scale

		# Read 64K frames at a time (256 KB of raw PCM data)
		# Read in large chunks for I/O efficiency (1M frames = 4 MB)
		read_frames = 1048576
		read_bytes = read_frames * self._bytes_per_frame
		data_offset = self._data_offset

		def _reader_loop () -> None:

			rx_buffer = numpy.array([], dtype=numpy.complex64)

			try:
				with open(self._file_path, 'rb') as f:
					f.seek(data_offset)

					while not self._stop_event.is_set():
						raw = f.read(read_bytes)
						if len(raw) < self._bytes_per_frame:
							break

						# Convert raw int16 pairs to complex64
						n_frames = len(raw) // self._bytes_per_frame
						samples = numpy.frombuffer(
							raw[:n_frames * self._bytes_per_frame], dtype=numpy.int16
						)
						iq = (samples[0::2] + 1j * samples[1::2]).astype(numpy.complex64) / 32768.0

						if iq_scale != 1.0:
							iq *= iq_scale

						rx_buffer = substation.devices.base.rechunk_samples(
							rx_buffer, iq, num_samples, callback
						)

				# Flush remaining samples
				if rx_buffer.size > 0:
					callback(rx_buffer, None)

			except Exception as exc:
				logger.error(f"IQ file read error: {exc}")

			logger.info("IQ file playback complete")

		# Run the reader directly in the calling thread (blocking).
		# The scanner calls read_samples_async via run_in_executor, which
		# expects a blocking call.  Live SDR devices block in their driver's
		# read loop; we block here until the file is fully read or cancelled.
		self._reader_thread = threading.current_thread()
		_reader_loop()

	def cancel_read_async (self) -> None:
		"""Stop file playback."""
		self._stop_event.set()

	def close (self) -> None:
		"""Stop playback and release resources."""
		self.cancel_read_async()
