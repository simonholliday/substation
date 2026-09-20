"""
IQ file playback device.

Streams a 2-channel WAV file (I/Q) through the scanner pipeline as if
it were a live SDR device.  Runs at full speed (no real-time pacing) —
the scanner processes samples as fast as it can.

Handles WAV files larger than 4 GB where the header's 32-bit size
fields have overflowed — the true frame count is computed from the
actual file size, not from the WAV header.

Supported formats: WAV with 2 channels (I and Q), PCM_16, any sample rate,
as plain PCM or WAVE_FORMAT_EXTENSIBLE, in a RIFF or an RF64 file.

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


# WAVE format codes: plain PCM, and the extensible header that names its
# real format in a SubFormat GUID whose first two bytes are that code.
_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE

# A 32-bit size field holding this value defers to the ds64 chunk (RF64).
_RF64_SIZE_IN_DS64 = 0xFFFFFFFF


def _parse_wav_header (file_path: str) -> tuple[int, int, int, int, int | None]:
	"""Parse a WAV header and return (sample_rate, channels, bits_per_sample, data_offset, data_size).

	Reads only the header — does not load sample data.  data_size is the
	data chunk's length in bytes where the header states it reliably, and
	None where it cannot: an RF64 file whose ds64 chunk is missing, or a
	RIFF file over 4 GB, whose 32-bit size fields have overflowed.  The
	caller then takes the data to run to the end of the file.
	"""

	file_size = os.path.getsize(file_path)

	with open(file_path, 'rb') as f:
		riff = f.read(4)
		if riff not in (b'RIFF', b'RF64', b'BW64'):
			raise ValueError(f"Not a WAV file (missing RIFF header): {file_path}")
		f.read(4)  # file size (may be overflowed — ignore)
		wave = f.read(4)
		if wave != b'WAVE':
			raise ValueError(f"Not a WAV file (missing WAVE marker): {file_path}")

		sample_rate = 0
		channels = 0
		bits_per_sample = 0
		data_offset = 0
		data_size: int | None = None
		ds64_data_size: int | None = None

		while True:
			chunk_id = f.read(4)
			if len(chunk_id) < 4:
				break
			chunk_size = struct.unpack('<I', f.read(4))[0]

			if chunk_id == b'fmt ':
				fmt_data = f.read(chunk_size)
				audio_fmt, channels, sample_rate, _, _, bits_per_sample = struct.unpack('<HHIIHH', fmt_data[:16])
				if audio_fmt == _WAVE_FORMAT_EXTENSIBLE and len(fmt_data) >= 26:
					audio_fmt = struct.unpack('<H', fmt_data[24:26])[0]
				if audio_fmt != _WAVE_FORMAT_PCM:
					raise ValueError(f"Unsupported WAV format {audio_fmt} (only PCM supported)")
			elif chunk_id == b'ds64':
				ds64_data = f.read(chunk_size)
				if len(ds64_data) >= 16:
					ds64_data_size = struct.unpack('<Q', ds64_data[8:16])[0]
			elif chunk_id == b'data':
				data_offset = f.tell()
				if chunk_size == _RF64_SIZE_IN_DS64:
					data_size = ds64_data_size
				elif file_size - data_offset <= 0xFFFFFFFF:
					data_size = chunk_size
				break
			else:
				f.seek(chunk_size, 1)

			# A chunk of odd length is followed by a pad byte.
			if chunk_size % 2:
				f.seek(1, 1)

	if data_offset == 0 or sample_rate == 0:
		raise ValueError(f"Invalid WAV file (missing fmt or data chunk): {file_path}")

	return sample_rate, channels, bits_per_sample, data_offset, data_size


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
		"""Open the IQ WAV file at file_path, recorded at center_freq Hz, and read its header."""

		self._file_path = file_path
		self._center_freq = center_freq
		self._stop_event = threading.Event()

		# Parse WAV header (works even if size fields overflowed)
		sample_rate, channels, bits_per_sample, data_offset, data_size = _parse_wav_header(file_path)

		if channels != 2:
			raise ValueError(
				f"IQ file must have exactly 2 audio channels (I and Q), "
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

		# Frame count from the data chunk's size where the header states it,
		# so chunks after the data are not read as IQ samples, and otherwise
		# from the file size (a RIFF header overflows at 4 GB).
		data_bytes = os.path.getsize(file_path) - data_offset
		if data_size is not None:
			data_bytes = min(data_bytes, data_size)
		self._frames = data_bytes // self._bytes_per_frame

		duration = self._frames / self._sample_rate
		logger.info(
			f"IQ file: {file_path} — {self._sample_rate/1e6:.3f} MHz, "
			f"{self._frames} frames ({duration:.0f}s / {duration/3600:.1f}h), "
			f"center {self._center_freq/1e6:.6f} MHz"
		)

	@property
	def sample_rate (self) -> float | None:
		"""The file's sample rate in Hz, from its WAV header."""
		return self._sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:
		"""Keep the file's own rate: a different one is ignored, with a warning."""
		if abs(value - self._sample_rate) > 1.0:
			logger.warning(
				f"IQ file sample rate is {self._sample_rate:.0f} Hz, "
				f"ignoring request to set {value:.0f} Hz"
			)

	@property
	def center_freq (self) -> float | None:
		"""The centre frequency the recording was made at, in Hz, as given."""
		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		"""Ignored: a recording's centre frequency is fixed."""
		pass

	@property
	def gain (self) -> float | str | None:
		"""The gain last set; it has no effect on a file."""
		return self._gain

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		"""Remember the gain; a file has none to set."""
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

		# Read in large chunks for I/O efficiency: 1M frames, 4 MB of PCM_16
		read_frames = 1048576
		read_bytes = read_frames * self._bytes_per_frame
		remaining = self._frames * self._bytes_per_frame
		rx_buffer = numpy.array([], dtype=numpy.complex64)

		# Runs in the calling thread and blocks until the file is read or the
		# scan cancels it, as a live SDR's driver loop does; the scanner calls
		# it through run_in_executor.  A read error propagates, so the scan
		# ends with it instead of finishing as if the file had ended.
		with open(self._file_path, 'rb') as f:
			f.seek(self._data_offset)

			while remaining >= self._bytes_per_frame and not self._stop_event.is_set():
				raw = f.read(min(read_bytes, remaining))
				if len(raw) < self._bytes_per_frame:
					break

				remaining -= len(raw)

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

		if self._stop_event.is_set():
			logger.info("IQ file playback stopped")
			return

		# Flush remaining samples
		if rx_buffer.size > 0:
			callback(rx_buffer, None)

		logger.info("IQ file playback complete")

	def cancel_read_async (self) -> None:
		"""Stop file playback."""
		self._stop_event.set()

	def close (self) -> None:
		"""Stop playback and release resources."""
		self.cancel_read_async()
