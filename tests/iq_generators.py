"""Synthetic IQ signal generators for testing."""

import typing

import numpy
import numpy.typing
import soundfile


def generate_tone_iq (
	freq_hz: float,
	sample_rate: float,
	duration_s: float,
	amplitude: float = 1.0,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""Pure carrier at *freq_hz* relative to baseband."""

	n = int(sample_rate * duration_s)
	t = numpy.arange(n) / sample_rate
	return (amplitude * numpy.exp(2j * numpy.pi * freq_hz * t)).astype(numpy.complex64)


def generate_fm_iq (
	audio_freq: float,
	deviation: float,
	sample_rate: float,
	duration_s: float,
	carrier_offset: float = 0.0,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""FM-modulated IQ: a single audio tone frequency-modulated onto a carrier."""

	n = int(sample_rate * duration_s)
	t = numpy.arange(n) / sample_rate
	phase = 2.0 * numpy.pi * (
		carrier_offset * t
		+ (deviation / audio_freq) * (1.0 - numpy.cos(2.0 * numpy.pi * audio_freq * t))
	)
	return numpy.exp(1j * phase).astype(numpy.complex64)


def generate_am_iq (
	audio_freq: float,
	mod_depth: float,
	sample_rate: float,
	duration_s: float,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""AM-modulated IQ: envelope = 1 + m*sin(2*pi*f*t)."""

	n = int(sample_rate * duration_s)
	t = numpy.arange(n) / sample_rate
	envelope = 1.0 + mod_depth * numpy.sin(2.0 * numpy.pi * audio_freq * t)
	return (envelope + 0j).astype(numpy.complex64)


def generate_noise_iq (
	sample_rate: float,
	duration_s: float,
	power_db: float = 0.0,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""Gaussian white noise IQ at the given power level."""

	n = int(sample_rate * duration_s)
	rng = numpy.random.default_rng(42)
	amplitude = 10.0 ** (power_db / 20.0) / numpy.sqrt(2.0)
	noise = amplitude * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
	return noise.astype(numpy.complex64)


def generate_bursty_fm_iq (
	offset_hz: float,
	sample_rate: float,
	n: int,
	start_index: int = 0,
	level: float = 0.02,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""FM voice stand-in: a 1 kHz tone with a 12 Hz square-wave envelope, at offset_hz from the centre.

	The peaked audio spectrum passes the audio-flatness gate, and the
	envelope's swing between FFT segments passes the RF-variance gate, so it
	activates a radio channel as speech does.  start_index continues the
	signal from an earlier block without a phase jump.
	"""

	t = (numpy.arange(n) + start_index) / sample_rate
	envelope = level * (numpy.sign(numpy.sin(2.0 * numpy.pi * 12.0 * t)) + 1.1)
	phase = 2.0 * numpy.pi * offset_hz * t - 2.5 * numpy.cos(2.0 * numpy.pi * 1000.0 * t)
	return (envelope * numpy.exp(1j * phase)).astype(numpy.complex64)


def write_iq_wav (path: typing.Any, iq: numpy.typing.NDArray[numpy.complex64], sample_rate: float) -> None:

	"""Write IQ samples as a stereo PCM_16 WAV, the format FileDevice plays."""

	stereo = numpy.column_stack((iq.real, iq.imag))
	peak = float(numpy.max(numpy.abs(stereo))) or 1.0
	soundfile.write(str(path), stereo / (peak * 1.01), int(sample_rate), subtype="PCM_16")
