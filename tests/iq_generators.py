"""Synthetic IQ signal generators for testing."""

import numpy
import numpy.typing


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
