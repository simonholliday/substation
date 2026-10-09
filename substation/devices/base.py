"""
Abstract base class for SDR device implementations.

Defines a unified interface for different SDR hardware (RTL-SDR, HackRF, etc.).
All device implementations must provide methods for:
- Setting sample rate and center frequency
- Configuring gain (manual or automatic)
- Reading IQ samples asynchronously via callbacks
- Resource cleanup

This abstraction allows the scanner to work with different hardware
without knowing the specific device type.

Also provides shared utilities used by multiple device wrappers.
"""

import abc
import logging
import typing

import numpy
import numpy.typing

logger = logging.getLogger(__name__)

# A receiver's IQ samples are scaled up when the median RMS of its first
# blocks is at or below MAX_UNSCALED_RMS, to bring it to TARGET_RMS, typical
# of an RTL-SDR's noise floor; below MIN_MEASURABLE_RMS nothing can be measured.
IQ_SCALE_TARGET_RMS = 0.01
IQ_SCALE_MAX_UNSCALED_RMS = 0.001
IQ_SCALE_MIN_MEASURABLE_RMS = 1e-10


def rechunk_samples (
	rx_buffer: numpy.typing.NDArray[numpy.complex64],
	samples: numpy.typing.NDArray[numpy.complex64],
	chunk_size: int,
	callback: typing.Callable,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""
	Accumulate samples and emit fixed-size chunks to a callback.

	SDR backends like HackRF and SoapySDR deliver variable-size blocks,
	but the scanner expects fixed-size blocks (sdr_device_sample_size).
	This helper carries leftover samples between calls so the boundary
	logic is identical for every backend.

	1. Concatenate any leftover from a previous call with the new samples
	2. Emit as many full chunks as possible to the callback
	3. Return any remaining samples to be carried into the next call

	When chunk_size is zero or negative the samples are forwarded directly
	to the callback without rechunking, and the leftover buffer is reset.

	Args:
		rx_buffer: Leftover samples from the previous call (may be empty)
		samples: Newly received samples to be rechunked
		chunk_size: Target chunk size; pass 0 or less to disable rechunking
		callback: Function invoked as callback(chunk, None) for each chunk

	Returns:
		The new leftover buffer to be passed back on the next call.
	"""

	if chunk_size <= 0:
		callback(samples, None)
		return numpy.array([], dtype=numpy.complex64)

	combined = numpy.concatenate((rx_buffer, samples)) if rx_buffer.size > 0 else samples

	num_chunks = combined.size // chunk_size

	# The callback is handed views into combined, not copies.  That is safe
	# because every producer passes in a freshly allocated array rather than a
	# driver buffer it will reuse.
	for i in range(num_chunks):
		start, end = i * chunk_size, (i + 1) * chunk_size
		callback(combined[start:end], None)

	leftover = combined.size % chunk_size

	if leftover > 0:
		return combined[-leftover:]

	return numpy.array([], dtype=numpy.complex64)


def iq_scale_from_rms (rms_values: list[float]) -> float:

	"""
	Decide the factor a receiver's IQ samples are scaled by, from the RMS of
	its first few blocks, and log the decision.

	Some receivers deliver IQ samples far below the [-1, 1] range the
	demodulators expect (an Airspy HF+ peaks around 0.001 to 0.005).  The
	median RMS stands for the noise floor, unmoved by a strong signal in one
	block.  When it is at or below IQ_SCALE_MAX_UNSCALED_RMS, the samples are
	scaled to bring it to IQ_SCALE_TARGET_RMS.  Otherwise, or with nothing to
	measure, the factor is 1.0.  The file and SoapySDR devices share it, so
	the thresholds live in one place (#4832).

	Args:
		rms_values: The RMS of each block measured, in the order read.

	Returns:
		The factor to multiply every IQ sample by.
	"""

	if not rms_values:
		logger.warning("IQ sample scale: no IQ samples to measure, using 1.0")
		return 1.0

	median_rms = float(numpy.median(rms_values))

	if median_rms < IQ_SCALE_MIN_MEASURABLE_RMS:
		logger.warning(f"IQ sample scale: signal too weak to measure (median RMS {median_rms:.3g}), using 1.0")
		return 1.0

	if median_rms > IQ_SCALE_MAX_UNSCALED_RMS:
		logger.debug(f"IQ sample scale: no normalisation needed (median RMS {median_rms:.6f})")
		return 1.0

	# INFO, because a factor other than 1.0 changes what every later IQ
	# sample's amplitude means, such as to the ADC saturation check
	scale = IQ_SCALE_TARGET_RMS / median_rms
	logger.info(f"IQ sample scale: median RMS {median_rms:.6f}, applying {scale:.1f}x normalisation")
	return scale


class BaseDevice (abc.ABC):
	"""
	Abstract base class defining the interface for SDR devices.

	This class uses Python's ABC (Abstract Base Class) pattern to enforce
	a consistent interface across different SDR hardware implementations.
	Subclasses must implement all abstract methods and properties.

	The interface is designed around asynchronous sample reading:
	- Set hardware parameters (frequency, sample rate, gain)
	- Start async streaming with a callback function
	- Hardware continuously calls the callback with IQ sample blocks
	- Cancel streaming when done
	- Clean up resources

	All device implementations must provide these properties and methods.
	"""

	# Multiplicative factor that the wrapper applies to raw IQ samples
	# before delivering them to the scanner.  Defaults to 1.0 (no scaling).
	# The SoapySDR wrapper sets this to whatever calibration factor it
	# computed at startup, so the scanner can interpret amplitudes
	# correctly — for example, the ADC saturation check needs to know
	# whether a 0.95-amplitude post-wrapper sample corresponds to a
	# real 0.95-amplitude raw sample (iq_scale==1.0) or to a much
	# smaller raw sample that has been amplified during normalisation.
	iq_scale: float = 1.0

	@property
	@abc.abstractmethod
	def sample_rate (self) -> float | None:
		"""Get the current sample rate in Hz"""
		pass

	@sample_rate.setter
	@abc.abstractmethod
	def sample_rate (self, value: float) -> None:
		"""Set the sample rate in Hz"""
		pass

	@property
	@abc.abstractmethod
	def center_freq (self) -> float | None:
		"""Get the current center frequency in Hz"""
		pass

	@center_freq.setter
	@abc.abstractmethod
	def center_freq (self, value: float) -> None:
		"""Set the center frequency in Hz"""
		pass

	@property
	@abc.abstractmethod
	def gain (self) -> float | str | None:
		"""Get the current gain setting (dB, 'auto', or None)"""
		pass

	@gain.setter
	@abc.abstractmethod
	def gain (self, value: float | str | None) -> None:
		"""Set the gain (dB, 'auto', or None)"""
		pass

	@abc.abstractmethod
	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:
		"""
		Start asynchronous sample reading.

		Blocking backends (RTL-SDR, file playback) run their read loop in
		the calling thread and only return at end-of-stream, cancellation,
		or error.  Non-blocking backends (HackRF, SoapySDR) start a
		background reader and return immediately; if their stream later
		dies on its own (device fault), they signal end-of-stream by
		invoking callback(None, None) so the scanner can shut down instead
		of waiting for samples that will never arrive.

		Args:
			callback: Function to call with samples (signature: callback(samples, context));
				samples is None to signal end-of-stream
			num_samples: Number of samples to read per callback
		"""
		pass

	@abc.abstractmethod
	def cancel_read_async (self) -> None:
		"""Cancel asynchronous sample reading"""
		pass

	@abc.abstractmethod
	def close (self) -> None:
		"""Close the device and release resources"""
		pass
