"""
RTL-SDR device implementation.

RTL-SDR is a low-cost SDR receiver based on TV tuner dongles (RTL2832U chip).
Typical specifications:
- Frequency range: 24 MHz - 1766 MHz (with gaps)
- Sample rate: up to 2.4 MHz (typical: 2.048 MHz)
- 8-bit ADC resolution
- USB 2.0 interface

This implementation wraps the pyrtlsdr library to conform to the BaseDevice interface.
"""

import importlib
import importlib.util
import logging
import sys
import types
import typing

import numpy

import substation.devices.base

logger = logging.getLogger(__name__)


def _import_pyrtlsdr () -> types.ModuleType:

	"""
	Import pyrtlsdr, even where the pkg_resources module it expects is missing.

	pyrtlsdr 0.3.0 runs `import pkg_resources` when it loads, only to read its
	own version number inside a try/except that falls back to 'unknown'.
	pkg_resources ships with setuptools, which Python 3.12 and later no longer
	put in a new venv, and which setuptools 81 removed.  Without it, the RTL-SDR
	cannot be opened at all.  pyrtlsdr 0.4.0 no longer needs it, but requires a
	librtlsdr symbol that distro builds lack, so the project stays on 0.3.0.

	When the real module is missing, an empty stand-in is registered for the
	length of the import only, so pyrtlsdr's version lookup fails harmlessly
	inside its own try/except.  Anything importing pkg_resources afterwards
	still gets an ImportError rather than the empty stand-in.
	"""

	stand_in = None

	if importlib.util.find_spec("pkg_resources") is None:
		stand_in = types.ModuleType("pkg_resources")
		sys.modules["pkg_resources"] = stand_in

	try:
		return importlib.import_module("rtlsdr")

	finally:
		if stand_in is not None and sys.modules.get("pkg_resources") is stand_in:
			del sys.modules["pkg_resources"]


rtlsdr = _import_pyrtlsdr()


class RtlSdrDevice (substation.devices.base.BaseDevice):
	
	"""
	Wrapper for RTL-SDR devices

	Provides a unified interface for RTL-SDR hardware.

	pyrtlsdr closes the device before it raises from any failed call, and
	closing frees librtlsdr's handle to it.  None of pyrtlsdr's methods check
	for that, so a later call would reach freed memory.  This wrapper checks
	first: once the device is closed, by close() or by pyrtlsdr after a
	failure, settings and reads raise OSError, and cancelling or closing
	again does nothing.
	"""

	def __init__ (self, device_index: int = 0) -> None:
		"""
		Initialize RTL-SDR device

		Args:
			device_index: Index of the RTL-SDR device to use (default: 0)
		"""
		self._device_index = device_index
		self._device = rtlsdr.RtlSdr(device_index)

		# librtlsdr starts every opened device with no correction.  The value is
		# remembered here rather than read back, because pyrtlsdr's getter treats
		# any negative correction as an error code and closes the device.
		self._freq_correction = 0

		# The centre frequency is also remembered rather than read back:
		# pyrtlsdr rounds the value it reports to the nearest kHz, while the
		# tuner keeps the exact frequency it was given.
		self._center_freq: float | None = None

	def _driver (self) -> typing.Any:

		"""
		Return the pyrtlsdr device, or raise OSError if it has been closed.

		Every call that reaches librtlsdr goes through here, so none can reach
		a handle that pyrtlsdr has already freed.
		"""

		if not getattr(self._device, 'device_opened', False):
			raise OSError("The RTL-SDR device is closed: an earlier call to it failed, or it was closed deliberately")

		return self._device

	@property
	def sample_rate (self) -> float:
		"""Get the current sample rate in Hz"""
		return self._driver().sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:
		"""Set the sample rate in Hz"""
		self._driver().sample_rate = value

	@property
	def center_freq (self) -> float:

		"""
		Get the current center frequency in Hz.

		Returns the value last set rather than asking the driver, because
		pyrtlsdr rounds the frequency it reports to the nearest kHz.  Most
		bands tune a few hundred Hz off a whole kHz, and the scanner places
		every radio channel relative to this value.
		"""

		if self._center_freq is None:
			return self._driver().center_freq

		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		"""Set the center frequency in Hz"""
		self._driver().center_freq = value
		self._center_freq = float(value)

	@property
	def gain (self) -> float | str | None:
		"""Get the current gain setting (dB, 'auto', or None)"""
		return self._driver().gain

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		"""Set the gain (dB, 'auto', or None)"""
		self._driver().gain = value

	@property
	def freq_correction (self) -> int:
		"""
		Get the current frequency correction in PPM (Parts Per Million).

		RTL-SDR uses a crystal oscillator that can drift with temperature.
		PPM correction compensates for this: +10 PPM means the crystal is
		10 parts per million fast, so we adjust down by that amount.

		Returns the last value set rather than asking the driver: pyrtlsdr
		reports a negative correction as a failure and closes the device.
		"""
		return self._freq_correction

	@freq_correction.setter
	def freq_correction (self, value: int) -> None:
		"""
		Set the frequency correction in PPM (Parts Per Million).

		Positive values: crystal is fast, correct downward
		Negative values: crystal is slow, correct upward
		Typical range: -100 to +100 PPM

		Setting the value the device already holds is skipped, because
		librtlsdr rejects it as an invalid parameter.
		"""
		if value == self._freq_correction:
			return

		self._driver().freq_correction = value
		self._freq_correction = value

	@property
	def serial (self) -> str | None:
		"""
		Get the device serial number if available.

		Serial numbers are useful for identifying specific dongles when
		multiple RTL-SDR devices are connected. Returns None if the
		serial cannot be determined.
		"""

		serial = None

		try:
			# Query all connected RTL-SDR serial numbers
			serials = rtlsdr.RtlSdr.get_device_serial_addresses()

			# Get serial for our device index
			if 0 <= self._device_index < len(serials):
				serial = serials[self._device_index]
		except (OSError, AttributeError, IndexError, ValueError) as exc:
			# Some RTL-SDR dongles don't have readable serial numbers;
			# the librtlsdr call can also fail on USB enumeration races
			# or when the driver version is missing the accessor.  Log
			# at DEBUG so production runs stay quiet.
			logger.debug(f"Could not read RTL-SDR serial: {exc}")
			serial = None

		# Convert bytes to string if necessary
		if isinstance(serial, bytes):
			serial = serial.decode('ascii', errors='replace')

		# Clean up whitespace and handle empty strings
		if isinstance(serial, str):
			serial = serial.strip()
			return serial if serial else None

		return None

	def read_samples (self, num_samples: int) -> typing.Any:

		"""
		Read samples synchronously

		Args:
			num_samples: Number of samples to read

		Returns:
			Complex IQ samples as a complex64 numpy array
		"""

		return numpy.asarray(self._driver().read_samples(num_samples)).astype(numpy.complex64)

	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:

		"""
		Start asynchronous sample reading

		pyrtlsdr delivers complex128.  Each block is passed on as complex64,
		as every other device delivers it, which halves the memory a queued
		slice takes; 8-bit samples lose nothing in the conversion.

		Args:
			callback: Function to call with samples (signature: callback(samples, context))
			num_samples: Number of samples to read per callback
		"""

		def deliver (samples: typing.Any, context: typing.Any) -> None:
			"""Pass one block on as complex64."""
			callback(numpy.asarray(samples).astype(numpy.complex64), context)

		self._driver().read_samples_async(deliver, num_samples)

	def cancel_read_async (self) -> None:

		"""
		Cancel asynchronous sample reading.

		Does nothing once the device is closed: streaming has already stopped,
		and librtlsdr's handle may have been freed.
		"""

		if not getattr(self._device, 'device_opened', False):
			return

		self._device.cancel_read_async()

	def close (self) -> None:
		"""Close the device and release resources"""
		try:
			self._device.close()
		except Exception as exc:
			logger.warning(f"Error closing RTL-SDR device: {exc}")
