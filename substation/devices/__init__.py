"""
Device abstraction layer for SDR hardware.

Provides a unified interface for different SDR devices through the
BaseDevice abstract class. Specific implementations handle the details
of RTL-SDR, HackRF, and potentially other hardware.

The factory function create_device() simplifies device instantiation
by accepting a device type string and returning the appropriate instance.
"""

import typing

import substation.devices.base


_DEVICE_FAMILIES: dict[str, str] = {
	'rtl': 'rtlsdr', 'rtlsdr': 'rtlsdr', 'rtl-sdr': 'rtlsdr',
	'hackrf': 'hackrf', 'hackrf-one': 'hackrf', 'hackrfone': 'hackrf',
	'airspy': 'airspy', 'airspy-r2': 'airspy', 'airspyr2': 'airspy',
	'airspyhf': 'airspyhf', 'airspy-hf': 'airspyhf', 'airspyhf+': 'airspyhf',
	'file': 'file',
}


def normalize_device_family (device_type: str) -> str:
	"""Return the canonical device family name for a CLI device type string.

	Maps all aliases to a canonical name (e.g. 'rtl', 'rtl-sdr' → 'rtlsdr').
	For 'soapy:<driver>' strings, returns the driver name.
	"""

	key = device_type.lower()

	if key in _DEVICE_FAMILIES:
		return _DEVICE_FAMILIES[key]

	if key.startswith('soapy:'):
		return key.split(':', 1)[1]

	return key


def create_device (device_type: str, device_index: int = 0, **kwargs: typing.Any) -> 'substation.devices.base.BaseDevice':
	"""
	Factory function to create SDR device instances.

	This function provides a simple way to instantiate SDR devices without
	needing to know the specific class names. It accepts various common
	name variations for each device type.

	Supported devices:
	- RTL-SDR (aliases: 'rtl', 'rtlsdr', 'rtl-sdr')
	  Low-cost receiver, 24 MHz - 1.7 GHz, up to 2.4 MHz sample rate

	- HackRF (aliases: 'hackrf', 'hackrf-one', 'hackrfone')
	  Wideband transceiver, 1 MHz - 6 GHz, 2-20 MHz sample rate

	- AirSpy R2 (aliases: 'airspy', 'airspy-r2', 'airspyr2')
	  High-dynamic-range receiver, 24 MHz - 1.8 GHz, 2.5/10 MHz sample rate
	  Requires: soapysdr-module-airspy (system package)

	- AirSpy HF+ Discovery (aliases: 'airspyhf', 'airspy-hf', 'airspyhf+')
	  HF/VHF receiver, 0.5 kHz - 31 MHz + 60-260 MHz, up to 768 kHz BW
	  Requires: soapysdr-module-airspyhf (system package)

	- Generic SoapySDR (prefix: 'soapy:<driver>')
	  Any SoapySDR-compatible device, e.g., 'soapy:rtlsdr', 'soapy:lime'

	Args:
		device_type: Type of device (case-insensitive, accepts aliases)
		device_index: Device index for multi-device setups (default: 0)

	Returns:
		Device instance implementing the BaseDevice interface

	Raises:
		ValueError: If device_type is not recognized
		RuntimeError: If device cannot be opened (hardware not found, etc.)

	Example:
		device = create_device('rtlsdr')
		device.sample_rate = 2048000
		device.center_freq = 446000000
		device.gain = 'auto'
	"""

	family = normalize_device_family(device_type)

	# Lazy imports: only load the binding for the requested device type.
	if family == 'rtlsdr':
		import substation.devices.rtlsdr
		return substation.devices.rtlsdr.RtlSdrDevice(device_index)

	if family == 'hackrf':
		import substation.devices.hackrf
		return substation.devices.hackrf.HackRfDevice(device_index)

	# SoapySDR-based devices (AirSpy R2, AirSpy HF+, generic)
	if family in ('airspy', 'airspyhf') or device_type.lower().startswith('soapy:'):
		import substation.devices.soapysdr
		return substation.devices.soapysdr.SoapySdrDevice(family, device_index)

	# IQ file playback
	if family == 'file':
		import substation.devices.file
		file_path = kwargs.get('file_path')
		center_freq = kwargs.get('center_freq')
		if not file_path or center_freq is None:
			raise ValueError("FileDevice requires file_path and center_freq kwargs")
		return substation.devices.file.FileDevice(file_path, center_freq)

	raise ValueError(f"Unsupported device_type: {device_type}")
