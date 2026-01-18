"""
HackRF device implementation
"""

import importlib
import numpy
import numpy.typing
import typing


class HackRfDevice:
	"""
	Wrapper for HackRF devices

	Provides a unified interface for HackRF hardware with automatic
	detection of various HackRF Python binding libraries.
	"""

	def __init__(self, device_index: int = 0) -> None:
		"""
		Initialize HackRF device

		Args:
			device_index: Index of the HackRF device to use (default: 0)

		Raises:
			RuntimeError: If HackRF bindings are not found or cannot be initialized
		"""
		self._module = self._import_hackrf_module()
		self._device_index = device_index
		self._rx_wrapper: typing.Callable | None = None
		self._sample_rate: float | None = None
		self._center_freq: float | None = None
		self._gain_db: float | None = None
		self._initialized_library: bool = False
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)

		# Initialize library and open device
		self._device = self._open_device()

	def _import_hackrf_module(self) -> typing.Any:
		"""
		Import HackRF module, trying multiple possible module names

		Returns:
			Imported HackRF module

		Raises:
			RuntimeError: If no HackRF binding module can be imported
		"""
		# Try python_hackrf first (most common)
		try:
			import python_hackrf.pylibhackrf.pyhackrf
			return python_hackrf.pylibhackrf.pyhackrf
		except Exception:
			pass

		# Try other module names as fallback
		for module_name in ('hackrf', 'pyhackrf'):
			try:
				return importlib.import_module(module_name)
			except Exception:
				continue

		raise RuntimeError("HackRF bindings not found; install python_hackrf")

	def _open_device(self) -> typing.Any:
		"""
		Open the HackRF device

		Returns:
			HackRF device object

		Raises:
			RuntimeError: If device cannot be opened
		"""
		# python_hackrf requires initialization before use
		if hasattr(self._module, 'pyhackrf_init'):
			self._module.pyhackrf_init()
			self._initialized_library = True

			# Get device list
			device_list = self._module.pyhackrf_device_list()
			if device_list.device_count == 0:
				raise RuntimeError("No HackRF devices found")

			if self._device_index >= device_list.device_count:
				raise RuntimeError(f"Device index {self._device_index} out of range (found {device_list.device_count} devices)")

			# Open device by serial number
			serial_number = device_list.serial_numbers[self._device_index]
			device = self._module.pyhackrf_open_by_serial(serial_number)
			if device is None:
				raise RuntimeError(f"Failed to open HackRF device at index {self._device_index}")

			del device_list
			return device

		# Fallback for other bindings
		raise RuntimeError("Unsupported HackRF bindings (pyhackrf_init not found)")

	def _call_module(self, name: str, *args: typing.Any) -> None:
		"""
		Call a module-level pyhackrf_* function, trying device-first signature.
		"""
		func = getattr(self._module, name, None)
		if func is None:
			return
		try:
			func(self._device, *args)
		except TypeError:
			func(*args)

	def _buffer_samples(
		self,
		samples: numpy.typing.NDArray[numpy.complex64],
		chunk_size: int,
		callback: typing.Callable
	) -> None:
		"""
		Buffer samples until chunk_size is reached, then emit fixed-size chunks.
		"""
		if chunk_size <= 0:
			callback(samples, None)
			return

		if self._rx_buffer.size == 0:
			buffered = samples
		else:
			buffered = numpy.concatenate((self._rx_buffer, samples))

		total = buffered.size
		offset = 0
		while total - offset >= chunk_size:
			chunk = buffered[offset:offset + chunk_size]
			callback(chunk, None)
			offset += chunk_size

		if offset < total:
			self._rx_buffer = buffered[offset:]
		else:
			self._rx_buffer = numpy.array([], dtype=buffered.dtype)

	@property
	def sample_rate(self) -> float | None:
		"""Get the current sample rate in Hz"""
		return self._sample_rate

	@sample_rate.setter
	def sample_rate(self, value: float) -> None:
		"""Set the sample rate in Hz"""
		self._sample_rate = value
		if hasattr(self._device, 'pyhackrf_set_sample_rate'):
			self._device.pyhackrf_set_sample_rate(value)
		elif hasattr(self._module, 'pyhackrf_set_sample_rate'):
			self._call_module('pyhackrf_set_sample_rate', value)
		elif hasattr(self._device, 'set_sample_rate'):
			self._device.set_sample_rate(value)
		elif hasattr(self._device, 'sample_rate'):
			self._device.sample_rate = value

	@property
	def center_freq(self) -> float | None:
		"""Get the current center frequency in Hz"""
		return self._center_freq

	@center_freq.setter
	def center_freq(self, value: float) -> None:
		"""Set the center frequency in Hz"""
		self._center_freq = value
		if hasattr(self._device, 'pyhackrf_set_freq'):
			self._device.pyhackrf_set_freq(int(value))
		elif hasattr(self._module, 'pyhackrf_set_freq'):
			self._call_module('pyhackrf_set_freq', int(value))
		elif hasattr(self._device, 'set_freq'):
			self._device.set_freq(value)
		elif hasattr(self._device, 'set_frequency'):
			self._device.set_frequency(value)
		elif hasattr(self._device, 'center_freq'):
			self._device.center_freq = value

	@property
	def gain(self) -> float | None:
		"""Get the current gain setting in dB"""
		return self._gain_db

	@gain.setter
	def gain(self, value: float | str | None) -> None:
		"""Set the gain (dB, 'auto', or None)"""
		self._gain_db = None if value == 'auto' else value

		if self._gain_db is None:
			return

		# Best-effort mapping for HackRF gain controls.
		# PyHackrfDevice has separate VGA and LNA gain controls
		if hasattr(self._device, 'pyhackrf_set_vga_gain'):
			self._device.pyhackrf_set_vga_gain(int(self._gain_db))
		elif hasattr(self._module, 'pyhackrf_set_vga_gain'):
			self._call_module('pyhackrf_set_vga_gain', int(self._gain_db))
		elif hasattr(self._device, 'set_vga_gain'):
			self._device.set_vga_gain(int(self._gain_db))
		elif hasattr(self._device, 'set_gain'):
			self._device.set_gain(self._gain_db)

		if hasattr(self._device, 'pyhackrf_set_lna_gain'):
			self._device.pyhackrf_set_lna_gain(int(self._gain_db))
		elif hasattr(self._module, 'pyhackrf_set_lna_gain'):
			self._call_module('pyhackrf_set_lna_gain', int(self._gain_db))
		elif hasattr(self._device, 'set_lna_gain'):
			self._device.set_lna_gain(int(self._gain_db))

	def _convert_samples(self, data: typing.Any) -> numpy.typing.NDArray[numpy.complex64]:
		"""
		Convert HackRF sample data to complex64 numpy array

		Args:
			data: Raw sample data from HackRF

		Returns:
			Complex IQ samples as numpy array (normalized to [-1, 1])

		Raises:
			RuntimeError: If data format is not supported
		"""
		if isinstance(data, numpy.ndarray):
			if numpy.iscomplexobj(data):
				return data.astype(numpy.complex64, copy=False)

			# HackRF provides int8 samples, need to normalize to [-1, 1]
			data = data.astype(numpy.float32, copy=False)
			if data.size % 2 != 0:
				data = data[:-1]
			iq = data.reshape(-1, 2)
			# Normalize: int8 range is -128 to 127, divide by 128 for [-1, 1] range
			return ((iq[:, 0] / 128.0) + 1j * (iq[:, 1] / 128.0)).astype(numpy.complex64)

		if isinstance(data, (bytes, bytearray, memoryview)):
			raw = numpy.frombuffer(data, dtype=numpy.int8)
			if raw.size % 2 != 0:
				raw = raw[:-1]
			iq = raw.astype(numpy.float32).reshape(-1, 2)
			return ((iq[:, 0] / 128.0) + 1j * (iq[:, 1] / 128.0)).astype(numpy.complex64)

		raise RuntimeError("Unsupported HackRF sample buffer type")

	def _extract_transfer_buffer(self, args: tuple[typing.Any, ...]) -> typing.Any | None:
		"""
		Extract sample buffer from callback arguments

		Args:
			args: Callback arguments tuple

		Returns:
			Sample buffer or None
		"""
		if len(args) == 0:
			return None

		# PyHackrfDevice callback signature: (device, buffer, buffer_length, valid_length)
		# The buffer is at args[1]
		if len(args) >= 2:
			return args[1]

		# Fallback for single argument or other binding types
		if len(args) == 1:
			obj = args[0]
			if hasattr(obj, 'buffer'):
				return obj.buffer
			if hasattr(obj, 'data'):
				return obj.data
			return obj

		return None

	def read_samples_async(self, callback: typing.Callable, _num_samples: int) -> None:
		"""
		Start asynchronous sample reading

		Args:
			callback: Function to call with samples (signature: callback(samples, context))
			_num_samples: Number of samples per callback (buffered for HackRF streaming)
		"""
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)

		def wrapper(*args: typing.Any) -> int:
			try:
				buffer_obj = self._extract_transfer_buffer(args)
				if buffer_obj is None:
					return 0
				samples = self._convert_samples(buffer_obj)
				self._buffer_samples(samples, _num_samples, callback)
				return 0
			except Exception as e:
				# Log error but don't crash streaming
				import sys
				print(f"Warning: HackRF callback error: {e}", file=sys.stderr)
				return 0

		self._rx_wrapper = wrapper

		# PyHackrfDevice uses set_rx_callback + pyhackrf_start_rx
		if hasattr(self._device, 'set_rx_callback') and hasattr(self._device, 'pyhackrf_start_rx'):
			self._device.set_rx_callback(wrapper)
			self._device.pyhackrf_start_rx()
		elif hasattr(self._module, 'pyhackrf_start_rx'):
			self._module.pyhackrf_start_rx(self._device, wrapper)
		elif hasattr(self._device, 'start_rx'):
			self._device.start_rx(wrapper)
		elif hasattr(self._device, 'start_rx_streaming'):
			self._device.start_rx_streaming(wrapper)
		else:
			raise RuntimeError("HackRF start_rx method not found")

	def cancel_read_async(self) -> None:
		"""Cancel asynchronous sample reading"""
		if hasattr(self._device, 'pyhackrf_stop_rx'):
			self._device.pyhackrf_stop_rx()
		elif hasattr(self._module, 'pyhackrf_stop_rx'):
			self._call_module('pyhackrf_stop_rx')
		elif hasattr(self._device, 'stop_rx'):
			self._device.stop_rx()
		elif hasattr(self._device, 'stop_rx_streaming'):
			self._device.stop_rx_streaming()

	def close(self) -> None:
		"""Close the device and release resources"""
		if hasattr(self._device, 'pyhackrf_close'):
			self._device.pyhackrf_close()
		elif hasattr(self._module, 'pyhackrf_close'):
			self._call_module('pyhackrf_close')
		elif hasattr(self._device, 'close'):
			self._device.close()

		# If we initialized the library, clean it up
		if self._initialized_library and hasattr(self._module, 'pyhackrf_exit'):
			self._module.pyhackrf_exit()
			self._initialized_library = False
