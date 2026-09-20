"""
HackRF device implementation with multi-binding support.

HackRF One is a wideband SDR transceiver with more capabilities than RTL-SDR:
- Frequency range: 1 MHz - 6 GHz (full coverage)
- Sample rate: 2-20 MHz
- 8-bit ADC/DAC resolution
- Transmit and receive capable
- USB 2.0 interface

Unlike RTL-SDR which has one standard Python library, HackRF has multiple
competing Python bindings with different APIs. This implementation automatically
detects and adapts to whichever binding is installed:
- python_hackrf (most common)
- hackrf
- pyhackrf

The adapter layer maps different function names and calling conventions
to a unified internal interface.
"""

import importlib
import logging
import threading
import typing

import numpy
import numpy.typing

import substation.devices.base

logger = logging.getLogger(__name__)


class HackRfDevice (substation.devices.base.BaseDevice):

	"""
	Wrapper for HackRF devices with automatic binding detection.

	Unifies multiple possible Python bindings into a single interface.
	Different HackRF libraries use different function names and signatures,
	so we detect what's available and create an adapter layer.

	In python_hackrf, the module only initialises the library and opens
	devices; every control call is a method of the opened device, which is
	where the wrapper looks for them.
	"""

	# How often the stream watchdog checks that the device is still
	# streaming, in seconds.
	STREAM_CHECK_INTERVAL_SECONDS = 0.5

	# Each control call, and the method names the bindings give it on the
	# opened device, in order of preference.
	_METHOD_NAMES: dict[str, tuple[str, ...]] = {
		'set_sample_rate': ('pyhackrf_set_sample_rate', 'set_sample_rate'),
		'set_freq': ('pyhackrf_set_freq', 'set_freq', 'set_frequency'),
		'set_lna_gain': ('pyhackrf_set_lna_gain', 'set_lna_gain'),
		'set_vga_gain': ('pyhackrf_set_vga_gain', 'set_vga_gain'),
		'start_rx': ('pyhackrf_start_rx', 'start_rx'),
		'stop_rx': ('pyhackrf_stop_rx', 'stop_rx'),
		'is_streaming': ('pyhackrf_is_streaming', 'is_streaming'),
		'close': ('pyhackrf_close', 'close'),
	}

	_REQUIRED_METHODS = ('set_sample_rate', 'set_freq', 'start_rx', 'stop_rx', 'close')

	def __init__ (self, device_index: int = 0) -> None:
		"""
		Initialize HackRF device with auto-detection of Python bindings.

		Args:
			device_index: Index of the HackRF device to use (default: 0)
		"""
		# Try to import any available HackRF binding
		self._module = self._import_hackrf_module()
		self._device_index = device_index
		self._device: typing.Any = None
		self._initialized_library = False

		# Buffer for accumulating samples into fixed-size chunks
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)
		self._rx_wrapper: typing.Callable | None = None

		# Set when the scan cancels streaming, so the stream watchdog knows
		# that the stream stopping is expected rather than a device fault.
		self._rx_cancelled = threading.Event()

		# Cache hardware state (HackRF doesn't provide getters for these)
		self._sample_rate: float | None = None
		self._center_freq: float | None = None
		self._gain_db: float | None = None

		# Initialize library and open the device, then find its control calls
		self._funcs: dict[str, typing.Callable] = {}
		self._open_device()

		try:
			self._setup_bindings()
		except RuntimeError:
			# Do not leave the device open and the library initialised
			for name in ('pyhackrf_close', 'close'):
				device_close = getattr(self._device, name, None)
				if callable(device_close):
					device_close()
					break
			if self._initialized_library and hasattr(self._module, 'pyhackrf_exit'):
				self._module.pyhackrf_exit()
			raise

	def _import_hackrf_module (self) -> typing.Any:
		"""
		Try to import any available HackRF Python binding.

		Tries multiple module paths in order of preference:
		1. python_hackrf: Most complete and maintained
		2. hackrf: Alternative binding
		3. pyhackrf: Older binding

		Returns the first successfully imported module.
		"""

		for module_name in ('python_hackrf.pylibhackrf.pyhackrf', 'hackrf', 'pyhackrf'):
			try:
				return importlib.import_module(module_name)
			except ImportError:
				continue

		raise RuntimeError(
			'HackRF bindings not found. Install them with: pip install "substation[hackrf]" '
			"(this builds python_hackrf, which needs the libhackrf development package; see INSTALL.md)"
		)

	def _setup_bindings (self) -> None:

		"""
		Find each control call on the opened device.

		Stores the device's bound method under our internal name in
		self._funcs, trying each binding's name in _METHOD_NAMES in turn, and
		raises RuntimeError naming any required call the device lacks.
		"""

		self._funcs = {}

		for key, names in self._METHOD_NAMES.items():
			for name in names:
				method = getattr(self._device, name, None)

				if callable(method):
					self._funcs[key] = method
					break

		missing = [key for key in self._REQUIRED_METHODS if key not in self._funcs]
		if missing:
			raise RuntimeError(
				f"HackRF binding is missing required functions: {', '.join(sorted(missing))}. "
				f"The installed binding may be incompatible."
			)

	def _call_safe (self, key: str, *args: typing.Any) -> typing.Any:

		"""Call a control method of the opened device by its internal name; a missing optional one does nothing."""

		method = self._funcs.get(key)

		if method is None:
			return None

		return method(*args)

	def _open_device (self) -> None:
		"""
		Initialize HackRF library and open device by index.

		Different bindings have different initialization sequences:
		1. python_hackrf: Call pyhackrf_init(), enumerate devices, open by serial
		2. Other bindings: Simply instantiate HackRF() object

		This method detects which pattern to use based on available functions.
		"""

		# Some bindings require explicit library initialization
		if hasattr(self._module, 'pyhackrf_init'):
			self._module.pyhackrf_init()
			self._initialized_library = True

		# Bindings with device enumeration support
		if hasattr(self._module, 'pyhackrf_device_list'):
			# Get list of all connected HackRF devices
			device_list = self._module.pyhackrf_device_list()

			if device_list.device_count == 0:
				raise RuntimeError("No HackRF devices found.")

			if self._device_index >= device_list.device_count:
				raise RuntimeError(f"Index {self._device_index} out of range ({device_list.device_count} found).")

			# Open specific device by serial number (more reliable than index)
			serial = device_list.serial_numbers[self._device_index]
			self._device = self._module.pyhackrf_open_by_serial(serial)

		# Simpler bindings: just instantiate the HackRF class
		elif hasattr(self._module, 'HackRF'):
			self._device = self._module.HackRF()

		if self._device is None:
			raise RuntimeError("Failed to open HackRF device.")

	@property
	def sample_rate (self) -> float | None:
		"""The sample rate last set, in Hz; HackRF cannot report its own."""
		return self._sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:
		"""Set the sample rate in Hz."""
		self._sample_rate = value
		self._call_safe ('set_sample_rate', value)

	@property
	def center_freq (self) -> float | None:
		"""The centre frequency last set, in Hz; HackRF cannot report its own."""
		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		"""Tune to a centre frequency in Hz."""
		self._center_freq = value
		self._call_safe('set_freq', int(value))

	@property
	def gain (self) -> float | None:
		"""Get the current gain setting (cached, HackRF doesn't provide getters)."""
		return self._gain_db

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		"""
		Set the receive gain.

		HackRF has two gain stages:
		- LNA (Low Noise Amplifier): 0-40 dB in 8 dB steps
		- VGA (Variable Gain Amplifier): 0-62 dB in 2 dB steps

		For simplicity, we set both to the same value. This isn't optimal
		(ideally you'd maximize LNA first, then use VGA for fine tuning),
		but it works for most applications.
		"""
		if value == 'auto' or value is None:
			# HackRF has no hardware AGC.  Convert 'auto'/None to a fixed
			# numeric gain — the getter will return float, never 'auto'.
			logger.warning(
				"HackRF does not support automatic gain. "
				"Defaulting to LNA=32 dB, VGA=30 dB. "
				"Set sdr_gain_db to a numeric value in config.yaml for manual control."
			)
			self._gain_db = 32.0
			self._call_safe('set_lna_gain', 32)
			self._call_safe('set_vga_gain', 30)
		else:
			self._gain_db = float(value)
			gain_val = int(self._gain_db)
			# Clamp to valid hardware ranges and step sizes
			lna_gain = min(gain_val, 40) - (min(gain_val, 40) % 8)  # 0-40 in 8 dB steps
			vga_gain = min(gain_val, 62) - (min(gain_val, 62) % 2)  # 0-62 in 2 dB steps
			self._call_safe('set_lna_gain', lna_gain)
			self._call_safe('set_vga_gain', vga_gain)
			if lna_gain != gain_val or vga_gain != gain_val:
				logger.info(f"HackRF gain adjusted from {gain_val} dB: LNA={lna_gain} dB (8 dB steps, max 40), VGA={vga_gain} dB (2 dB steps, max 62)")
			else:
				logger.info(f"HackRF gain: LNA={lna_gain} dB, VGA={vga_gain} dB")

	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:
		"""
		Start asynchronous sample streaming.

		HackRF delivers samples in variable-size blocks (typically 262144 bytes).
		We need to:
		1. Convert raw int8 samples to complex64
		2. Buffer and rechunk to the requested num_samples size
		3. Call the user callback with fixed-size blocks

		Returns once streaming has started.  If the device stops streaming
		without being cancelled, as when it is unplugged, or the callback
		fails, the callback is called once with (None, None), which is how
		BaseDevice signals the end of a stream.
		"""

		# Clear any leftover samples from previous streaming session
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)
		self._rx_cancelled.clear()
		stream_ended = threading.Event()

		def end_stream () -> None:
			"""Report the end of the stream to the scanner, once."""
			if not stream_ended.is_set() and not self._rx_cancelled.is_set():
				stream_ended.set()
				callback(None, None)

		def wrapper (*args: typing.Any) -> int:
			"""
			Callback invoked by HackRF library for each sample block.

			python_hackrf passes (device, buffer, buffer_length, valid_length),
			and only the first valid_length bytes of the buffer hold samples.
			Other bindings pass (device, buffer) or (buffer).

			Returns 0 to continue streaming, non-zero to stop.
			"""
			try:
				buffer_obj = args[1] if len(args) >= 2 else args[0]
				if len(args) >= 4:
					buffer_obj = buffer_obj[:args[3]]

				# Convert raw int8 IQ to complex64 normalized samples
				samples = self._convert_samples(buffer_obj)
				# Rechunk to requested size and call user callback
				self._buffer_samples(samples, num_samples, callback)
				return 0

			except Exception:
				# The scanner's callback does not fail in normal running, so
				# this is a fault: stop the stream and say so, rather than
				# drop the block and carry on.
				logger.exception("HackRF receive callback failed; stopping the stream")
				end_stream()
				return -1

		self._rx_wrapper = wrapper

		# Different bindings register callbacks differently
		if hasattr(self._device, 'set_rx_callback'):
			# Pattern 1: device.set_rx_callback(func), then start_rx()
			self._device.set_rx_callback(wrapper)
			self._call_safe('start_rx')
		else:
			# Pattern 2: start_rx(callback)
			self._call_safe('start_rx', wrapper)

		# libhackrf stops calling back when the device is lost, and nothing
		# else would tell the scanner, which would then wait forever.
		if 'is_streaming' in self._funcs:
			threading.Thread(target=self._watch_stream, args=(end_stream,), name="hackrf-stream-watchdog", daemon=True).start()

	def _watch_stream (self, end_stream: typing.Callable[[], None]) -> None:

		"""Poll the device until streaming stops; if the scan did not cancel it, report the end of the stream."""

		while not self._rx_cancelled.wait(self.STREAM_CHECK_INTERVAL_SECONDS):

			try:
				streaming = bool(self._call_safe('is_streaming'))
			except Exception as exc:
				logger.debug(f"HackRF streaming check failed: {exc}")
				streaming = False

			if not streaming:
				if not self._rx_cancelled.is_set():
					logger.error("HackRF stopped streaming")
				end_stream()
				return

	def cancel_read_async (self) -> None:
		"""Stop streaming; the stream watchdog then stops quietly."""
		self._rx_cancelled.set()
		self._call_safe('stop_rx')

	def close (self) -> None:
		"""Stop streaming, close the device, and release the HackRF library."""
		try:
			self.cancel_read_async()
		except Exception as exc:
			logger.debug(f"Error cancelling async read during close: {exc}")
		self._call_safe('close')

		if self._initialized_library and hasattr(self._module, 'pyhackrf_exit'):
			self._module.pyhackrf_exit()
			self._initialized_library = False

	def _convert_samples (self, data: typing.Any) -> numpy.typing.NDArray[numpy.complex64]:
		"""
		Convert raw HackRF int8 samples to complex64 normalized to [-1, 1].

		HackRF sends samples as interleaved int8 pairs: [I0, Q0, I1, Q1, I2, Q2, ...]
		Each value is in the range [-128, 127].

		Conversion process:
		1. Parse as int8 array
		2. Reshape to (N, 2) for [I, Q] pairs
		3. Normalize to [-1, 1] range by dividing by 128
		4. Combine into complex numbers: I + jQ
		"""

		# Some bindings already provide complex samples
		if isinstance(data, numpy.ndarray) and numpy.iscomplexobj(data):
			return data.astype(numpy.complex64)

		# Convert buffer to numpy int8 array
		raw = numpy.frombuffer(data, dtype=numpy.int8) if not isinstance(data, numpy.ndarray) else data

		# Samples must come in I/Q pairs (even number of bytes)
		n_complex = raw.size // 2
		if n_complex == 0:
			return numpy.array([], dtype=numpy.complex64)

		# Pre-allocate complex array
		complex_samples = numpy.empty(n_complex, dtype=numpy.complex64)

		# Extract I and Q components directly to the complex array.
		# Normalize from [-128, 127] to approximately [-1, 1].
		# This avoids large temporary float32 arrays and reshapes.
		complex_samples.real = raw[0:n_complex * 2:2].astype(numpy.float32) / 128.0
		complex_samples.imag = raw[1:n_complex * 2:2].astype(numpy.float32) / 128.0

		return complex_samples

	def _buffer_samples (self, samples: numpy.typing.NDArray[numpy.complex64], chunk_size: int, callback: typing.Callable) -> None:

		"""
		Rechunk variable-size HackRF blocks into fixed-size chunks for the
		scanner.  Delegates to the shared rechunk_samples helper so all
		device backends use identical boundary logic.
		"""

		self._rx_buffer = substation.devices.base.rechunk_samples(
			self._rx_buffer, samples, chunk_size, callback
		)
