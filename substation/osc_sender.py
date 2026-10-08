"""
OSC event forwarding for Substation.

Bridges RadioScanner's existing channel-state and recording callbacks
onto OSC (Open Sound Control) messages, so downstream tools can react
to radio activity in real time.  Two OSC endpoints are supported:

- **Sequencer** (always): receives `/radio/state` and `/radio/recording`.
  Defaults to 127.0.0.1:9000, matching the Subsequence generative MIDI
  sequencer's OSC server.
- **Sampler** (optional): if `sampler_host` is provided, also receives
  `/sample/import` whenever a recording is finalised, so a sample-based
  instrument (e.g. Subsample on 127.0.0.1:9002) can load the new WAV
  without having to watch the output directory.

This module is **not** imported by any other part of Substation.  It
relies on python-osc, which is only installed when the `osc` optional
extra is present — keep the import here at module level so the
ImportError is immediate and obvious if a user forgets to install it.
Run `pip install -e ".[osc]"` to enable OSC support.

OSC address / argument reference (the Substation outbound addresses):

    /radio/state      band_name:str  channel_index:int  is_active:int(0/1)  snr_db:float  ctcss_hz:float  dcs_code:int
    /radio/recording  band_name:str  channel_index:int  file_path:str  ctcss_hz:float  dcs_code:int
    /sample/import    file_path:str                  (only when sampler_host is set)

ctcss_hz / dcs_code carry any subaudible tone detected on the activation.
OSC has no native null, so 0.0 / 0 mean "no tone detected" (valid CTCSS
tones start at 67 Hz and DCS codes are always nonzero).  DCS codes are
octal, and dcs_code is the code's integer value: DCS 023 is sent as 19.
Format it in octal (f"{dcs_code:03o}") to show it as a radio does.
"""

import logging
import typing

import pythonosc.udp_client

if typing.TYPE_CHECKING:
	# Only needed for the type annotation on attach(); avoids a runtime
	# import cycle so this module stays independent of the scanner.
	import substation.scanner


logger = logging.getLogger(__name__)


class OscEventSender:

	"""
	Send the scanner's events as OSC messages, to a sequencer and, optionally, a
	sampler.

	```python
	osc = substation.osc_sender.OscEventSender(
		host="127.0.0.1", port=9000,
		sampler_host="127.0.0.1", sampler_port=9002,
	)
	osc.attach(scanner)
	```

	The messages it sends:

	- `/radio/state`, when a radio channel turns ON or OFF: `band_name` (str),
	  `channel_index` (int), `is_active` (int, 1 or 0), `snr_db` (float),
	  `ctcss_hz` (float), `dcs_code` (int).
	- `/radio/recording`, when a recording is saved: `band_name` (str),
	  `channel_index` (int), `file_path` (str), `ctcss_hz` (float),
	  `dcs_code` (int).
	- `/sample/import`, when a recording is saved and `sampler_host` is set:
	  `file_path` (str).

	`ctcss_hz` and `dcs_code` carry any subaudible tone detected on the
	activation.  OSC has no null, so 0.0 and 0 mean that no tone was detected:
	CTCSS tones start at 67 Hz, and no DCS code is 0.  A DCS code is octal,
	and `dcs_code` is its integer value, so DCS 023 arrives as 19; format it in
	octal, as `f"{dcs_code:03o}"`, to show it as a radio does.  Tone detection
	has not yet been thoroughly tested with real radios.

	Each message goes over UDP without waiting for a reply.  A send that fails,
	such as one to a host that cannot be reached, is logged as a warning and
	never stops the scan.  It needs the `osc` extra:
	`pip install "substation[osc]"`.
	"""

	def __init__ (
		self,
		host: str = '127.0.0.1',
		port: int = 9000,
		sampler_host: str | None = None,
		sampler_port: int = 9002,
	) -> None:

		"""
		Set up the sender, opening a UDP client for each receiver.

		Args:
			host: The sequencer's host name or IP address.  The default,
				localhost, suits Subsequence running on the same machine.
			port: The sequencer's UDP port.  The default is Subsequence's.
			sampler_host: The sampler's host name or IP address.  When it is set,
				each saved recording is also sent to it as `/sample/import`, so
				the sampler can import the file.  None sends nothing to a sampler.
			sampler_port: The sampler's UDP port, used only when `sampler_host` is
				set.  The default is Subsample's.
		"""

		self._client = pythonosc.udp_client.SimpleUDPClient(host, port)

		self._sampler_client: pythonosc.udp_client.SimpleUDPClient | None
		if sampler_host is not None:
			self._sampler_client = pythonosc.udp_client.SimpleUDPClient(sampler_host, sampler_port)
			logger.info(
				f"OSC sender → sequencer {host}:{port}, sampler {sampler_host}:{sampler_port}"
			)
		else:
			self._sampler_client = None
			logger.info(f"OSC sender → sequencer {host}:{port} (no sampler)")

	def on_state_change (
		self,
		band_name: str,
		channel_index: int,
		is_active: bool,
		snr_db: float,
		ctcss_hz: float | None = None,
		dcs_code: int | None = None,
	) -> None:

		"""
		Send `/radio/state` to the sequencer: the handler for `channel_state`
		that `attach()` subscribes.

		The message's arguments are `band_name`, `channel_index`, `is_active` as
		1 or 0, `snr_db`, `ctcss_hz` and `dcs_code`, with 0.0 and 0 for no tone.

		It runs on the scanner's event loop, so it returns quickly.  A failed send
		is caught and logged; anything else is a fault in Substation, and reaches
		the scanner, which logs it with its traceback.

		Args:
			band_name: The band's name.
			channel_index: The radio channel's index in the band.
			is_active: Whether the radio channel turned ON.
			snr_db: The radio channel's signal-to-noise ratio, in dB.
			ctcss_hz: The CTCSS tone detected, in Hz, or None.
			dcs_code: The DCS code detected, as its integer value, or None.
		"""

		# OSC has no native boolean — encode as 0 or 1.  Explicit ternary
		# instead of int(is_active) so the intent is obvious at a glance.
		active_int = 1 if is_active else 0

		# Null-as-sentinel mapping.  See docstring for the rationale.
		ctcss_f = float(ctcss_hz) if ctcss_hz is not None else 0.0
		dcs_i = int(dcs_code) if dcs_code is not None else 0

		try:
			self._client.send_message(
				'/radio/state',
				[band_name, int(channel_index), active_int, float(snr_db), ctcss_f, dcs_i],
			)

		except (OSError, ValueError, TypeError) as exc:
			# OSError covers UDP socket failures (host unreachable, EMFILE,
			# etc.).  ValueError / TypeError cover pythonosc's argument
			# encoding errors if an unexpected type ever slips through.
			# Anything else (AttributeError, KeyError, ...) is a programming
			# bug and is deliberately allowed to propagate, so the scanner
			# logs it with its traceback.
			logger.warning(f"OSC /radio/state send failed: {exc}")

	def on_recording_saved (
		self,
		band_name: str,
		channel_index: int,
		file_path: str,
		ctcss_hz: float | None = None,
		dcs_code: int | None = None,
	) -> None:

		"""
		Send `/radio/recording` to the sequencer, and `/sample/import` to the
		sampler when one is set: the handler for `recording_saved` that `attach()`
		subscribes.

		`/radio/recording`'s arguments are `band_name`, `channel_index`,
		`file_path`, `ctcss_hz` and `dcs_code`, with 0.0 and 0 for no tone.
		`/sample/import` carries only `file_path`; the sampler can read a tone from
		the recording's metadata.

		It runs on the scanner's event loop.

		Args:
			band_name: The band's name.
			channel_index: The radio channel's index in the band.
			file_path: The saved recording.
			ctcss_hz: The CTCSS tone detected during the activation, in Hz, or None.
			dcs_code: The DCS code detected, as its integer value, or None.
		"""

		path_str = str(file_path)

		ctcss_f = float(ctcss_hz) if ctcss_hz is not None else 0.0
		dcs_i = int(dcs_code) if dcs_code is not None else 0

		try:
			self._client.send_message(
				'/radio/recording',
				[band_name, int(channel_index), path_str, ctcss_f, dcs_i],
			)

		except (OSError, ValueError, TypeError) as exc:
			logger.warning(f"OSC /radio/recording send failed: {exc}")

		if self._sampler_client is not None:
			try:
				self._sampler_client.send_message('/sample/import', [path_str])

			except (OSError, ValueError, TypeError) as exc:
				logger.warning(f"OSC /sample/import send failed: {exc}")

	def _on_state_event (self, **kwargs: typing.Any) -> None:

		"""Event adapter: converts kwargs to positional args for on_state_change."""

		self.on_state_change(
			kwargs['band'], kwargs['index'], kwargs['is_active'], kwargs['snr_db'],
			ctcss_hz=kwargs.get('ctcss_hz'), dcs_code=kwargs.get('dcs_code'),
		)

	def _on_recording_event (self, **kwargs: typing.Any) -> None:

		"""Event adapter: converts kwargs to positional args for on_recording_saved."""

		self.on_recording_saved(
			kwargs['band'], kwargs['index'], kwargs['file_path'],
			ctcss_hz=kwargs.get('ctcss_hz'), dcs_code=kwargs.get('dcs_code'),
		)

	def attach (self, scanner: "substation.scanner.RadioScanner") -> None:

		"""
		Subscribe this sender to a scanner's `channel_state` and
		`recording_saved` events, so each is sent as it happens.

		Args:
			scanner: The scanner to send events from.
		"""

		scanner.on('channel_state', self._on_state_event)
		scanner.on('recording_saved', self._on_recording_event)
