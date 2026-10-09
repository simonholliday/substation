"""
Substation - Software-defined radio band scanner.

A Python application for scanning and recording activity on radio bands.

Supported hardware:
- RTL-SDR (native driver)
- HackRF One (native driver)
- Airspy R2, Airspy HF+ Discovery, and any other SoapySDR-supported
  device (via the SoapySDR wrapper)

Features:
- Automatic channel detection using SNR (Signal-to-Noise Ratio) with
  per-band configurable hysteresis and layered noise rejection (RF
  power variance and audio spectral flatness at turn-on, then minimum
  length and spectral flatness after recording) to reject false
  recordings; at present the flatness checks reject hiss only on AM
  bands
- Audio silence timeout to stop recording when an AM carrier persists
  after voice ends
- Demodulation of NFM, AM, and SSB (USB/LSB via Weaver's method) with
  streaming polyphase FIR resampler for artefact-free block processing
- CTCSS (51 standard tones) and DCS (23-bit Golay-coded) subaudible tone
  detection on NFM, embedded in recording metadata; not yet thoroughly
  tested with real radios
- A reception class on every shipped band, saying how UK law treats
  listening to it; out of the box only general-reception bands, such as
  amateur and CB radio, record, and the rest only detect activity
- Automatic per-channel recording in WAV (Broadcast WAV with embedded
  frequency / timestamp / modulation / tone metadata) or FLAC (lossless,
  Vorbis comments) with spectral-subtraction noise reduction and optional
  experimental dynamics-curve expander
- PPM frequency calibration against a known reference signal
- Unified event emitter (on / off / emit) — six events covering channel
  state, recording lifecycle, noise floor, and per-slice SNR snapshots;
  the channel_state event carries any CTCSS tone or DCS code detected
  as a property of each activation; used by the OSC bridge
- OSC event forwarding to downstream tools (MIDI sequencer, sampler,
  VJ software, ...), switched on by the osc settings in config.yaml, or
  from Python with substation.osc_sender

Typical usage:
    substation --init                              # Write a starter config.yaml
    substation --band amateur_2m
    substation --list-bands
    substation --band air_civil_1 --device-type hackrf
    substation --band air_civil_bristol --device-type airspyhf
"""

import importlib
import typing

if typing.TYPE_CHECKING:

	# The documented Python interface, declared where the source can be read
	# for it: subsystem.co reads these assignments by parsing, never by
	# importing, and generates the Python reference from them (#4718).  At
	# run time __getattr__ supplies them instead, so that importing
	# substation.config stays light, with neither NumPy nor python-osc.

	import substation.config
	import substation.osc_sender
	import substation.scanner

	load_config = substation.config.load_config
	RadioScanner = substation.scanner.RadioScanner
	OscEventSender = substation.osc_sender.OscEventSender


__all__ = ["load_config", "RadioScanner", "OscEventSender"]

# The module each name in __all__ is defined in, for __getattr__.
_EXPORTS = {
	"load_config": "substation.config",
	"RadioScanner": "substation.scanner",
	"OscEventSender": "substation.osc_sender",
}


def __getattr__ (name: str) -> typing.Any:

	"""Import a name in __all__ from its module the first time it is used, as in `substation.RadioScanner`."""

	module = _EXPORTS.get(name)

	if module is None:
		raise AttributeError(f"module 'substation' has no attribute {name!r}")

	return getattr(importlib.import_module(module), name)
