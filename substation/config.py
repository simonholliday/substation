"""
Configuration loading and validation for Substation.

This module handles loading configuration from YAML files and validates it using
Pydantic models. The configuration system supports:
- Type validation with helpful error messages
- Default values for optional parameters
- Band type templates (e.g., "DMR", "TETRA") that can be applied to multiple bands
- Per-band overrides of type defaults

Each setting's description is the docstring written directly under its field.
Every model sets use_attribute_docstrings, so those docstrings become the
descriptions in AppConfig.model_json_schema(), which the published
configuration reference is generated from.  A test fails if a setting has none.

Pydantic provides automatic type checking, validation, and clear error messages
when configuration is invalid, which is much better than runtime errors or silent
failures.
"""

import fractions
import logging
import pathlib
import typing

import pydantic
import yaml

import substation.constants
import substation.device_families

logger = logging.getLogger(__name__)

# Top-level sections that earlier releases accepted and this one no longer
# uses, each with the reason a user is told.  `substation --init` copied every
# section of the shipped file into the user's config.yaml, so rejecting these
# would stop configurations written that way from loading after an upgrade.
# They are dropped with a warning instead; misspelt keys are still rejected.
REMOVED_SECTIONS = {
	'supervisor': "the Supervisor dashboard integration has been removed",
}


def _fraction_constructor (loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> fractions.Fraction:
	"""
	YAML constructor for the !fraction tag.

	Converts a string like "25000/3" into a fractions.Fraction object.
	This allows representing precise radio frequencies that cannot be
	perfectly represented as floating-point numbers.
	"""

	value = loader.construct_scalar(node)
	return fractions.Fraction(value)


# Custom loader subclass so the !fraction tag doesn't mutate the global SafeLoader.
class _YamlLoader (yaml.SafeLoader):
	pass

_YamlLoader.add_constructor('!fraction', _fraction_constructor)


def _normalize_label (value: typing.Any) -> typing.Any:
	"""
	Normalize string labels to uppercase for case-insensitive matching.

	Used for modulation types ("NFM", "AM") and band types ("DMR", "TETRA")
	so users can write them in any case in the config file.

	Args:
		value: Input value (string, None, or other type)

	Returns:
		Uppercase string if input is string, otherwise original value
	"""

	if value is None:
		return value

	if isinstance(value, str):
		return value.strip().upper()

	return value


def _normalize_gain (value: typing.Any) -> typing.Any:
	"""
	Normalize gain values to either 'auto' or a float.

	Gain can be specified as:
	- None → 'auto'
	- "auto" (any case) → 'auto'
	- Number → float (e.g., 20.0 for 20 dB)

	This allows flexible configuration while ensuring consistent internal representation.

	Args:
		value: Input gain value (None, string, or number)

	Returns:
		Either 'auto' string or float value

	Raises:
		ValueError: If string value is not 'auto' and can't be parsed as number
	"""

	if value is None:
		return 'auto'

	if isinstance(value, str):
		text = value.strip().lower()

		if text == 'auto':
			return 'auto'

		try:
			return float(text)
		except ValueError as exc:
			raise ValueError("sdr_gain_db must be a number or 'auto'") from exc

	return float(value)

class ScannerConfig(pydantic.BaseModel):
	"""
	Scanner settings, applying to every band.

	These settings control how the scanner reads IQ samples from the SDR, how it
	buffers them, and whether it calibrates the receiver's frequency at startup.
	"""

	# Reject unknown fields (catch typos in config file), and publish each
	# field's docstring as its description in the JSON schema
	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	sdr_device_sample_size: int = pydantic.Field(default=131072, gt=0)
	"""
	Size, in IQ samples, of the blocks each slice is built from. The scanner
	rounds every slice up to a whole number of these blocks, so this also sets
	the shortest possible slice. On an RTL-SDR it must be a multiple of 256, for
	example 65536.
	"""

	band_time_slice_ms: int = pydantic.Field(default=200, gt=0)
	"""
	How often the scanner analyses the spectrum, in milliseconds. The scanner
	reads, queues, and processes IQ samples one slice at a time, and rounds each
	slice up to a whole number of `sdr_device_sample_size` blocks, so at low
	sample rates a slice can be longer than this. Shorter slices detect brief
	transmissions sooner; longer slices use less CPU and more memory.
	"""

	sample_queue_maxsize: int = pydantic.Field(default=200, gt=0)
	"""
	Number of slices of IQ samples the scanner holds while processing catches
	up. When the queue is full, the scanner drops newly arriving slices with a
	warning and keeps the slices already queued. A larger queue tolerates longer
	processing spikes, such as several radio channels activating at once. Each
	queued slice holds every IQ sample in it, so at high sample rates a full
	queue can need more memory than a small computer has: lower it there.
	"""

	calibration_frequency_hz: float | None = pydantic.Field(default=93.7e6, gt=0)
	"""
	Frequency in Hz of a strong, steady local signal, such as an FM broadcast
	station, that the scanner uses at startup to measure and correct the
	receiver's frequency error. If no strong, steady signal is found there, the
	scanner skips calibration with a warning and leaves the receiver's correction
	unchanged. Only RTL-SDR receivers have a PPM correction control; other devices
	skip calibration. Set to null to turn calibration off.
	"""

	stuck_channel_threshold_seconds: float | None = pydantic.Field(default=60.0, gt=0)
	"""
	Time in seconds after which the scanner warns that a radio channel has stayed
	active, which usually points to interference or a stuck transmitter. The
	scanner repeats the warning at most once a minute and keeps detecting and
	recording. Recommended: 30-120 seconds. Set to null to turn the warning off.
	"""


class DynamicsCurveConfig(pydantic.BaseModel):

	"""
	Settings for the experimental dynamics curve.

	The dynamics curve is a dual-region expander applied to each audio sample: a
	smoothstep S-curve reduces the level of quiet audio below the threshold,
	which suppresses noise, and a sin² hump gently boosts audio above the
	threshold, which gives voice more presence.
	"""

	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	threshold_dbfs: float = pydantic.Field(default=-25.0, lt=0.0)
	"""
	Level in dBFS that divides the cut region from the boost region. Typical
	values: -20 to -35 dBFS.
	"""

	cut_db: float = pydantic.Field(default=6.0, ge=0.0)
	"""
	Gain reduction in dB at the midpoint of the cut S-curve. The largest
	reduction, at `floor_dbfs`, is twice this value. Set to 0 to turn the cut
	region off.
	"""

	boost_db: float = pydantic.Field(default=1.5, ge=0.0)
	"""
	Largest gain boost in dB, at the peak of the boost hump. Set to 0 to turn the
	boost region off.
	"""

	floor_dbfs: float = pydantic.Field(default=-60.0, lt=0.0)
	"""
	Level in dBFS below which the output is silenced. Must be below
	`threshold_dbfs`.
	"""

	cut_curve: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
	"""
	Where the cut S-curve is steepest. 0.5 is symmetric; lower values move the
	steepest part towards `threshold_dbfs`, and higher values towards
	`floor_dbfs`.
	"""

	boost_curve: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
	"""
	Skew of the boost hump, on the same scale as `cut_curve`. 0.5 is symmetric.
	"""

	@pydantic.model_validator(mode='after')
	def _validate_levels (self) -> 'DynamicsCurveConfig':

		"""
		Cross-field validation for the level parameters.

		Enforces that the floor sits strictly below the threshold so the
		cut region has a non-zero width.  Logs a warning (does not raise)
		if the boost configuration could push the output above 0 dBFS —
		that case is also caught by a defensive clamp inside the function,
		but a startup warning gives the user a chance to dial it back
		before listening to the result.
		"""

		if self.floor_dbfs >= self.threshold_dbfs:
			raise ValueError(
				f"floor_dbfs ({self.floor_dbfs}) must be strictly below "
				f"threshold_dbfs ({self.threshold_dbfs})"
			)

		# Maximum boost output occurs near the midpoint of the boost
		# region, where the input level is roughly threshold_dbfs / 2 and
		# the boost is up to boost_db.  If that sum exceeds 0 dBFS the
		# defensive clamp inside apply_dynamics_curve will engage.
		boost_peak_dbfs = (self.threshold_dbfs / 2.0) + self.boost_db

		if boost_peak_dbfs > 0.0:
			logger.warning(
				f"dynamics_curve: boost_db={self.boost_db} dB combined with "
				f"threshold_dbfs={self.threshold_dbfs} would push the boost "
				f"region above 0 dBFS (~{boost_peak_dbfs:+.1f} dBFS at the peak); "
				f"the defensive clamp will engage and the curve will not match "
				f"the configured shape.  Reduce boost_db or lower threshold_dbfs."
			)

		return self


class RecordingConfig(pydantic.BaseModel):
	"""
	Recording settings, applying to every band that records.

	These settings control how the scanner buffers, processes, and saves the
	audio it demodulates.
	"""

	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	buffer_size_seconds: float = pydantic.Field(default=30.0, gt=0.0)
	"""
	Most audio, in seconds, the scanner holds in memory for each radio channel
	before writing it to disk. If disk writes fall behind, the scanner drops the
	oldest audio. A larger buffer tolerates slower disks and uses more memory.
	"""

	disk_flush_interval_seconds: float = pydantic.Field(default=5.0, gt=0.0)
	"""
	How often the scanner writes buffered audio to disk, in seconds. Shorter
	intervals hold less audio in memory and cost more disk activity.
	"""

	audio_sample_rate: int = pydantic.Field(default=16000, gt=2 * substation.constants.NFM_VOICE_LOWPASS_HZ)
	"""
	Sample rate of the recorded audio, in Hz. 16 kHz covers the voice band;
	higher rates keep more high-frequency content and use more disk space.
	"""

	audio_format: typing.Literal['wav', 'flac'] = 'wav'
	"""
	File format for recordings. WAV is uncompressed and embeds Broadcast WAV
	(BEXT) metadata with the time each recording starts, so audio editors such
	as Audacity, Reaper, and iZotope RX place each recording on a timeline at
	its real capture time. FLAC is lossless and typically 20-45% smaller than WAV,
	depending on the band and the signal, but cannot carry BEXT timeline
	metadata: the date, time, and frequency are stored as text tags.
	"""

	audio_output_dir: str = './audio'
	"""
	Directory where the scanner saves recordings, in a folder for each date and
	then each band: `<audio_output_dir>/YYYY-MM-DD/<band>/`.
	"""

	fade_in_ms: float | None = pydantic.Field(default=15.0, ge=0.0)
	"""
	Length of the fade-in at the start of each recording, in milliseconds, which
	prevents a click from a sudden onset. Set to 0 or null for no fade.
	"""

	fade_out_ms: float | None = pydantic.Field(default=50.0, ge=0.0)
	"""
	Length of the fade-out at the end of each recording, in milliseconds, which
	prevents a click from a sudden cutoff. Set to 0 or null for no fade.
	"""

	soft_limit_drive: float = pydantic.Field(default=1.25, gt=0.0)
	"""
	Drive of the soft limiter that keeps recordings from clipping. Higher values
	compress loud signals more strongly; lower values leave more of the dynamic
	range untouched.
	"""

	noise_reduction_enabled: bool = pydantic.Field(default=True)
	"""
	Whether the scanner applies spectral-subtraction noise reduction to recorded
	audio. Set to false to record the demodulated audio without it.
	"""

	recording_hold_time_ms: float = pydantic.Field(default=500.0, ge=0.0)
	"""
	Time in milliseconds the scanner keeps recording after a radio channel's
	signal drops below its off threshold.
	"""

	discard_empty_enabled: bool = pydantic.Field(default=True)
	"""
	Whether the scanner discards noise-only recordings, using spectral flatness
	analysis. It rejects noise triggers before a recording starts, and discards
	files that turn out to be mostly noise when they close.
	"""

	min_recording_seconds: float = pydantic.Field(default=0.5, ge=0.0)
	"""
	Shortest recording the scanner keeps, in seconds. It deletes shorter
	recordings when they close, which catches brief transients, such as radar
	pulses and ignition noise, that pass the spectral checks. Set to 0 to keep
	every recording.
	"""

	audio_silence_timeout_ms: float = pydantic.Field(default=3000.0, ge=0.0)
	"""
	Audio silence timeout in milliseconds. Stops recording when demodulated audio
	has been silent for this long, even if the RF carrier is still present
	(common on AM airband where the carrier persists after voice stops). Set to 0
	to rely on RF-only detection.
	"""

	trim_carrier_transients: bool = pydantic.Field(default=False)
	"""
	Whether the scanner removes the sharp clicks a transmitter makes at key-on
	and key-off. It trims only transients bordered by silence, so voice is never
	affected. Recommended for AM airband listening.
	"""

	dynamics_curve_enabled: bool = pydantic.Field(default=False)
	"""
	Whether the scanner applies the experimental dynamics curve, set by
	`dynamics_curve`, to recorded audio.
	"""

	dynamics_curve: DynamicsCurveConfig = pydantic.Field(default_factory=DynamicsCurveConfig)
	"""
	Settings for the experimental dynamics curve. The scanner uses them only when
	`dynamics_curve_enabled` is true.
	"""


class BandTypeConfig(pydantic.BaseModel):
	"""
	A template of settings shared by one type of radio service, such as DMR,
	TETRA, or PMR.

	A band names its template in `type` and inherits the settings the template
	gives, such as the radio channel spacing and modulation that every DMR band
	shares. A band overrides an inherited setting by giving it itself. Every
	setting in a template is optional, and a template passes on none that it
	leaves out or sets to null. Each band must still end up with the settings a
	band requires.
	"""

	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	channel_spacing: float | None = pydantic.Field(default=None, gt=0)
	"""
	Spacing between radio channel centre frequencies, in Hz, for example 12.5 kHz
	for PMR and DMR, or 25 kHz for marine VHF.
	"""

	sample_rate: float | None = pydantic.Field(default=None, gt=0)
	"""
	SDR sample rate in Hz. Must be high enough to cover the whole band plus
	margins, for example 2 MHz for the 188 kHz PMR446 band.
	"""

	channel_width: float | None = pydantic.Field(default=None, gt=0)
	"""
	Occupied bandwidth of each radio channel, in Hz. When null, the scanner uses
	84% of `channel_spacing`, which leaves guard bands between radio channels.
	"""

	modulation: str | None = None
	"""
	Modulation the scanner demodulates for recording: `NFM`, `AM`, `USB`, or
	`LSB`, in any letter case.
	"""

	recording_enabled: bool = False
	"""
	Whether the scanner records audio from detected transmissions. Set to false
	for detection only, with no audio files.
	"""

	snr_threshold_db: float | None = pydantic.Field(default=None)
	"""
	Signal-to-noise ratio in dB at which a radio channel counts as active. Lower
	values detect weaker signals and trigger more often on noise.
	"""

	sdr_gain_db: float | str | None = 'auto'
	"""
	SDR gain in dB, or `auto` for automatic gain control. Higher gain is more
	sensitive, and also amplifies noise and can cause clipping.
	"""

	@pydantic.field_validator('modulation', mode='before')
	@classmethod
	def _validate_label (cls, value: typing.Any) -> typing.Any:
		"""Normalize modulation labels to uppercase for case-insensitive matching."""
		return _normalize_label(value)

	@pydantic.field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain (cls, value: typing.Any) -> typing.Any:
		"""Normalize gain to 'auto' or float."""
		return _normalize_gain(value)


class DeviceOverrideConfig(pydantic.BaseModel):
	"""
	Settings that replace a band's own when the scanner runs on one family of SDR
	device.

	Every setting is optional. Only the settings given here replace the band's,
	and only when the device selected with `--device-type` is in that family.
	"""

	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	sample_rate: float | None = pydantic.Field(default=None, gt=0)
	"""
	SDR sample rate in Hz on this device. Must cover the band's span.
	"""

	sdr_gain_db: float | str | None = None
	"""
	SDR gain in dB on this device, or `auto` for automatic gain control.
	"""

	sdr_gain_elements: dict[str, float] | None = pydantic.Field(default=None, examples=[{'LNA': 10, 'MIX': 5, 'VGA': 12}])
	"""
	Gain in dB for each of this device's gain stages, keyed by stage name, for
	example `LNA`, `MIX`, and `VGA`.
	"""

	sdr_device_settings: dict[str, str] | None = pydantic.Field(default=None, examples=[{'biastee': 'true'}])
	"""
	Device-specific settings on this device, such as bias tee control, as string
	keys and values.
	"""

	snr_threshold_db: float | None = None
	"""
	Signal-to-noise ratio in dB at which a radio channel counts as active on this
	device.
	"""

	activation_variance_db: float | None = pydantic.Field(default=None, ge=0)
	"""
	Power variance in dB that a radio channel must show to count as active on
	this device. Set to 0 to turn the check off.
	"""

	@pydantic.field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain (cls, value: typing.Any) -> typing.Any:
		"""Normalize gain to 'auto' or float, preserving None as 'not overridden'."""
		if value is None:
			return None
		return _normalize_gain(value)


class BandConfig(pydantic.BaseModel):
	"""
	One band to scan.

	A band sets a frequency range, the spacing of its radio channels, and how the
	scanner treats them. The scanner monitors every radio channel in the band at
	once.
	"""

	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	freq_start: float = pydantic.Field(gt=0)
	"""
	Start of the band's frequency range in Hz, for example 446.00625 MHz for
	PMR446.
	"""

	freq_end: float = pydantic.Field(gt=0)
	"""
	End of the band's frequency range in Hz, for example 446.19375 MHz for
	PMR446. Must be above `freq_start`.
	"""

	channel_spacing: float = pydantic.Field(gt=0)
	"""
	Spacing between radio channel centre frequencies, in Hz, for example 12.5 kHz.
	The scanner places radio channels from `freq_start` to `freq_end` at this
	spacing.
	"""

	sample_rate: float = pydantic.Field(gt=0)
	"""
	SDR sample rate in Hz. Must be high enough to cover the band's span plus one
	radio channel's width and margins, for example 2 MHz for PMR446.
	"""

	channel_width: float | None = pydantic.Field(default=None, gt=0)
	"""
	Occupied bandwidth of each radio channel, in Hz. When null, the scanner uses
	84% of `channel_spacing`, which leaves 16% as guard bands.
	"""

	type: str | None = None
	"""
	Name of a template in `band_defaults`, such as `DMR` or `TETRA`, whose
	settings the band inherits.
	"""

	modulation: str | None = None
	"""
	Modulation the scanner demodulates: `NFM`, `AM`, `USB`, or `LSB`, in any
	letter case. Recording needs one; without it, the scanner only detects
	activity in the band.
	"""

	recording_enabled: bool = False
	"""
	Whether the scanner records audio from active radio channels. Set to false
	for detection only, with no audio files.
	"""

	exclude_channel_indices: list[int] = pydantic.Field(default_factory=list)
	"""
	Radio channel numbers the scanner skips, counting from 1, as shown in log
	output and filenames. Use it for known interference or out-of-band radio
	channels: `[1, 2]` skips the first two.
	"""

	snr_threshold_db: float = pydantic.Field(default=12.0)
	"""
	Signal-to-noise ratio in dB at which a radio channel turns on. Lower values,
	such as 8-10 dB, detect weaker signals and trigger more often on noise.
	"""

	hysteresis_db: float = pydantic.Field(default=3.0, ge=0)
	"""
	Margin in dB between the on and off thresholds. A radio channel turns on when
	its SNR rises above `snr_threshold_db`, and off when it falls below
	`snr_threshold_db` minus this margin, which stops it toggling while the SNR
	hovers near the threshold. Use a lower value, such as 1.5 dB, when scanning
	weak signals with a low `snr_threshold_db`.
	"""

	sdr_gain_db: float | str | None = 'auto'
	"""
	SDR gain in dB, or `auto` for automatic gain control. `auto` is convenient,
	but a manual gain, for example 20-40 dB on RTL-SDR, often works better.
	"""

	sdr_gain_elements: dict[str, float] | None = pydantic.Field(default=None, examples=[{'LNA': 10, 'MIX': 5, 'VGA': 12}])
	"""
	Gain in dB for each gain stage, on devices with several, keyed by stage name:
	the AirSpy R2, for example, has `LNA`, `MIX`, and `VGA`. Stage names depend on
	the device, and the scanner logs the available stages and their ranges at
	startup. When set, it takes priority over `sdr_gain_db`.
	"""

	sdr_device_settings: dict[str, str] | None = pydantic.Field(default=None, examples=[{'biastee': 'true'}])
	"""
	Device-specific settings the scanner passes to the SDR through SoapySDR's
	`writeSetting()`, such as bias tee control, an external clock, or device
	calibration. Keys and values are device-specific strings.
	"""

	activation_variance_db: float | None = pydantic.Field(default=None, ge=0)
	"""
	Power variance in dB, across a detection slice, that a radio channel must
	show to turn on. It suppresses triggers from stationary noise that crosses
	the SNR threshold with no real signal: voice and data vary by 5-15 dB or more
	over a slice, and stationary noise by under 2 dB. It applies whether or not
	the band records. When null, the scanner uses
	`substation.constants.ACTIVATION_VARIANCE_DB`. Set to 0 to turn the check
	off.
	"""

	device_overrides: dict[str, DeviceOverrideConfig] | None = pydantic.Field(
		default=None,
		examples=[{'airspy': {'sample_rate': 2.5e6, 'sdr_gain_elements': {'LNA': 14, 'MIX': 5, 'VGA': 12}}}],
	)
	"""
	Settings that replace this band's own on one family of SDR device, keyed by
	family: `rtlsdr`, `hackrf`, `airspy`, `airspyhf`, or a SoapySDR driver name.
	A key can use any `--device-type` spelling of its family, in any letter
	case. When the scanner runs with a matching `--device-type`, it merges
	those settings onto the band's. A key that names no family the scanner
	knows logs a warning, unless it is written as `soapy:` followed by a
	SoapySDR driver name.
	"""

	@pydantic.field_validator('modulation', 'type', mode='before')
	@classmethod
	def _validate_label (cls, value: typing.Any) -> typing.Any:
		"""Normalize labels to uppercase for case-insensitive matching."""
		return _normalize_label(value)

	@pydantic.field_validator('device_overrides', mode='before')
	@classmethod
	def _normalize_override_keys (cls, value: typing.Any) -> typing.Any:

		"""
		Key each device override by its device family.

		The scanner looks overrides up by family, so a key written as another
		spelling (`AirSpy`, `rtl-sdr`, `airspy-hf`) was silently never used.
		Keys that name the same family are merged in the order they appear.
		A key naming no known family is kept, since it may be a SoapySDR
		driver, but warned about unless its `soapy:` prefix says so.
		"""

		if not isinstance(value, dict):
			return value

		normalized: dict[typing.Any, typing.Any] = {}

		for key, settings in value.items():

			if not isinstance(key, str):
				normalized[key] = settings
				continue

			family = substation.device_families.normalize_device_family(key)
			written_as_soapy = key.strip().lower().startswith(substation.device_families.SOAPY_PREFIX)

			if family not in substation.device_families.KNOWN_DEVICE_FAMILIES and not written_as_soapy:
				logger.warning(
					f"device_overrides key '{key}' names no device type the scanner knows, so it applies only with "
					f"--device-type soapy:{family}. If that is intended, write the key as 'soapy:{family}'."
				)

			existing = normalized.get(family)
			if isinstance(existing, dict) and isinstance(settings, dict):
				normalized[family] = _deep_merge(existing, settings)
			else:
				normalized[family] = settings

		return normalized

	@pydantic.field_validator('exclude_channel_indices', mode='before')
	@classmethod
	def _validate_exclusions (cls, value: typing.Any) -> list[int]:
		"""
		Validate and normalize channel exclusion list.

		Converts None to empty list, validates that all values are valid
		1-based channel numbers (the numbers shown in logs and filenames).
		"""
		if value is None:
			return []

		if not isinstance(value, list):
			raise ValueError("exclude_channel_indices must be a list of integers")

		indices: list[int] = []
		for item in value:
			idx = int(item)

			if idx < 1:
				raise ValueError("exclude_channel_indices entries are 1-based channel numbers and must be >= 1")

			indices.append(idx)

		return indices

	@pydantic.field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain (cls, value: typing.Any) -> typing.Any:
		"""Normalize gain to 'auto' or float."""
		return _normalize_gain(value)

	@property
	def required_bandwidth (self) -> float:

		"""
		Bandwidth in Hz that the SDR must capture to scan this band: its span,
		plus one radio channel's width and half a channel spacing at each edge.
		"""

		channel_width = self.channel_width
		if channel_width is None:
			channel_width = self.channel_spacing * substation.constants.CHANNEL_WIDTH_FRACTION

		return self.freq_end - self.freq_start + channel_width + self.channel_spacing

	@pydantic.model_validator(mode='after')
	def _validate_band (self) -> 'BandConfig':
		"""
		Cross-field validation after all fields are parsed.

		Validates:
		- freq_end > freq_start
		- Sets default channel_width if not specified
		- Ensures snr_threshold_db is high enough for hysteresis to work
		- Warns if sdr_gain_elements overrides sdr_gain_db
		"""
		if self.freq_start >= self.freq_end:
			raise ValueError('freq_start must be less than freq_end')

		# Default channel width to 84% of spacing (leaves guard bands)
		if self.channel_width is None:
			self.channel_width = self.channel_spacing * substation.constants.CHANNEL_WIDTH_FRACTION

		# Recording needs a demodulator.  A label with none, such as a typo
		# for NFM, would otherwise turn recording off with only an INFO line.
		if self.recording_enabled and self.modulation not in substation.constants.DEMODULATED_MODULATIONS:
			logger.warning(
				f"recording_enabled is true, but modulation {self.modulation!r} has no demodulator, so this band "
				f"detects transmissions without recording them. Recording works with: {', '.join(substation.constants.DEMODULATED_MODULATIONS)}."
			)

		# Warn if SNR threshold is at or below hysteresis margin.
		# OFF threshold = snr_threshold_db - hysteresis_db, which can go
		# negative (means "turn off at noise floor") — valid for weak signals.
		if self.snr_threshold_db <= self.hysteresis_db:
			logger.warning(
				f"snr_threshold_db ({self.snr_threshold_db}) <= hysteresis_db ({self.hysteresis_db}): "
				f"OFF threshold will be {self.snr_threshold_db - self.hysteresis_db:.1f} dB "
				f"(channel turns off at noise floor level)"
			)

		# Which of the two gains applies depends on the device, known only
		# when a scan starts, and the scanner logs it then.
		if self.sdr_gain_elements is not None and self.sdr_gain_db is not None:
			logger.debug(
				"sdr_gain_elements is set: a device with per-element gain uses it in place of "
				"sdr_gain_db, and any other device uses sdr_gain_db"
			)

		return self


class AppConfig(pydantic.BaseModel):
	"""
	The whole configuration.

	A configuration file has four top-level sections: `scanner`, `recording`,
	`band_defaults`, and `bands`.
	"""

	model_config = pydantic.ConfigDict(extra='forbid', use_attribute_docstrings=True)

	scanner: ScannerConfig
	"""
	Scanner settings, applying to every band.
	"""

	recording: RecordingConfig = pydantic.Field(default_factory=RecordingConfig)
	"""
	Recording settings, applying to every band that records.
	"""

	band_defaults: dict[str, BandTypeConfig] = pydantic.Field(default_factory=dict)
	"""
	Templates of settings for types of radio service, keyed by type name. A band
	inherits a template's settings by naming it in `type`.
	"""

	bands: dict[str, BandConfig]
	"""
	The bands to scan, keyed by band name. At least one band is required.
	"""

	@pydantic.model_validator(mode='after')
	def _validate_bands (self) -> 'AppConfig':
		"""Ensure at least one band is configured."""
		if not self.bands:
			raise ValueError('bands must contain at least one band')

		return self


def _deep_merge (
	base: dict[str, typing.Any],
	override: dict[str, typing.Any],
) -> dict[str, typing.Any]:

	"""Recursively merge *override* onto *base*, returning a new dict.

	For each key in override: if both values are dicts, recurse; otherwise the
	override value wins (including explicit None / YAML null).  One exception:
	when the base value is a dict and the override is None — e.g. a user
	config listing a section header with no keys under it — the base dict is
	kept rather than wiped out.  Keys present in base but absent from override
	are preserved unchanged.  Neither input is mutated.
	"""

	result = dict(base)

	for key, override_value in override.items():
		base_value = result.get(key)
		if isinstance(base_value, dict) and isinstance(override_value, dict):
			result[key] = _deep_merge(base_value, override_value)
		elif isinstance(base_value, dict) and override_value is None:
			pass
		else:
			result[key] = override_value

	return result


def _locate_default_config () -> pathlib.Path:

	"""Return the path to the bundled config.yaml.default.

	The file ships as package data inside the substation package (declared
	in pyproject.toml under [tool.setuptools.package-data]), so a plain
	__file__-relative path resolves identically for editable installs and
	wheels installed from PyPI or a Git URL — no git checkout required.
	A real filesystem Path is returned deliberately (rather than
	importlib.resources): callers read and copy the file, and the package
	never runs from a zipped location.

	Raises FileNotFoundError if the file is missing (broken installation).
	"""

	default = pathlib.Path(__file__).parent / "config.yaml.default"

	if not default.exists():
		raise FileNotFoundError(
			f"Bundled config.yaml.default not found at {default}. "
			"The package installation may be corrupted."
		)

	return default


def _resolve_user_config_path (
	explicit: str | pathlib.Path | None,
) -> pathlib.Path | None:

	"""Return the user's config override path, or None if no user config exists.

	Priority: explicit path argument → ./config.yaml in CWD → None.
	When an explicit path is provided it must exist; no CWD fallback is tried.
	"""

	if explicit is not None:

		# Accept plain strings too — load_config is the public entry point
		# for module users and a str path is the natural thing to pass.
		explicit = pathlib.Path(explicit)

		if explicit.exists():
			return explicit

		raise FileNotFoundError(f"Config file not found: {explicit}")

	cwd_config = pathlib.Path.cwd() / "config.yaml"
	if cwd_config.exists():
		return cwd_config

	return None


def _load_raw_config (config_path: pathlib.Path) -> dict:

	"""Load raw configuration data from YAML file.

	Performs basic validation (file exists, contains valid YAML, root is a dict)
	but doesn't validate the structure or types yet (that's done by Pydantic).
	"""

	with open(config_path, 'r') as f:
		data = yaml.load(f, Loader=_YamlLoader)

	if data is None:
		raise ValueError(f"Config file is empty: {config_path}")

	if not isinstance(data, dict):
		raise ValueError('Config root must be a mapping')

	return data


def _drop_removed_sections (data: dict, source: str) -> dict:

	"""
	Return data without the top-level sections in REMOVED_SECTIONS.

	Logs a warning for each one found, naming where it was found and why it
	is no longer used.  The input is not changed.
	"""

	found = [key for key in data if key in REMOVED_SECTIONS]

	for key in found:
		logger.warning(f"Ignoring the '{key}' section in {source}: {REMOVED_SECTIONS[key]}. Delete the section to stop this warning.")

	return {key: value for key, value in data.items() if key not in REMOVED_SECTIONS}


def _normalize_template_names (data: dict) -> dict:

	"""
	Return data with its band_defaults template names in upper case.

	Template names match a band's type in any letter case, so `air` and
	`AIR` are one template.  Normalising before the shipped file and the
	user's file are merged lets a user's `air` adjust the shipped `AIR`
	rather than sit beside it and replace it whole.  Names that collide
	within one file are merged in the order they appear.  The input is not
	changed.
	"""

	templates = data.get('band_defaults')
	if not isinstance(templates, dict):
		return data

	normalized: dict[typing.Any, typing.Any] = {}

	for name, settings in templates.items():
		key = name.strip().upper() if isinstance(name, str) else name
		existing = normalized.get(key)

		if isinstance(existing, dict) and isinstance(settings, dict):
			normalized[key] = _deep_merge(existing, settings)
		else:
			normalized[key] = settings

	return {**data, 'band_defaults': normalized}


def _apply_band_defaults (data: dict) -> dict:

	"""
	Apply band type defaults to individual bands.

	This implements the template/inheritance system: if a band specifies
	type: "DMR", it inherits all defaults from band_defaults.DMR, with
	explicit band values taking precedence.

	Example:
		band_defaults:
			DMR:
				channel_spacing: 12500
				modulation: "NFM"
		bands:
			dmr_band1:
				type: "DMR"          # Inherits channel_spacing and modulation
				freq_start: 446e6
				freq_end: 447e6

	The merging is shallow (no recursive merging of nested dicts).

	Args:
		data: Raw configuration dictionary

	Returns:
		Configuration with band defaults merged into individual bands
	"""

	defaults_raw = data.get('band_defaults')
	bands_raw = data.get('bands')

	# Skip if no defaults or bands defined
	if not isinstance(defaults_raw, dict) or not isinstance(bands_raw, dict):
		return data

	# Normalize type names to uppercase for case-insensitive matching
	normalized_types: dict[str, dict] = {}
	for type_name, type_defaults in defaults_raw.items():
		if not isinstance(type_name, str) or not isinstance(type_defaults, dict):
			continue

		normalized_types[type_name.strip().upper()] = type_defaults

	# Merge defaults into each band that specifies a type
	merged_bands: dict[str, dict | typing.Any] = {}
	for band_name, band_config in bands_raw.items():
		if not isinstance(band_config, dict):
			merged_bands[band_name] = band_config
			continue

		band_type = band_config.get('type')

		# If band has a type, merge in the defaults for that type
		if isinstance(band_type, str):
			type_key = band_type.strip().upper()
			type_defaults = normalized_types.get(type_key)

			if isinstance(type_defaults, dict):
				# Defaults first, then band config (band values override).  A
				# template setting that is null is not passed on: in a template,
				# null means the template leaves that setting to the band.
				merged = {key: value for key, value in type_defaults.items() if value is not None}
				merged.update(band_config)
				merged_bands[band_name] = merged
				continue

		# No type or type not found: use band config as-is
		if isinstance(band_type, str) and band_type.strip().upper() not in normalized_types:
			available = ', '.join(sorted(normalized_types.keys())) or '(none)'
			logger.warning(
				f"Band '{band_name}' specifies type '{band_type}' which is not defined in band_defaults. "
				f"Available types: {available}. No defaults will be inherited."
			)
		merged_bands[band_name] = band_config

	# Return modified config with merged bands
	merged = dict(data)
	merged['band_defaults'] = normalized_types
	merged['bands'] = merged_bands

	return merged


def load_config (path: str | pathlib.Path | None = None) -> AppConfig:

	"""Load configuration, merging config.yaml.default with config.yaml.

	Always loads config.yaml.default as the base.  If a user config.yaml exists
	(or an explicit path is given, as a str or pathlib.Path), it is deep-merged
	on top so user settings override defaults while unspecified keys inherit
	default values.
	"""

	default_path = _locate_default_config()
	base = _normalize_template_names(_load_raw_config(default_path))

	user_path = _resolve_user_config_path(path)

	if user_path is not None and user_path.resolve() == default_path.resolve():
		user_path = None

	if user_path is not None:
		user = _normalize_template_names(_drop_removed_sections(_load_raw_config(user_path), str(user_path)))
		raw = _deep_merge(base, user)
		logger.debug("Loaded config from %s + %s", default_path.name, user_path.name)
	else:
		raw = base
		logger.debug("Loaded config from %s (no user overrides)", default_path.name)

	data = _apply_band_defaults(raw)

	return AppConfig.model_validate(data)


def validate_config (config: dict | AppConfig) -> AppConfig:

	"""
	Validate configuration data and return a typed AppConfig.

	Can accept either a raw dict (from YAML) or an existing AppConfig.
	Useful for testing or programmatic configuration.

	Args:
		config: Configuration as dict or AppConfig

	Returns:
		Validated AppConfig object

	Raises:
		pydantic.ValidationError: If config values are invalid
	"""

	if isinstance(config, AppConfig):
		return config

	return AppConfig.model_validate(_apply_band_defaults(_drop_removed_sections(config, "the configuration")))


def get_band_config (config: dict | AppConfig, band_name: str) -> BandConfig:

	"""
	Extract configuration for a specific band.

	Validates the config if needed, then returns the configuration for
	the requested band.

	Args:
		config: Configuration as dict or AppConfig
		band_name: Name of the band (key in config.bands)

	Returns:
		Configuration for the specified band

	Raises:
		KeyError: If band name not found in configuration
		pydantic.ValidationError: If config is invalid
	"""

	typed_config = validate_config(config)

	if band_name not in typed_config.bands:
		available_bands = ', '.join(typed_config.bands.keys())
		raise KeyError(f"Band '{band_name}' not found in configuration. Available bands: {available_bands}")

	return typed_config.bands[band_name]
