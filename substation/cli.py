"""
Command-line interface for Substation.

Provides a simple command-line tool for running the scanner with arguments
for selecting bands, SDR devices, and listing available configuration.

Typical usage:
	substation --init                              # Write a starter config.yaml
	substation --band amateur_2m                   # Scan the 2m amateur band
	substation --list-bands                        # Show available bands
	substation --band air_civil_bristol --device-type hackrf  # Use HackRF
	substation --band pmr --iq-file recording.wav --center-freq 446059313
"""

import argparse
import asyncio
import datetime
import logging
import pathlib
import sys

import substation
import substation.config
import substation.scanner

logger = logging.getLogger(__name__)

def list_bands (config_path: pathlib.Path | None) -> None:

	"""
	Display a summary of all bands defined in the configuration.

	Loads the configuration and prints a formatted table showing each band's
	name, frequency range, modulation type, and channel spacing. Useful for
	discovering what bands are available before starting a scan.

	Args:
		config_path: Optional path to user config override file

	Exits:
		Exits with code 1 if configuration cannot be loaded
	"""

	try:
		config_data = substation.config.load_config(config_path)
		bands = config_data.bands

		print("\nAvailable bands:")
		print("=" * 60)

		for band_name, band_config in bands.items():
			# Convert frequencies to MHz for readability
			freq_start = band_config.freq_start / 1e6
			freq_end = band_config.freq_end / 1e6
			modulation = band_config.modulation or 'Unknown'
			# Convert spacing to kHz for readability
			channel_spacing = band_config.channel_spacing / 1e3

			print(f"\n{band_name}:")
			print(f"  Frequency range: {freq_start:.3f} - {freq_end:.3f} MHz")
			print(f"  Modulation: {modulation}")
			print(f"  Radio channel spacing: {channel_spacing:.1f} kHz")
			if band_config.reception_class:
				print(f"  Reception class: {band_config.reception_class}")
			print(f"  Records: {'yes' if band_config.recording_enabled else 'no, detection only'}")

			# A band that cannot fit its own sample rate cannot be scanned, and
			# would otherwise fail only after its device had been opened.
			if band_config.required_bandwidth > band_config.sample_rate:
				print(f"  Cannot be scanned: needs {band_config.required_bandwidth / 1e6:.3f} MHz, but its sample rate captures {band_config.sample_rate / 1e6:.3f} MHz")

		print()

	except Exception as e:
		print(f"Error loading configuration: {e}", file=sys.stderr)
		sys.exit(1)

def init_config () -> None:

	"""
	Scaffold a starter config.yaml in the current directory, then exit.

	Copies the bundled, fully-commented default configuration to
	./config.yaml.  This gives users who installed from PyPI or a Git URL
	(with no repository checkout) an editable starting point — every band
	and setting is present and documented, and anything left untouched
	falls back to the same built-in default at runtime.

	Refuses to overwrite an existing config.yaml, and refuses to run inside
	the substation source tree so it can never scatter a config into the
	repo itself.

	Exits:
		Exits with code 1 if config.yaml already exists or if run from the
		source checkout.
	"""

	cwd = pathlib.Path.cwd()

	# Guard against scaffolding into the substation source checkout — the
	# repo is the application, not a place for a user config.
	if (cwd / "substation" / "__init__.py").exists():
		print(
			"This looks like the substation source repository (it contains the "
			"substation/ package). Run --init from the directory where you want "
			"your config.yaml instead.",
			file=sys.stderr,
		)
		sys.exit(1)

	target = cwd / "config.yaml"

	if target.exists():
		print(
			f"Refusing to overwrite existing {target}: nothing was created. "
			"Edit it directly, or move it aside and re-run --init.",
			file=sys.stderr,
		)
		sys.exit(1)

	default_text = substation.config._locate_default_config().read_text(encoding='utf-8')
	target.write_text(default_text, encoding='utf-8')

	print(f"Created {target}")
	print("This is the full default configuration, fully commented. Edit it to suit")
	print("your hardware and bands, then start a scan, e.g. `substation --band amateur_2m`.")
	print("Any setting you delete falls back to the built-in default.")


def _report_failure (what: str, exc: Exception) -> None:

	"""
	Log why a scan failed, with its traceback only at DEBUG.

	Most failures concern the receiver, such as one that is missing or has
	been unplugged, and the message says what happened.  A traceback helps
	only with a fault in Substation itself, so it waits for --log-level
	DEBUG, where someone reporting a bug can ask for it.
	"""

	if logger.isEnabledFor(logging.DEBUG):
		logger.error(f"{what}: {exc}", exc_info=True)
	else:
		logger.error(f"{what}: {exc} (--log-level DEBUG shows where it happened)")


async def run_scanner (config_path: pathlib.Path | None, band_name: str, device_type: str, device_index: int) -> None:

	"""
	Initialize and run the scanner with a live SDR device.

	Args:
		config_path: Optional path to user config override file
		band_name: Name of the band to scan (must exist in config.bands)
		device_type: SDR device type, as given to --device-type
		device_index: Device index for multi-device setups (0 for first device)

	Exits:
		Exits with code 1 if configuration is invalid, the band doesn't exist,
		or the scan stops because of an error
	"""

	try:
		config_data = substation.config.load_config(config_path)

		if not band_name:
			logger.error("No band specified. Use --band to select a band.")
			sys.exit(1)

		if band_name not in config_data.bands:
			available = ', '.join(config_data.bands.keys())
			logger.error(f"Band '{band_name}' not found. Available bands: {available}")
			sys.exit(1)

		scan = substation.scanner.RadioScanner(
			config=config_data,
			band_name=band_name,
			device_type=device_type,
			device_index=device_index
		)

		await scan.scan ()

	except Exception as e:
		_report_failure("Error running scanner", e)
		sys.exit(1)


async def run_scanner_file (config_path: pathlib.Path | None, band_name: str, iq_file: str, center_freq: float, start_time: datetime.datetime) -> None:

	"""
	Process an IQ WAV file through the scanner pipeline.

	Streams the file at full speed (no real-time pacing) using a virtual
	clock that advances with sample position.  Output recordings use the
	virtual timestamps for directory and file naming.

	Args:
		config_path: Optional path to user config override file
		band_name: Name of the band to scan
		iq_file: Path to 2-channel IQ WAV file
		center_freq: Center frequency of the recording in Hz
		start_time: Start datetime for the recording (used for output timestamps)

	Exits:
		Exits with code 1 if configuration is invalid, the band doesn't exist,
		or playback stops because of an error
	"""

	try:
		config_data = substation.config.load_config(config_path)

		if band_name not in config_data.bands:
			available = ', '.join(config_data.bands.keys())
			logger.error(f"Band '{band_name}' not found. Available bands: {available}")
			sys.exit(1)

		# Read sample rate from the WAV file to initialise the virtual clock
		import soundfile
		info = soundfile.info(iq_file)
		file_sample_rate = float(info.samplerate)

		clock = substation.scanner.VirtualClock(start_time, file_sample_rate)

		scan = substation.scanner.RadioScanner(
			config=config_data,
			band_name=band_name,
			device_type='file',
			clock=clock,
			device_kwargs={
				'file_path': iq_file,
				'center_freq': center_freq,
			},
		)

		logger.info(
			f"IQ file playback: {iq_file}, "
			f"center {center_freq/1e6:.6f} MHz, "
			f"start {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
		)

		await scan.scan()

	except Exception as e:
		_report_failure("Error processing IQ file", e)
		sys.exit(1)


def parser () -> argparse.ArgumentParser:

	"""
	Build the parser for `substation`, without parsing anything.

	Building is kept apart from parsing so that subsystem.co can read every
	option and its help without running a scan, and generate the published
	command-line reference from them (#4717).  The help is therefore where a
	fact about one option belongs, rather than the README.
	"""

	command = argparse.ArgumentParser(
		prog='substation',
		description='An SDR band scanner that detects, demodulates, and records radio transmissions.',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  substation --init                        # Write a starter config.yaml here
  substation --band amateur_2m             # Scan the 2m amateur band with RTL-SDR
  substation --band marine_vhf_calling --device-type hackrf  # Scan marine VHF with HackRF
  substation --list-bands                  # List all available bands
  substation --band pmr --iq-file rec.wav --center-freq 446059313  # File playback

Exit status:
  0  The scan ended: an IQ recording played to its end, or Ctrl+C stopped
     it. --init and --list-bands exit 0 when they succeed.
  1  A scan stopped because of an error, such as a receiver that fails or
     is unplugged, so a service manager can restart it. Also when Substation
     cannot start: --band is missing or names no band, an option it needs is
     missing or malformed, the configuration or the IQ recording cannot be
     read, or --init finds a config.yaml already there.
  2  The command line was not understood: an unknown option, or a value of
     the wrong kind.
"""
	)

	# User config override file (merged on top of config.yaml.default)
	command.add_argument(
		'--config', '-c',
		default=None,
		help='Your configuration file, merged over the shipped defaults (default: config.yaml in the current directory, if there is one)'
	)

	# Which band to scan (required for scanning, not for --list-bands)
	command.add_argument(
		'--band', '-b',
		default=None,
		help='The band to scan, by the name --list-bands shows. Required unless --init or --list-bands is given.'
	)

	# Which SDR hardware to use
	command.add_argument(
		'--device-type', '-t',
		default='rtlsdr',
		help='The receiver: rtlsdr, hackrf, airspy, airspyhf, or soapy:<driver> for any other SoapySDR device (default: rtlsdr)'
	)

	# Device index for systems with multiple SDRs
	command.add_argument(
		'--device-index', '-i',
		type=int,
		default=0,
		help='Which receiver of that type to use, counting from 0, when more than one is plugged in (default: 0)'
	)

	# Utility flag to list available bands
	command.add_argument(
		'--list-bands',
		action='store_true',
		help=(
			"List the available bands, with each band's reception class and whether it records, and exit. "
			"A band too wide for its receiver to capture at its configured rate is marked as one that cannot "
			"be scanned: narrow it, or split it into several bands, in your own configuration."
		)
	)

	# Scaffold a starter config.yaml in the current directory
	command.add_argument(
		'--init',
		action='store_true',
		help='Create a starter config.yaml (a copy of the documented defaults) in the current directory and exit. Refuses to overwrite an existing file.'
	)

	# IQ file playback
	command.add_argument(
		'--iq-file',
		default=None,
		help=(
			"Play back an IQ recording in place of a receiver: a WAV file whose two audio channels hold I and Q "
			"as 16-bit PCM, at any number of IQ samples per second, which is read from the file. RF64 and "
			"WAVE_FORMAT_EXTENSIBLE files work, and so do files over 4 GB whose header sizes have overflowed. "
			"The file is read as fast as it can be processed, not in real time, and each recording takes its "
			"time from --start-time. A band wider than the file can hold is refused."
		)
	)

	command.add_argument(
		'--center-freq',
		type=float,
		default=None,
		help="The frequency, in Hz, the receiver was tuned to when it made the IQ recording. Required with --iq-file. It need not be the band's centre."
	)

	command.add_argument(
		'--start-time',
		default=None,
		help='When the IQ recording began, as "YYYY-MM-DD HH:MM:SS": the time and filename of each recording count from it (default: 2000-01-01 00:00:00)'
	)

	# How much the scanner logs; DEBUG adds what each device reports about
	# itself at startup, such as its gain elements
	command.add_argument(
		'--log-level',
		default='INFO',
		choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
		type=str.upper,
		help=(
			"How much to log: DEBUG, INFO, WARNING, or ERROR (default: INFO). DEBUG adds what each device "
			"reports about itself at startup, such as its gain elements, and, when a scan fails, where in "
			"Substation it failed."
		)
	)

	return command


def main () -> int:

	"""
	Main entry point for the command-line interface.

	Parses command-line arguments, sets up logging, and dispatches to either
	list_bands(), run_scanner(), or run_scanner_file() based on the arguments.

	Returns:
		Exit code: 0 for success, 1 for error
	"""

	args = parser().parse_args()

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format='%(asctime)s - %(levelname)s - %(message)s'
	)

	config_path = pathlib.Path(args.config) if args.config else None

	# Scaffold a starter config.yaml and exit (doesn't require --band)
	if args.init:
		init_config()
		return 0

	# Handle --list-bands mode (doesn't require --band)
	if args.list_bands:
		list_bands(config_path)
		return 0

	# Validate that --band is provided for scanning mode
	if not args.band:
		print("Error: --band is required unless using --list-bands.", file=sys.stderr)
		return 1

	# IQ file playback mode
	if args.iq_file:
		if args.center_freq is None:
			print("Error: --center-freq is required with --iq-file.", file=sys.stderr)
			return 1

		if args.start_time:
			try:
				start_dt = datetime.datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
			except ValueError:
				print('Error: --start-time must be "YYYY-MM-DD HH:MM:SS".', file=sys.stderr)
				return 1
		else:
			start_dt = datetime.datetime(2000, 1, 1)

		try:
			asyncio.run(run_scanner_file(
				config_path=config_path,
				band_name=args.band,
				iq_file=args.iq_file,
				center_freq=args.center_freq,
				start_time=start_dt,
			))
			return 0
		except KeyboardInterrupt:
			return 0
		except Exception:
			return 1

	# Live SDR scanning mode
	try:
		asyncio.run(run_scanner(config_path=config_path, band_name=args.band, device_type=args.device_type, device_index=args.device_index))
		return 0

	except KeyboardInterrupt:
		return 0

	except Exception:
		return 1

if __name__ == '__main__':
	sys.exit(main())
