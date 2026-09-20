"""
Device type names and the device families they belong to.

The CLI's --device-type accepts several spellings for each kind of SDR, and a
band's device_overrides is keyed by the same names.  Both resolve through
normalize_device_family() to one canonical family name.

This module imports nothing, so substation.config can use it: importing the
configuration must never load numpy or a hardware binding.
"""

DEVICE_FAMILY_ALIASES: dict[str, str] = {
	'rtl': 'rtlsdr', 'rtlsdr': 'rtlsdr', 'rtl-sdr': 'rtlsdr',
	'hackrf': 'hackrf', 'hackrf-one': 'hackrf', 'hackrfone': 'hackrf',
	'airspy': 'airspy', 'airspy-r2': 'airspy', 'airspyr2': 'airspy',
	'airspyhf': 'airspyhf', 'airspy-hf': 'airspyhf', 'airspyhf+': 'airspyhf',
	'file': 'file',
}

KNOWN_DEVICE_FAMILIES: frozenset[str] = frozenset(DEVICE_FAMILY_ALIASES.values())

SOAPY_PREFIX = 'soapy:'


def normalize_device_family (device_type: str) -> str:

	"""Return the canonical device family name for a device type string.

	Maps all aliases to a canonical name (e.g. 'rtl', 'rtl-sdr' → 'rtlsdr'),
	in any letter case.  For 'soapy:<driver>' strings, returns the driver
	name.  Anything else is returned in lower case, as a SoapySDR driver name.
	"""

	key = device_type.strip().lower()

	if key in DEVICE_FAMILY_ALIASES:
		return DEVICE_FAMILY_ALIASES[key]

	if key.startswith(SOAPY_PREFIX):
		return key[len(SOAPY_PREFIX):]

	return key
