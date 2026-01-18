"""
Configuration loading and validation for SDR Scanner
"""

import yaml


def load_config(config_path: str) -> dict:
	"""
	Load configuration from YAML file

	Args:
		config_path: Path to the YAML configuration file

	Returns:
		Dictionary containing configuration data

	Raises:
		FileNotFoundError: If config file doesn't exist
		yaml.YAMLError: If config file is invalid YAML
	"""
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)


def validate_config(config: dict) -> None:
	"""
	Validate configuration structure and required fields

	Args:
		config: Configuration dictionary

	Raises:
		ValueError: If configuration is invalid
	"""
	# Check for required top-level keys
	required_keys = ['bands', 'scanner']
	for key in required_keys:
		if key not in config:
			raise ValueError(f"Missing required configuration key: {key}")

	# Validate bands configuration
	if not isinstance(config['bands'], dict):
		raise ValueError("'bands' must be a dictionary")

	# Validate scanner configuration
	if not isinstance(config['scanner'], dict):
		raise ValueError("'scanner' must be a dictionary")


def get_band_config(config: dict, band_name: str) -> dict:
	"""
	Get configuration for a specific band

	Args:
		config: Full configuration dictionary
		band_name: Name of the band to retrieve

	Returns:
		Dictionary containing band configuration

	Raises:
		KeyError: If band doesn't exist in configuration
	"""
	if band_name not in config['bands']:
		available_bands = ', '.join(config['bands'].keys())
		raise KeyError(f"Band '{band_name}' not found in configuration. Available bands: {available_bands}")

	return config['bands'][band_name]

