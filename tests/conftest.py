"""Shared fixtures for Substation tests."""

import typing

import pytest

import substation.config
import substation.scanner


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_band_dict () -> dict:

	"""Minimal valid band config dict for test_nfm."""

	return {
		"freq_start": 446.00625e6,
		"freq_end": 446.09375e6,
		"channel_spacing": 12500.0,
		"sample_rate": 1.024e6,
		"snr_threshold_db": 12.0,
		"modulation": "NFM",
		"recording_enabled": True,
		"sdr_gain_db": 30,
	}


@pytest.fixture
def minimal_config_dict (minimal_band_dict: dict, tmp_path: typing.Any) -> dict:

	"""Full valid raw config dict."""

	return {
		"scanner": {
			"sdr_device_sample_size": 16384,
			"band_time_slice_ms": 100,
		},
		"recording": {
			"audio_sample_rate": 16000,
			"audio_output_dir": str(tmp_path / "audio"),
		},
		"bands": {
			"test_nfm": minimal_band_dict,
		},
	}


@pytest.fixture
def app_config (minimal_config_dict: dict) -> substation.config.AppConfig:

	"""Validated AppConfig instance."""

	return substation.config.validate_config(minimal_config_dict)


@pytest.fixture
def scanner_instance (app_config: substation.config.AppConfig) -> substation.scanner.RadioScanner:

	"""RadioScanner initialised with FFT params but no SDR device."""

	sc = substation.scanner.RadioScanner(
		config=app_config,
		band_name="test_nfm",
		device_type="rtlsdr",
	)
	sc._precompute_fft_params()
	return sc
