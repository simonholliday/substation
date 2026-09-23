"""Tests for configuration loading and validation."""

import fractions
import logging
import pathlib

import pydantic
import pytest
import yaml

import substation.config
import substation.constants


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYamlLoading:

	def test_load_raw_config (self, tmp_path, minimal_config_dict):
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		data = substation.config._load_raw_config(cfg_path)
		assert "bands" in data
		assert "test_nfm" in data["bands"]

	def test_load_missing_file (self):
		with pytest.raises(FileNotFoundError):
			substation.config.load_config(pathlib.Path("/nonexistent/config.yaml"))

	def test_fraction_tag (self, tmp_path):
		content = "value: !fraction 25000/3\n"
		path = tmp_path / "frac.yaml"
		path.write_text(content)
		data = yaml.load(path.read_text(), Loader=substation.config._YamlLoader)
		assert data["value"] == fractions.Fraction(25000, 3)

	def test_empty_yaml_raises (self, tmp_path):
		path = tmp_path / "empty.yaml"
		path.write_text("")
		with pytest.raises(ValueError):
			substation.config._load_raw_config(path)

	def test_load_config_defaults_only (self, tmp_path, monkeypatch):
		"""load_config() with no user config loads config.yaml.default.

		Runs from an empty directory, because load_config() also reads a
		config.yaml in the working directory, and a developer's own one would
		otherwise decide whether this test passes.
		"""
		monkeypatch.chdir(tmp_path)
		config = substation.config.load_config()
		assert len(config.bands) > 0

	def test_load_config_with_user_override (self, tmp_path, monkeypatch):
		"""User config overrides specific values from defaults."""
		user_cfg = tmp_path / "config.yaml"
		user_cfg.write_text(yaml.dump({
			"recording": {"audio_output_dir": "/tmp/test_override"},
		}))
		config = substation.config.load_config(user_cfg)
		assert config.recording.audio_output_dir == "/tmp/test_override"
		# Other recording defaults should be preserved
		assert config.recording.audio_sample_rate == 16000

	def test_load_config_accepts_str_path (self, tmp_path):
		"""load_config is the public module entry point and must accept a plain str path.

		Regression: the examples passed './config.yaml' as a str and crashed
		with AttributeError before str coercion was added.
		"""
		user_cfg = tmp_path / "config.yaml"
		user_cfg.write_text(yaml.dump({
			"recording": {"audio_output_dir": "/tmp/str_path_override"},
		}))
		config = substation.config.load_config(str(user_cfg))
		assert config.recording.audio_output_dir == "/tmp/str_path_override"

	def test_removed_supervisor_section_is_ignored_with_a_warning (self, tmp_path, caplog):
		"""Regression: a config.yaml written by --init before Supervisor was removed stopped loading.

		`substation --init` copied the shipped `supervisor` section into every
		user's file, so rejecting it broke everyone who followed the Quick
		Start.  The section is now dropped, the rest of the file still applies,
		and the warning names the section and the file.
		"""
		user_cfg = tmp_path / "config.yaml"
		user_cfg.write_text(yaml.dump({
			"supervisor": {"enabled": False, "port": 9004},
			"recording": {"audio_output_dir": "/tmp/kept_override"},
		}))

		config = substation.config.load_config(user_cfg)

		assert config.recording.audio_output_dir == "/tmp/kept_override"
		assert "'supervisor' section" in caplog.text
		assert str(user_cfg) in caplog.text

	def test_removed_section_is_ignored_in_a_dict_config (self, minimal_config_dict, caplog):
		"""Programs passing a dict get the same treatment as config.yaml."""
		minimal_config_dict["supervisor"] = {"enabled": True}

		config = substation.config.validate_config(minimal_config_dict)

		assert "supervisor" not in config.model_dump()
		assert "'supervisor' section" in caplog.text
		assert "supervisor" in minimal_config_dict

	def test_misspelt_section_is_still_rejected (self, tmp_path):
		"""Only sections that really were removed are ignored; a typo still fails and names the key."""
		user_cfg = tmp_path / "config.yaml"
		user_cfg.write_text(yaml.dump({"recordng": {"audio_output_dir": "/tmp/x"}}))

		with pytest.raises(pydantic.ValidationError, match="recordng"):
			substation.config.load_config(user_cfg)


# ---------------------------------------------------------------------------
# Schema descriptions
# ---------------------------------------------------------------------------

class TestSchemaDescriptions:

	def test_every_setting_has_a_description (self):
		"""Every setting in the published JSON schema carries a description.

		The configuration reference is generated from this schema, so a setting
		without a description would appear there with no explanation.  Each
		field's description is the docstring written directly under it.
		"""
		schema = substation.config.AppConfig.model_json_schema()

		models = [("AppConfig", schema)] + list(schema.get("$defs", {}).items())
		undescribed = [
			f"{model_name}.{setting}"
			for model_name, model in models
			for setting, details in model.get("properties", {}).items()
			if not details.get("description")
		]

		assert undescribed == [], f"Settings with no docstring under their field: {undescribed}"


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

class TestDeepMerge:

	def test_scalar_override (self):
		base = {"a": 1, "b": 2}
		override = {"a": 10}
		result = substation.config._deep_merge(base, override)
		assert result == {"a": 10, "b": 2}

	def test_nested_dict_merge (self):
		base = {"section": {"x": 1, "y": 2}}
		override = {"section": {"x": 10}}
		result = substation.config._deep_merge(base, override)
		assert result == {"section": {"x": 10, "y": 2}}

	def test_new_key_added (self):
		base = {"a": 1}
		override = {"b": 2}
		result = substation.config._deep_merge(base, override)
		assert result == {"a": 1, "b": 2}

	def test_none_override_preserves_base_dict (self):
		"""YAML section with all children commented out parses as None."""
		base = {"section": {"x": 1, "y": 2}}
		override = {"section": None}
		result = substation.config._deep_merge(base, override)
		assert result == {"section": {"x": 1, "y": 2}}

	def test_inputs_not_mutated (self):
		base = {"section": {"x": 1}}
		override = {"section": {"x": 10}}
		substation.config._deep_merge(base, override)
		assert base == {"section": {"x": 1}}
		assert override == {"section": {"x": 10}}


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------

class TestBandValidation:

	def test_valid_band (self, minimal_config_dict):
		config = substation.config.validate_config(minimal_config_dict)
		band = config.bands["test_nfm"]
		assert band.freq_start < band.freq_end

	def test_freq_start_ge_freq_end_raises (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["freq_start"] = 500e6
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 400e6
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_snr_threshold_below_hysteresis_warns (self, minimal_config_dict, caplog):
		minimal_config_dict["bands"]["test_nfm"]["snr_threshold_db"] = 2.0
		with caplog.at_level(logging.WARNING):
			config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].snr_threshold_db == 2.0
		assert "OFF threshold" in caplog.text

	def test_sample_rate_below_band_span_accepted (self, minimal_config_dict):
		# Band span vs sample_rate is checked at scanner init, not config validation.
		# Config should accept bands wider than their sample_rate.
		minimal_config_dict["bands"]["test_nfm"]["sample_rate"] = 50000.0
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sample_rate == 50000.0

	def test_channel_width_defaults (self, app_config):
		band = app_config.bands["test_nfm"]
		expected = band.channel_spacing * substation.constants.CHANNEL_WIDTH_FRACTION
		assert band.channel_width == pytest.approx(expected)

	def test_modulation_uppercase (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["modulation"] = "nfm"
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].modulation == "NFM"

	def test_gain_auto (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["sdr_gain_db"] = "auto"
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sdr_gain_db == "auto"

	def test_gain_numeric (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["sdr_gain_db"] = 42
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sdr_gain_db == 42.0

	def test_extra_field_rejected (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["bogus_field"] = True
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)


class TestRecordingValidation:

	def test_buffer_size_zero_raises (self, minimal_config_dict):
		minimal_config_dict["recording"] = {"buffer_size_seconds": 0}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_invalid_audio_format_raises (self, minimal_config_dict):
		minimal_config_dict["recording"] = {"audio_format": "mp3"}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_valid_audio_formats (self, minimal_config_dict):
		for fmt in ("wav", "flac"):
			minimal_config_dict["recording"] = {"audio_format": fmt}
			config = substation.config.validate_config(minimal_config_dict)
			assert config.recording.audio_format == fmt


# ---------------------------------------------------------------------------
# Band defaults inheritance
# ---------------------------------------------------------------------------

class TestBandDefaults:

	def test_type_inherits_defaults (self, minimal_config_dict):
		minimal_config_dict["band_defaults"] = {
			"TEST_TYPE": {"snr_threshold_db": 8.0, "sdr_gain_db": 25}
		}
		minimal_config_dict["bands"]["test_nfm"]["type"] = "TEST_TYPE"
		# Band's own snr_threshold_db (12) should override the default (8)
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].snr_threshold_db == 12.0
		# sdr_gain_db from default (25) should be used if not set in band
		# (but our band has it set to 30, so it stays 30)
		assert config.bands["test_nfm"].sdr_gain_db == 30

	def test_band_inherits_what_it_does_not_set (self, minimal_config_dict):
		"""A band takes each setting it leaves out from its template, and keeps the ones it gives."""
		minimal_config_dict["band_defaults"] = {
			"TEST_TYPE": {"snr_threshold_db": 8.0, "sdr_gain_db": 25}
		}
		band = minimal_config_dict["bands"]["test_nfm"]
		band["type"] = "TEST_TYPE"
		del band["snr_threshold_db"]

		config = substation.config.validate_config(minimal_config_dict)

		assert config.bands["test_nfm"].snr_threshold_db == 8.0
		assert config.bands["test_nfm"].sdr_gain_db == 30

	def test_null_in_a_template_is_not_passed_on (self, minimal_config_dict):
		"""Regression: a template setting written as null, as the generated reference shows it, broke every band of that type.

		The raw merge copied the null into each band, and a band's
		snr_threshold_db must be a number.  Null in a template now means the
		template does not set it, so the band's own default applies.
		"""
		minimal_config_dict["band_defaults"] = {
			"TEST_TYPE": {"snr_threshold_db": None, "channel_width": None, "modulation": None, "sample_rate": None}
		}
		band = minimal_config_dict["bands"]["test_nfm"]
		band["type"] = "TEST_TYPE"
		del band["snr_threshold_db"]

		config = substation.config.validate_config(minimal_config_dict)

		default_threshold = substation.config.BandConfig.model_fields["snr_threshold_db"].default
		assert config.bands["test_nfm"].snr_threshold_db == default_threshold
		assert config.bands["test_nfm"].modulation == "NFM"
		assert config.bands["test_nfm"].sample_rate == 1.024e6

	def test_template_name_in_another_case_adjusts_the_shipped_template (self, tmp_path, monkeypatch):
		"""Regression: a user's `air` template sat beside the shipped `AIR` and replaced it whole.

		The shipped AIR bands then lost the settings the user did not repeat:
		air_civil_bristol stopped recording and dropped to automatic gain.
		"""
		monkeypatch.chdir(tmp_path)
		shipped = substation.config.load_config().bands["air_civil_bristol"]

		user_cfg = tmp_path / "config.yaml"
		user_cfg.write_text(yaml.dump({"band_defaults": {"air": {"snr_threshold_db": 12}}}))
		adjusted = substation.config.load_config(user_cfg).bands["air_civil_bristol"]

		assert shipped.recording_enabled
		assert adjusted.recording_enabled
		assert adjusted.sdr_gain_db == shipped.sdr_gain_db

	def test_unknown_type_warns (self, minimal_config_dict, caplog):
		minimal_config_dict["band_defaults"] = {"KNOWN_TYPE": {"sdr_gain_db": 10}}
		minimal_config_dict["bands"]["test_nfm"]["type"] = "UNKNOWN_TYPE"
		with caplog.at_level(logging.WARNING):
			substation.config.validate_config(minimal_config_dict)
		assert "UNKNOWN_TYPE" in caplog.text


class TestDeviceOverrideKeys:

	@pytest.mark.parametrize("key", ["AirSpy", "airspy-r2", " AIRSPYR2 "])
	def test_any_spelling_of_a_family_is_its_key (self, minimal_config_dict, key):
		"""Regression: an override keyed by another spelling of a device type was silently never applied."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {key: {"sample_rate": 2.5e6}}

		config = substation.config.validate_config(minimal_config_dict)

		assert config.bands["test_nfm"].device_overrides == {"airspy": substation.config.DeviceOverrideConfig(sample_rate=2.5e6)}

	def test_unknown_key_is_warned_about (self, minimal_config_dict, caplog):
		"""A key that names no device type is kept, as it may be a SoapySDR driver, but a typo gets a warning."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {"airpsy": {"sample_rate": 2.5e6}}

		with caplog.at_level(logging.WARNING):
			config = substation.config.validate_config(minimal_config_dict)

		assert "airpsy" in config.bands["test_nfm"].device_overrides
		assert "'airpsy' names no device type" in caplog.text

	def test_soapy_prefix_marks_a_driver_on_purpose (self, minimal_config_dict, caplog):
		"""Written as soapy:<driver>, a SoapySDR driver's key is accepted quietly, under the driver's name."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {"soapy:lime": {"sample_rate": 2.5e6}}

		with caplog.at_level(logging.WARNING):
			config = substation.config.validate_config(minimal_config_dict)

		assert list(config.bands["test_nfm"].device_overrides) == ["lime"]
		assert "names no device type" not in caplog.text

	def test_two_spellings_of_one_family_are_merged (self, minimal_config_dict):
		"""Keys for the same family combine, the later one winning where both set a value."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"rtlsdr": {"sample_rate": 2.4e6, "snr_threshold_db": 9.0},
			"RTL-SDR": {"snr_threshold_db": 7.0},
		}

		override = substation.config.validate_config(minimal_config_dict).bands["test_nfm"].device_overrides["rtlsdr"]

		assert (override.sample_rate, override.snr_threshold_db) == (2.4e6, 7.0)


class TestRecordingNeedsADemodulator:

	def test_recording_with_an_unknown_modulation_warns (self, minimal_config_dict, caplog):
		"""Regression: a modulation with no demodulator, such as a typo for NFM, silently turned recording off."""
		minimal_config_dict["bands"]["test_nfm"]["modulation"] = "NMF"

		with caplog.at_level(logging.WARNING):
			substation.config.validate_config(minimal_config_dict)

		assert "'NMF' has no demodulator" in caplog.text

	def test_detection_only_band_with_any_label_is_quiet (self, minimal_config_dict, caplog):
		"""A band that does not record may carry any label, such as TETRA."""
		band = minimal_config_dict["bands"]["test_nfm"]
		band["modulation"] = "TETRA"
		band["recording_enabled"] = False

		with caplog.at_level(logging.WARNING):
			substation.config.validate_config(minimal_config_dict)

		assert "no demodulator" not in caplog.text

	def test_shipped_configuration_warns_about_none (self, tmp_path, monkeypatch, caplog):
		"""Every shipped band that records has a modulation the scanner can demodulate."""
		monkeypatch.chdir(tmp_path)

		with caplog.at_level(logging.WARNING):
			substation.config.load_config()

		assert "no demodulator" not in caplog.text

	def test_the_list_matches_the_demodulators (self):
		"""The configuration's list of recordable modulations is the DSP code's own."""
		import substation.dsp.demodulation

		assert set(substation.constants.DEMODULATED_MODULATIONS) == set(substation.dsp.demodulation.DEMODULATORS)


class TestAudioSampleRate:

	def test_rate_too_low_for_the_voice_band_is_rejected (self, minimal_config_dict):
		"""Regression: 6 kHz loaded, then the NFM voice filter could not be built and the scan ended at the first activation."""
		minimal_config_dict["recording"]["audio_sample_rate"] = 6000

		with pytest.raises(pydantic.ValidationError, match="audio_sample_rate"):
			substation.config.validate_config(minimal_config_dict)

	def test_telephone_rate_is_accepted (self, minimal_config_dict):
		"""8 kHz holds the whole voice band."""
		minimal_config_dict["recording"]["audio_sample_rate"] = 8000

		assert substation.config.validate_config(minimal_config_dict).recording.audio_sample_rate == 8000


class TestRequiredBandwidth:

	def test_span_plus_one_channel_and_edge_margins (self, app_config):
		"""The band's span, one radio channel's width, and half a spacing at each edge."""
		band = app_config.bands["test_nfm"]

		assert band.required_bandwidth == pytest.approx((446.09375e6 - 446.00625e6) + 0.84 * 12500 + 12500)


class TestGainElements:

	def test_gain_elements_accepted (self, minimal_config_dict):
		"""Per-element gain dict should be accepted by config validation."""
		minimal_config_dict["bands"]["test_nfm"]["sdr_gain_elements"] = {"LNA": 10, "VGA": 12}
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sdr_gain_elements == {"LNA": 10.0, "VGA": 12.0}

	def test_gain_elements_none_by_default (self, app_config):
		"""sdr_gain_elements should be None when not specified."""
		assert app_config.bands["test_nfm"].sdr_gain_elements is None



class TestDeviceSettings:

	def test_device_settings_accepted (self, minimal_config_dict):
		"""Device settings dict should be accepted by config validation."""
		minimal_config_dict["bands"]["test_nfm"]["sdr_device_settings"] = {"biastee": "true"}
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sdr_device_settings == {"biastee": "true"}

	def test_device_settings_none_by_default (self, app_config):
		"""sdr_device_settings should be None when not specified."""
		assert app_config.bands["test_nfm"].sdr_device_settings is None


class TestActivationVariance:

	def test_activation_variance_db_accepted (self, minimal_config_dict):
		"""activation_variance_db should be accepted by config validation."""
		minimal_config_dict["bands"]["test_nfm"]["activation_variance_db"] = 3.0
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].activation_variance_db == 3.0

	def test_activation_variance_db_zero_accepted (self, minimal_config_dict):
		"""Setting to 0 should be allowed (disables the check)."""
		minimal_config_dict["bands"]["test_nfm"]["activation_variance_db"] = 0
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].activation_variance_db == 0.0

	def test_activation_variance_db_none_by_default (self, app_config):
		"""activation_variance_db should be None when not specified."""
		assert app_config.bands["test_nfm"].activation_variance_db is None

	def test_activation_variance_db_negative_rejected (self, minimal_config_dict):
		"""Negative values should fail validation (ge=0 constraint)."""
		minimal_config_dict["bands"]["test_nfm"]["activation_variance_db"] = -1.0
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)


class TestDynamicsCurveConfig:

	"""Validation tests for the experimental dynamics_curve recording stage."""

	def test_disabled_by_default (self, app_config):
		"""dynamics_curve_enabled should default to False so existing behaviour is unchanged."""
		assert app_config.recording.dynamics_curve_enabled is False

	def test_default_parameters_are_sane (self, app_config):
		"""The default DynamicsCurveConfig should have sensible values."""
		curve = app_config.recording.dynamics_curve
		assert curve.threshold_dbfs == -25.0
		assert curve.cut_db == 6.0
		assert curve.boost_db == 1.5
		assert curve.floor_dbfs == -60.0
		assert curve.cut_curve == 0.5
		assert curve.boost_curve == 0.5

	def test_custom_parameters_accepted (self, minimal_config_dict):
		"""All six parameters should be accepted when set explicitly."""
		minimal_config_dict["recording"] = {
			"dynamics_curve_enabled": True,
			"dynamics_curve": {
				"threshold_dbfs": -30.0,
				"cut_db": 8.0,
				"boost_db": 2.0,
				"floor_dbfs": -55.0,
				"cut_curve": 0.3,
				"boost_curve": 0.7,
			},
		}
		config = substation.config.validate_config(minimal_config_dict)
		assert config.recording.dynamics_curve_enabled is True
		assert config.recording.dynamics_curve.threshold_dbfs == -30.0
		assert config.recording.dynamics_curve.cut_db == 8.0
		assert config.recording.dynamics_curve.boost_curve == 0.7

	def test_floor_above_threshold_rejected (self, minimal_config_dict):
		"""floor_dbfs >= threshold_dbfs should fail validation."""
		minimal_config_dict["recording"] = {
			"dynamics_curve": {
				"threshold_dbfs": -30.0,
				"floor_dbfs": -20.0,
			},
		}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_threshold_at_zero_rejected (self, minimal_config_dict):
		"""threshold_dbfs must be strictly less than 0."""
		minimal_config_dict["recording"] = {
			"dynamics_curve": {
				"threshold_dbfs": 0.0,
			},
		}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_negative_cut_db_rejected (self, minimal_config_dict):
		"""cut_db must be non-negative."""
		minimal_config_dict["recording"] = {
			"dynamics_curve": {
				"cut_db": -1.0,
			},
		}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_curve_out_of_range_rejected (self, minimal_config_dict):
		"""cut_curve and boost_curve must be in [0, 1]."""
		minimal_config_dict["recording"] = {
			"dynamics_curve": {
				"cut_curve": 1.5,
			},
		}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_boost_clipping_warning (self, minimal_config_dict, caplog):
		"""A boost configuration that would push the boost peak above 0 dBFS should log a warning (but still validate)."""
		minimal_config_dict["recording"] = {
			"dynamics_curve_enabled": True,
			"dynamics_curve": {
				"threshold_dbfs": -3.0,
				"boost_db": 5.0,
			},
		}
		with caplog.at_level(logging.WARNING, logger="substation.config"):
			config = substation.config.validate_config(minimal_config_dict)
		assert config.recording.dynamics_curve_enabled is True
		assert any("dynamics_curve" in record.message for record in caplog.records)


class TestExcludeChannelIndices:

	def test_valid_indices (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [1, 3]
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].exclude_channel_indices == [1, 3]

	def test_zero_index_raises (self, minimal_config_dict):
		"""Channel numbers are 1-based (matching logs and filenames), so 0 is invalid."""
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [0]
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_negative_index_raises (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [-1]
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_the_shipped_pmr_band_skips_no_radio_channels (self, tmp_path, monkeypatch):
		"""Regression: a local exclusion of radio channels 1 to 3, among them the busiest, reached every user of the shipped pmr band."""
		monkeypatch.chdir(tmp_path)

		assert substation.config.load_config().bands["pmr"].exclude_channel_indices == []


class TestDeviceOverrides:

	def test_device_overrides_accepted (self, minimal_config_dict):
		"""device_overrides dict should be accepted by config validation."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6, "sdr_gain_elements": {"LNA": 14}},
		}
		config = substation.config.validate_config(minimal_config_dict)
		overrides = config.bands["test_nfm"].device_overrides
		assert overrides is not None
		assert "airspy" in overrides
		assert overrides["airspy"].sample_rate == 2.5e6

	def test_device_overrides_none_by_default (self, app_config):
		"""device_overrides should be None when not specified."""
		assert app_config.bands["test_nfm"].device_overrides is None

	def test_device_overrides_extra_field_rejected (self, minimal_config_dict):
		"""Typos in override fields are caught by extra='forbid'."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sampl_rate": 2.5e6},
		}
		with pytest.raises(pydantic.ValidationError):
			substation.config.validate_config(minimal_config_dict)

	def test_device_overrides_gain_auto (self, minimal_config_dict):
		"""'auto' string should be normalized in device overrides."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspyhf": {"sdr_gain_db": "Auto"},
		}
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].device_overrides["airspyhf"].sdr_gain_db == "auto"

	def test_device_overrides_gain_none_preserved (self, minimal_config_dict):
		"""None gain in override means 'not overridden', not 'auto'."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6},
		}
		config = substation.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].device_overrides["airspy"].sdr_gain_db is None

	def test_device_overrides_multiple_devices (self, minimal_config_dict):
		"""Multiple device overrides on the same band."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6},
			"airspyhf": {"sample_rate": 0.912e6, "snr_threshold_db": 6},
		}
		config = substation.config.validate_config(minimal_config_dict)
		overrides = config.bands["test_nfm"].device_overrides
		assert overrides["airspy"].sample_rate == 2.5e6
		assert overrides["airspyhf"].sample_rate == 0.912e6
		assert overrides["airspyhf"].snr_threshold_db == 6
