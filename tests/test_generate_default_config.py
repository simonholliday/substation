"""Tests for the generated config.yaml.default and the generator that writes it."""

import pytest
import yaml

import substation.config
import substation.generate_default_config


def _shipped_text () -> str:

	"""The committed config.yaml.default, as text."""

	return substation.generate_default_config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


class TestGeneratedFile:

	def test_committed_file_matches_generator (self):
		"""The committed config.yaml.default is exactly what the generator writes.

		The file is generated from the field docstrings and defaults in
		substation/config.py and from substation/band_library.yaml.  Editing the
		file by hand, or changing a setting without regenerating, fails here.
		"""
		assert _shipped_text() == substation.generate_default_config.render(), (
			"config.yaml.default is out of date with substation/config.py or "
			"substation/band_library.yaml. Regenerate it with: "
			f"{substation.generate_default_config.REGENERATE_COMMAND}"
		)

	def test_live_values_equal_field_defaults (self):
		"""Every live scanner and recording value in the shipped file is its field's declared default.

		The shipped file is merged under every user's config.yaml, so a live
		value that differed from the default would mean the schema says one
		thing and every run does another.
		"""
		raw = yaml.load(_shipped_text(), Loader=substation.config._YamlLoader)

		def check (section: dict, model: type) -> None:
			assert set(section) == set(model.model_fields), f"{model.__name__} settings missing from or extra in the file"

			for name, field in model.model_fields.items():
				nested = substation.generate_default_config._nested_model(field)

				if nested is not None:
					check(section[name], nested)
					continue

				assert section[name] == field.get_default(call_default_factory=True), f"{model.__name__}.{name}"

		check(raw["scanner"], substation.config.ScannerConfig)
		check(raw["recording"], substation.config.RecordingConfig)

	@pytest.mark.parametrize("model", [
		substation.config.BandTypeConfig,
		substation.config.BandConfig,
		substation.config.DeviceOverrideConfig,
	])
	def test_every_band_setting_is_listed (self, model):
		"""Every template, band, and device override setting appears in the file's commented reference."""
		text = _shipped_text()

		missing = [name for name in model.model_fields if f"\n#   {name}: " not in text]

		assert missing == []

	def test_shipped_file_loads (self):
		"""The shipped file is a valid configuration with the band library in it."""
		config = substation.config.validate_config(yaml.load(_shipped_text(), Loader=substation.config._YamlLoader))

		assert "pmr" in config.bands
		assert "PMR" in config.band_defaults


class TestYamlValue:

	@pytest.mark.parametrize("value, expected", [
		(None, "null"),
		(True, "true"),
		(131072, "131072"),
		(30.0, "30"),
		(-25.0, "-25"),
		(1.25, "1.25"),
		(93.7e6, "93.7e+6"),
		(2.5e6, "2.5e+6"),
		("auto", "auto"),
		("./audio", "./audio"),
		("true", '"true"'),
		([], "[]"),
		({"LNA": 10, "MIX": 5}, "{LNA: 10, MIX: 5}"),
	])
	def test_values_are_written_as_a_person_would_type_them (self, value, expected):
		"""Values are written compactly, and each reads back as the value it came from."""
		assert substation.generate_default_config.yaml_value(value) == expected

	def test_value_that_would_read_back_differently_is_refused (self, monkeypatch):
		"""A value the YAML loader would misread fails loudly instead of being shipped."""
		monkeypatch.setattr(substation.generate_default_config, "_yaml_text", lambda value: "yes")

		with pytest.raises(ValueError, match="read back"):
			substation.generate_default_config.yaml_value("yes")
