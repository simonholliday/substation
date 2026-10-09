"""Tests for CLI argument parsing and entry points."""

import argparse
import sys
import unittest.mock

import pytest
import yaml

import substation.cli
import substation.config
import substation.device_families


class TestListBands:

	def test_list_bands_prints_names (self, tmp_path, minimal_config_dict, capsys):
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		substation.cli.list_bands(cfg_path)
		captured = capsys.readouterr()
		assert "test_nfm" in captured.out


class TestInitConfig:

	def test_init_creates_config (self, tmp_path, monkeypatch):
		"""--init writes a config.yaml that loads and contains the shipped bands."""
		monkeypatch.chdir(tmp_path)

		with unittest.mock.patch("sys.argv", ["substation", "--init"]):
			rc = substation.cli.main()
		assert rc == 0

		written = tmp_path / "config.yaml"
		assert written.exists()

		# The scaffolded file must be a valid, loadable config with bands.
		config = substation.config.load_config(written)
		assert "pmr" in config.bands

	def test_init_refuses_to_overwrite (self, tmp_path, monkeypatch):
		"""--init must not clobber an existing config.yaml."""
		monkeypatch.chdir(tmp_path)
		existing = tmp_path / "config.yaml"
		existing.write_text("scanner: {sdr_device_sample_size: 1, band_time_slice_ms: 1}\n")

		with pytest.raises(SystemExit) as exc_info:
			with unittest.mock.patch("sys.argv", ["substation", "--init"]):
				substation.cli.main()
		assert exc_info.value.code == 1

		# The original file is untouched.
		assert "sdr_device_sample_size: 1" in existing.read_text()

	def test_init_refuses_in_source_checkout (self, tmp_path, monkeypatch):
		"""--init must refuse to run where a substation/ package dir exists."""
		monkeypatch.chdir(tmp_path)
		(tmp_path / "substation").mkdir()
		(tmp_path / "substation" / "__init__.py").write_text("")

		with pytest.raises(SystemExit) as exc_info:
			with unittest.mock.patch("sys.argv", ["substation", "--init"]):
				substation.cli.main()
		assert exc_info.value.code == 1
		assert not (tmp_path / "config.yaml").exists()


class TestMainArgParsing:

	def test_list_bands_flag (self, tmp_path, minimal_config_dict, capsys):
		"""--list-bands should list bands and return exit status 0."""
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		with unittest.mock.patch("sys.argv", ["substation", "--list-bands", "-c", str(cfg_path)]):
			assert substation.cli.main() == 0
		captured = capsys.readouterr()
		assert "test_nfm" in captured.out

	def test_missing_band_exits_error (self, tmp_path, minimal_config_dict):
		"""Requesting a non-existent band should exit with error."""
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		with pytest.raises(SystemExit) as exc_info:
			with unittest.mock.patch("sys.argv", [
				"substation", "-b", "nonexistent_band", "-c", str(cfg_path)
			]):
				substation.cli.main()
		assert exc_info.value.code != 0


class TestLogLevel:

	def test_log_level_option_sets_the_level (self, tmp_path, minimal_config_dict, monkeypatch):
		"""The README told users to enable debug logging, which --log-level DEBUG now does."""
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		levels = []
		monkeypatch.setattr(substation.cli.logging, "basicConfig", lambda **kwargs: levels.append(kwargs["level"]))

		with unittest.mock.patch("sys.argv", ["substation", "--list-bands", "-c", str(cfg_path), "--log-level", "debug"]):
			assert substation.cli.main() == 0

		assert levels == [substation.cli.logging.DEBUG]


class TestParser:

	def test_building_the_parser_parses_nothing (self, monkeypatch):
		"""subsystem.co builds the parser to generate the command-line reference, so building it must not read the command line (#4717)."""
		monkeypatch.setattr(sys, "argv", ["substation", "--no-such-option"])
		command = substation.cli.parser()

		assert isinstance(command, argparse.ArgumentParser)
		assert command.prog == "substation"

	def test_the_help_says_each_exit_status (self):
		"""The exit status is said in the help, which the command-line reference prints."""
		text = substation.cli.parser().format_help()

		assert "Exit status:" in text
		for status in ("  0  The scan ended", "  1  A scan stopped because of an error", "  2  The command line was not understood"):
			assert status in text

	def test_the_device_type_help_names_every_spelling_it_accepts (self):
		"""The README's device cards listed the other spellings, and the help, which the command-line reference prints, did not."""
		action = next(action for action in substation.cli.parser()._actions if "--device-type" in action.option_strings)
		named = {word.strip(".,()") for word in action.help.split()}

		for spelling in substation.device_families.DEVICE_FAMILY_ALIASES:
			if spelling != "file":
				assert spelling in named, spelling

	def test_an_unknown_option_exits_2 (self, capsys):
		"""The help says an option it does not recognise exits 2."""
		with unittest.mock.patch("sys.argv", ["substation", "--no-such-option"]):
			with pytest.raises(SystemExit) as exc_info:
				substation.cli.main()

		assert exc_info.value.code == 2
