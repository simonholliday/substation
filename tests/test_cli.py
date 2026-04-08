"""Tests for CLI argument parsing and entry points."""

import sys
import unittest.mock

import pytest
import yaml

import substation.cli


class TestListBands:

	def test_list_bands_prints_names (self, tmp_path, minimal_config_dict, capsys):
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		substation.cli.list_bands(str(cfg_path))
		captured = capsys.readouterr()
		assert "test_nfm" in captured.out


class TestMainArgParsing:

	def test_list_bands_flag (self, tmp_path, minimal_config_dict, capsys):
		"""--list-bands should list bands and exit cleanly (code 0 or None)."""
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		try:
			with unittest.mock.patch("sys.argv", ["substation", "--list-bands", "-c", str(cfg_path)]):
				substation.cli.main()
		except SystemExit as exc:
			assert exc.code in (0, None)
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
