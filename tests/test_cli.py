"""Tests for CLI argument parsing and entry points."""

import sys
import unittest.mock

import pytest
import yaml

import substation.cli
import substation.config


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
