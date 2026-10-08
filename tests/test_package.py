"""Tests for the Python interface the package declares in __all__, which subsystem.co documents (#4718)."""

import subprocess
import sys

import pytest

import substation
import substation.config
import substation.osc_sender
import substation.scanner


class TestPythonInterface:

	def test_all_names_the_documented_interface (self):
		"""__all__ is the surface the README documents and subsystem.co's Python reference prints."""
		assert substation.__all__ == ["load_config", "RadioScanner", "OscEventSender"]

	@pytest.mark.parametrize(("name", "module"), [
		("load_config", substation.config),
		("RadioScanner", substation.scanner),
		("OscEventSender", substation.osc_sender),
	])
	def test_each_name_is_the_object_its_module_defines (self, name, module):
		"""substation.RadioScanner and the rest are the objects themselves, imported when first used."""
		assert getattr(substation, name) is getattr(module, name)

	def test_a_name_the_package_does_not_declare_is_an_attribute_error (self):
		with pytest.raises(AttributeError):
			getattr(substation, "no_such_name")

	def test_importing_the_configuration_loads_neither_numpy_nor_python_osc (self):
		"""subsystem.co imports substation.config bare for the configuration reference, so declaring __all__ must not make that import heavy."""
		result = subprocess.run(
			[sys.executable, "-c", "import sys, substation.config; print('numpy' in sys.modules, 'pythonosc' in sys.modules)"],
			capture_output=True,
			text=True,
		)

		assert result.returncode == 0, result.stderr
		assert result.stdout.split() == ["False", "False"]
