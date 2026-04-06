"""Tests for the device factory and mock protocol."""

import sys
import types
import unittest.mock

import pytest

import sdr_scanner.devices


def _mock_create_device (alias, device_type, mock_class_name):
	"""Helper: mock the submodule and call create_device."""
	mock_device = unittest.mock.MagicMock()
	mock_cls = unittest.mock.MagicMock(return_value=mock_device)
	fake_module = types.ModuleType(f"sdr_scanner.devices.{device_type}")
	setattr(fake_module, mock_class_name, mock_cls)

	patches = {
		f"sdr_scanner.devices.{device_type}": fake_module,
	}
	with unittest.mock.patch.dict(sys.modules, patches):
		# Also set the attribute on the parent module so `sdr_scanner.devices.rtlsdr` resolves
		setattr(sdr_scanner.devices, device_type, fake_module)
		try:
			device = sdr_scanner.devices.create_device(alias, device_index=0)
		finally:
			delattr(sdr_scanner.devices, device_type)
	return mock_cls


class TestCreateDevice:

	def test_unknown_type_raises (self):
		with pytest.raises(ValueError, match="Unsupported"):
			sdr_scanner.devices.create_device("unknown_sdr")

	def test_rtlsdr_aliases (self):
		for alias in ("rtl", "rtlsdr", "rtl-sdr"):
			mock_cls = _mock_create_device(alias, "rtlsdr", "RtlSdrDevice")
			mock_cls.assert_called_once_with(0)

	def test_hackrf_aliases (self):
		for alias in ("hackrf", "hackrf-one", "hackrfone"):
			mock_cls = _mock_create_device(alias, "hackrf", "HackRfDevice")
			mock_cls.assert_called_once_with(0)

	def test_case_insensitive (self):
		mock_cls = _mock_create_device("RTLSDR", "rtlsdr", "RtlSdrDevice")
		mock_cls.assert_called_once()
