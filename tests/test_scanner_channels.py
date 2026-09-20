"""Tests for channel frequency calculation, IQ extraction, and device overrides."""

import fractions

import numpy
import pytest

import substation.config
import substation.devices
import substation.dsp.demodulation
import substation.scanner

import iq_generators


class TestCalculateChannels:

	def test_pmr446_channels (self, scanner_instance):
		"""PMR446: 8 channels from 446.00625 to 446.09375 at 12.5 kHz spacing."""
		channels = scanner_instance.channels
		# Our test config has freq_start=446.00625e6, freq_end=446.09375e6, spacing=12500
		expected_n = 8
		assert len(channels) == expected_n
		# Check first and last
		assert channels[0] == pytest.approx(446.00625e6)
		assert channels[-1] == pytest.approx(446.09375e6)

	def test_two_channels (self, minimal_config_dict):
		"""A band spanning exactly one spacing should have 2 channels."""
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 446.00625e6 + 12500.0
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		assert len(sc.channels) == 2
		assert sc.channels[0] == pytest.approx(446.00625e6)
		assert sc.channels[1] == pytest.approx(446.00625e6 + 12500.0)

	def test_no_float_drift (self, minimal_config_dict):
		"""Even with many channels, the last frequency should match freq_end exactly."""
		# 100 channels
		minimal_config_dict["bands"]["test_nfm"]["freq_start"] = 100e6
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 101.2375e6
		minimal_config_dict["bands"]["test_nfm"]["channel_spacing"] = 12500.0
		minimal_config_dict["bands"]["test_nfm"]["sample_rate"] = 2.4e6
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		assert len(sc.channels) == 100
		assert sc.channels[-1] == pytest.approx(101.2375e6, rel=1e-10)

	def test_excluded_channels (self, minimal_config_dict):
		"""Excluded indices should be removed from the channel list."""
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [1, 3]
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		# Original 8 channels minus 2 excluded = 6
		assert len(sc.channels) == 6


class TestChannelExtraction:

	def test_extract_tone_on_channel (self, scanner_instance):
		"""A tone at a channel frequency should be present in the extracted IQ."""
		sc = scanner_instance
		sc._precompute_fft_params()
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 0.05, amplitude=0.5)
		noise = iq_generators.generate_noise_iq(sc.sample_rate, 0.05, power_db=-40)
		iq = (tone + noise)[:sc.samples_per_slice].astype(numpy.complex64)
		extracted = sc._extract_channel_iq(iq, ch_freq)
		# Extracted signal should have significant energy (the tone was shifted to baseband)
		power = numpy.mean(numpy.abs(extracted) ** 2)
		assert power > 0.01

	def test_phase_continuity (self, scanner_instance):
		"""Extracting the same channel across two blocks should have no phase jump.

		The blocks split at 12345 IQ samples, where the channel's oscillator
		has not completed a whole number of cycles, so an oscillator that
		restarted with each block would show a jump.  At a slice boundary it
		would not: every channel completes whole cycles there.
		"""
		sc = scanner_instance
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		split = 12345
		assert abs(offset_hz * split / sc.sample_rate - round(offset_hz * split / sc.sample_rate)) > 0.1

		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 2 * split / sc.sample_rate, amplitude=0.5)
		ext1 = sc._extract_channel_iq(tone[:split], ch_freq, sample_offset=0)
		ext2 = sc._extract_channel_iq(tone[split:], ch_freq, sample_offset=split)

		# A tone at the channel's centre comes out as a steady phasor.  The
		# channel filter carries its memory across the join, so compare
		# where each block has settled, at its end.
		phase_jump = numpy.abs(numpy.angle(ext2[-1]) - numpy.angle(ext1[-1]))
		phase_jump = min(phase_jump, 2 * numpy.pi - phase_jump)
		assert phase_jump < 0.05


class TestDcOffset:

	def test_center_shifted_when_channel_on_dc (self, minimal_config_dict):
		"""Center frequency shifts by half a channel when a channel falls on DC."""
		# 9 channels (odd) → midpoint at 100.05 MHz lands exactly on channel 4
		minimal_config_dict["bands"]["test_nfm"]["freq_start"] = 100e6
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 100e6 + 12500 * 8
		minimal_config_dict["bands"]["test_nfm"]["channel_spacing"] = 12500
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		midpoint = (100e6 + 100e6 + 12500 * 8) / 2  # 100050000
		# Center should have been shifted by half a channel spacing
		assert sc.center_freq == pytest.approx(midpoint + 6250)

	def test_center_unchanged_when_dc_clear (self, scanner_instance):
		"""Center frequency unchanged when DC falls between channels."""
		sc = scanner_instance
		midpoint = (sc.freq_start + sc.freq_end) / 2
		# PMR's midpoint (446.1 MHz) falls between channels — no shift
		assert sc.center_freq == pytest.approx(midpoint)


class TestNormalizeDeviceFamily:

	def test_rtlsdr_aliases (self):
		assert substation.devices.normalize_device_family("rtlsdr") == "rtlsdr"
		assert substation.devices.normalize_device_family("rtl") == "rtlsdr"
		assert substation.devices.normalize_device_family("RTL-SDR") == "rtlsdr"

	def test_hackrf_aliases (self):
		assert substation.devices.normalize_device_family("hackrf") == "hackrf"
		assert substation.devices.normalize_device_family("HackRF-One") == "hackrf"

	def test_airspy_aliases (self):
		assert substation.devices.normalize_device_family("airspy") == "airspy"
		assert substation.devices.normalize_device_family("airspy-r2") == "airspy"
		assert substation.devices.normalize_device_family("AirSpyR2") == "airspy"

	def test_airspyhf_aliases (self):
		assert substation.devices.normalize_device_family("airspyhf") == "airspyhf"
		assert substation.devices.normalize_device_family("airspy-hf") == "airspyhf"
		assert substation.devices.normalize_device_family("airspyhf+") == "airspyhf"

	def test_soapy_driver (self):
		assert substation.devices.normalize_device_family("soapy:lime") == "lime"
		assert substation.devices.normalize_device_family("soapy:airspy") == "airspy"

	def test_unknown_passthrough (self):
		assert substation.devices.normalize_device_family("bladerf") == "bladerf"


class TestDeviceOverrideApplied:

	def test_override_applied_on_matching_device (self, minimal_config_dict):
		"""Device override should replace base values when device matches."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6, "snr_threshold_db": 8.0},
		}
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="airspy",
		)
		assert sc.sample_rate == 2.5e6
		assert sc.snr_threshold_db == 8.0

	def test_override_not_applied_on_different_device (self, minimal_config_dict):
		"""Device override should not be applied when device doesn't match."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6},
		}
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr",
		)
		assert sc.sample_rate == 1.024e6

	def test_override_preserves_base_fields (self, minimal_config_dict):
		"""Fields not in the override should keep their base values."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6},
		}
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="airspy",
		)
		# snr_threshold_db should be unchanged from base config
		assert sc.snr_threshold_db == 12.0


class TestSsbExtraction:

	@staticmethod
	def _ssb_scanner (modulation):
		"""A scanner for a 7.1-7.2 MHz band at 5 kHz spacing, the HF template's."""
		config = substation.config.validate_config({
			"scanner": {"sdr_device_sample_size": 16384, "band_time_slice_ms": 100},
			"recording": {"audio_sample_rate": 16000, "audio_output_dir": "/tmp/unused"},
			"bands": {"hf": {"freq_start": 7.1e6, "freq_end": 7.2e6, "channel_spacing": 5000.0, "sample_rate": 256e3,
				"snr_threshold_db": 10.0, "modulation": modulation, "recording_enabled": True, "sdr_gain_db": 30}},
		})
		sc = substation.scanner.RadioScanner(config=config, band_name="hf", device_type="rtlsdr")
		sc._precompute_fft_params()
		return sc

	@pytest.mark.parametrize("modulation", ["USB", "LSB"])
	def test_top_of_the_voice_band_is_kept (self, modulation):
		"""Regression: SSB audio above about 2.1 kHz was cut, by 8 dB at 2.5 kHz, on every shipped HF band.

		The channel filter was centred on the dial frequency, so on a 5 kHz
		spacing it passed only 2.1 kHz of the sideband.  A 1 kHz and a
		2.5 kHz tone go through extraction and demodulation together, so the
		AGC treats them alike.
		"""
		sc = self._ssb_scanner(modulation)
		channel = sc.channels[5]
		sign = 1 if modulation == "USB" else -1
		n = sc.samples_per_slice
		state = None
		blocks = []

		for block in range(4):
			t = (numpy.arange(n) + block * n) / sc.sample_rate
			offset = channel - sc.center_freq
			iq = 0.05 * numpy.exp(2j * numpy.pi * (offset + sign * 1000) * t) + 0.05 * numpy.exp(2j * numpy.pi * (offset + sign * 2500) * t)
			extracted = sc._extract_channel_iq(iq.astype(numpy.complex64), channel)
			sc.sample_counter += n
			audio, state = substation.dsp.demodulation.DEMODULATORS[modulation](extracted, sc.sample_rate, 16000, state=state)
			blocks.append(audio)

		audio = numpy.concatenate(blocks[2:])
		spectrum = numpy.abs(numpy.fft.rfft(audio * numpy.hanning(len(audio))))
		freqs = numpy.fft.rfftfreq(len(audio), 1 / 16000)
		level = lambda hz: spectrum[numpy.argmin(numpy.abs(freqs - hz))]

		assert 20 * numpy.log10(level(2500) / level(1000)) > -1.0

	def test_other_modulations_are_extracted_around_the_dial (self):
		"""Only SSB moves the channel filter; NFM keeps it centred on the radio channel."""
		assert self._ssb_scanner("NFM").extraction_offset_hz == 0.0
