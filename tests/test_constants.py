"""Smoke tests for tuning constants."""

import sdr_scanner.constants


def test_hysteresis_positive ():
	assert sdr_scanner.constants.HYSTERESIS_DB > 0


def test_welch_segments_minimum ():
	assert sdr_scanner.constants.WELCH_SEGMENTS >= 2


def test_ema_alpha_range ():
	alpha = sdr_scanner.constants.NOISE_FLOOR_EMA_ALPHA
	assert 0.0 < alpha <= 1.0


def test_warmup_slices_positive ():
	assert sdr_scanner.constants.NOISE_FLOOR_WARMUP_SLICES >= 1


def test_trim_constants_positive ():
	assert sdr_scanner.constants.TRIM_AMPLITUDE_THRESHOLD > 0
	assert sdr_scanner.constants.TRIM_PRE_SAMPLES >= 0
	assert sdr_scanner.constants.TRIM_POST_SAMPLES >= 0


def test_nfm_constants_types ():
	assert isinstance(sdr_scanner.constants.NFM_DEEMPHASIS_TAU, float)
	assert isinstance(sdr_scanner.constants.NFM_DEVIATION_HZ, float)
	assert sdr_scanner.constants.NFM_DEVIATION_HZ > 0


def test_am_constants ():
	assert sdr_scanner.constants.AM_AGC_ATTACK_MS > 0
	assert sdr_scanner.constants.AM_AGC_RELEASE_MS > sdr_scanner.constants.AM_AGC_ATTACK_MS
	assert 0.0 < sdr_scanner.constants.AM_OUTPUT_GAIN <= 1.0
