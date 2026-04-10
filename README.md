# Substation

## Overview
Substation is a high-performance tool for monitoring and recording radio activity using Software Defined Radio (SDR) hardware. It is designed to be used in two ways:

1.  **As a Command-Line Tool**: Quickly scan and record bands using simple terminal commands.
2.  **As a Python Module**: Integrate radio scanning, detection, and callbacks directly into your own Python applications.

By connecting a supported USB receiver (like an RTL-SDR or HackRF), you can scan wide ranges of the radio spectrum - such as Airband or Maritime frequencies - and automatically record transmissions as they occur. The software handles the technical signal processing and hardware management in the background, allowing for efficient 24/7 monitoring even on modest hardware like a Raspberry Pi.

## Hardware Requirements
To use this software, a compatible Software Defined Radio (SDR) USB device is required. The code has been specifically tested and verified with the following hardware:

*   **[RTL-SDR Blog V4 and V3](https://www.rtl-sdr.com/about-rtl-sdr/)**: High-quality, low-cost receivers. 24 MHz - 1.7 GHz, up to 2.4 MHz sample rate.
*   **[HackRF One](https://greatscottgadgets.com/hackrf/one/)**: A wideband transceiver capable of monitoring much larger frequency spans. 1 MHz - 6 GHz, 2-20 MHz sample rate.
*   **[AirSpy R2](https://airspy.com/airspy-r2/)**: High-dynamic-range receiver with 12-bit ADC. 24 MHz - 1.8 GHz, 2.5/10 MHz sample rate. Requires SoapySDR (see below).
*   **[AirSpy HF+ Discovery](https://airspy.com/airspy-hf-discovery/)**: Precision HF/VHF receiver. 0.5 kHz - 31 MHz + 60-260 MHz, up to 768 kHz bandwidth. Requires SoapySDR (see below).
*   Any other device supported by **[SoapySDR](https://github.com/pothosware/SoapySDR)** via the `soapy:<driver>` device type.

### Device Characteristics

Different SDR devices have very different capabilities, and settings that work well on one device may need adjusting on another. Understanding these differences helps you get the best results.

| | RTL-SDR Blog V4 | HackRF One | AirSpy R2 | AirSpy HF+ Discovery |
| :--- | :--- | :--- | :--- | :--- |
| **Frequency range** | 24 MHz - 1.7 GHz | 1 MHz - 6 GHz | 24 MHz - 1.8 GHz | 0.5 kHz - 31 MHz, 60-260 MHz |
| **Max bandwidth** | 2.4 MHz | 20 MHz | 10 MHz | 768 kHz |
| **ADC resolution** | 8-bit | 8-bit | 12-bit (16-bit effective) | 18-bit |
| **Sensitivity** | Good | Moderate | Very good | Excellent (HF specialist) |
| **AGC** | Hardware AGC | No AGC | Via SoapySDR | Hardware AGC (multi-loop) |
| **Gain stages** | Single (auto or manual) | LNA + VGA (manual only) | LNA + Mixer + VGA | LNA (on/off) + RF attenuator |
| **Best for** | General VHF/UHF, low cost | Wideband monitoring | High-quality VHF/UHF | HF and VHF precision |
| **Sample rates** | Up to 2.4 MHz | 2-20 MHz | 2.5 or 10 MHz | 0.192-0.912 MHz (discrete) |

**Key practical differences:**

- **Sensitivity and SNR thresholds**: Higher-sensitivity devices (AirSpy HF+, AirSpy R2) detect weaker signals than the RTL-SDR. This means an `snr_threshold_db` that works well on RTL-SDR (e.g., 4.5 dB) may trigger on too many weak/noisy signals on an AirSpy. Consider raising the threshold to 6-10 dB for higher-sensitivity devices, *and* enable [`activation_variance_db`](#rejecting-emptynoise-recordings) to filter out the noise triggers that the SNR check can't catch.

- **Sample rates**: The AirSpy HF+ Discovery only supports specific discrete sample rates (0.192, 0.228, 0.384, 0.456, 0.650, 0.768, 0.912 MHz). If you request an unsupported rate, the device will use the nearest supported rate and a warning will be logged. Always check the supported rates in the startup log and set `sample_rate` accordingly.

- **Gain architecture**: Each device has a different gain structure. The HackRF has no automatic gain control — it will warn and set sensible defaults if you use `sdr_gain_db: auto`. The AirSpy HF+ Discovery has an RF attenuator (negative dB range) rather than a conventional gain amplifier. See the [Gain Tuning](#gain-tuning) section for details.

- **Band definitions**: Because of these differences, you may want separate band entries for different devices. For example, `air_civil_bristol` (1.024 MHz, threshold 4.5 dB) for RTL-SDR and `air_civil_bristol_airspyhf` (0.912 MHz, threshold 6-8 dB) for the AirSpy HF+.

## Key Features & Optimizations
- **Advanced Signal Detection**: Uses Welch's Power Spectral Density (PSD) estimation for stable, low-variance activity detection. The noise floor is EMA-smoothed across slices to eliminate jitter, with a warmup period that absorbs SDR hardware startup transients before detection begins.
- **Statistical Noise Rejection**: A temporal variance check at channel turn-on distinguishes real signals (voice, data bursts — high variance) from stationary noise (low variance), eliminating the "empty hiss" recordings that plague high-sensitivity receivers. Works for any modulation type. See [Rejecting empty/noise recordings](#rejecting-emptynoise-recordings) below.
- **Parallel Multi-Channel Recording**: Simultaneously detects and records all active channels in a band - unlike traditional handheld scanners which only play one channel at a time.
- **High-Fidelity Demodulation**: Implements stateful AM and NFM demodulation with continuous phase tracking and DC-blocking, eliminating pops and discontinuities between audio blocks.
- **Precise Transition Trimming**: After coarse PSD-based detection, demodulated audio is scanned at sample level to find exact signal boundaries. Padding is added around the boundary and faded with a half-cosine S-curve, preserving signal content (including attack transients) while eliminating clicks.
- **Hardware Efficiency**:
    - **Vectorized Math**: Heavy processing is delegated to NumPy and SciPy for maximum throughput.
    - **Zero-Copy Architecture**: Uses memory stride tricks for overlapping FFT segments, avoiding expensive data copying.
    - **Lazy Evaluation**: Computationally expensive segment analysis is performed only when transitions are detected, drastically reducing idle CPU load.
    - **Pre-Allocated Ring Buffer**: Per-channel audio buffering uses a fixed NumPy array with modulo wrap-around, eliminating per-flush concatenation and GC pressure.
- **State-of-the-Art Processing**:
    - **Vectorized AGC**: High-quality, smooth automatic gain control for AM with independent attack and release timings.
    - **Noise-Floor-Guided Spectral Subtraction**: The band-wide PSD noise floor is passed to the spectral subtraction stage for more reliable noise frame classification, reducing musical noise artifacts compared to percentile-only heuristics.
    - **Float64 Filter State**: IIR filter states (channel extraction, decimation) use double precision to prevent rounding drift in long-running sessions.
- **Parallel Scanning**: Supports multiple SDR devices (RTL-SDR and HackRF) simultaneously with asynchronous I/O.
- **Archive Ready**: Automatic recording to Broadcast WAV (BWF) with embedded metadata (frequency, timestamps, modulation).

## Quick Start
1) Install dependencies (see `installation.txt` for SDR drivers).
2) Configure bands in `config.yaml`.
3) Install package in editable mode:
```bash
pip install -e .
```
4) Run:

```bash
substation --band air_civil_bristol --device-type rtlsdr --device-index 0
```

Audio files are written to:
```
./audio/YYYY-MM-DD/<band>/<timestamp>_<band>_<channel>_<snr>dB_<device>_<index>.wav
```

## Command Line
```bash
substation --band <band> [--config <path>] [--device-type rtlsdr|hackrf|airspy|airspyhf|soapy:<driver>] [--device-index N]
substation --list-bands
```

## Python Module Usage
You can also use the scanner as a library in your own code. This allows you to respond to radio events programmatically.

```python
import asyncio

import substation.config
import substation.scanner

# State Callback: Triggered whenever a signal starts or stops
def my_state_handler (band: str, ch: int, active: bool, snr: float) -> None:
	print (f"Channel {ch} is now {'ON' if active else 'OFF'} ({snr:.1f} dB)")

# Recording Callback: Triggered when a file is finalized and closed
def my_recording_handler (band: str, ch: int, file_path: str) -> None:
	print (f"Recording finished: {file_path}")

async def main () -> None:

	"""
	Initialize the scanner and respond to real-time events.
	"""

	# Load configuration
	config_data = substation.config.load_config ("config.yaml")

	# Initialize scanner instance
	scanner = substation.scanner.RadioScanner (
		config=config_data,
		band_name="pmr",
		device_type="rtlsdr"
	)

	# Register the handlers
	scanner.add_state_callback (my_state_handler)
	scanner.add_recording_callback (my_recording_handler)

	# Start the asynchronous scan loop
	await scanner.scan ()

if __name__ == "__main__":
	asyncio.run (main ())
```

See [examples/scan_demo.py](examples/scan_demo.py) for a more detailed implementation.

Options:
- `--config`, `-c`: path to config file (default `config.yaml`).
- `--band`, `-b`: band name to scan (required unless `--list-bands`).
- `--device-type`, `-t`: `rtlsdr`, `hackrf`, `airspy`, `airspyhf`, or `soapy:<driver>` (default `rtlsdr`).
- `--device-index`, `-i`: device index (default `0`).
- `--list-bands`: list available bands and exit.

## Configuration
Config file is YAML. The top-level keys are `scanner`, `recording`, `band_defaults`, and `bands`.

Scanner
```
scanner:
  sdr_device_sample_size: 131072
  band_time_slice_ms: 200
  sample_queue_maxsize: 30
  calibration_frequency_hz: 93.7e+6
  stuck_channel_threshold_seconds: 60
```
- `sdr_device_sample_size`: number of IQ samples per SDR callback. Higher values reduce callback overhead but increase latency.
- `band_time_slice_ms`: time slice used for PSD/SNR detection. Must be a multiple of `sdr_device_sample_size` (rounded up internally).
- `sample_queue_maxsize`: async queue depth. 10-50 is typical; higher tolerates bursts but uses more RAM.
- `calibration_frequency_hz`: optional known signal for PPM correction; set to `null` to disable.
- `stuck_channel_threshold_seconds`: optional duration in seconds after which a constant signal will trigger a "Stuck Channel" warning. Useful for identifying interference or stuck transmitters. Set to `null` to disable.

Recording
```
recording:
  buffer_size_seconds: 30
  disk_flush_interval_seconds: 5
  audio_sample_rate: 16000
  audio_output_dir: "./audio"
  fade_in_ms: 15
  fade_out_ms: 50
  soft_limit_drive: 1.25
```
- `buffer_size_seconds`: max in-memory audio per channel before drops.
- `disk_flush_interval_seconds`: how often to flush to disk.
- `audio_sample_rate`: output WAV rate (Hz).
- `fade_in_ms`/`fade_out_ms`: half-cosine fades applied to the padding region at channel start/stop (signal content is never attenuated).
- `soft_limit_drive`: post-processing soft limiter drive. Typical range 1.5-3.0 (higher = stronger limiting).
- `noise_reduction_enabled`: toggle spectral subtraction noise reduction (default: true).
- `recording_hold_time_ms`: duration in ms to continue recording after signal drops below threshold (default: 500).

Band Defaults
```
band_defaults:
  AIR:
    channel_spacing: 8.333e+3
    modulation: AM
    snr_threshold_db: 4.5
    sdr_gain_db: 30
```
These settings are merged into each band of the same `type`.

Bands
```
bands:
  air_civil_bristol:
    type: AIR
    freq_start: 125.5e+6
    freq_end: 126.0e+6
    sample_rate: 1.0e+6
    exclude_channel_indices: [33, 34]
```
Per-band keys:
- `freq_start` / `freq_end`: Hz.
- `channel_spacing`: Hz.
- `sample_rate`: Hz. Must cover the band plus margins; higher rates increase CPU.
- `channel_width`: optional; defaults to `channel_spacing * 0.84`.
- `type`: used to inherit defaults from `band_defaults`.
- `modulation`: `AM` or `NFM`.
- `recording_enabled`: enable recording for this band. Optional, defaults to `false` (can also be set in `band_defaults`).
- `snr_threshold_db`: detection threshold (dB above noise floor).
- `activation_variance_db`: optional minimum power variance (dB) across the detection window required for a channel to be considered active. Filters out stationary-noise triggers. Applies to all bands regardless of recording state. See [Rejecting empty/noise recordings](#rejecting-emptynoise-recordings) below. Defaults to `3.0`; set to `0` to disable.
- `sdr_gain_db`: numeric or `auto`.
- `sdr_gain_elements`: optional dict mapping gain element names to dB values for per-stage control (e.g., `{LNA: 10, MIX: 5, VGA: 12}`). Available elements are logged at startup. Takes priority over `sdr_gain_db`.
- `sdr_device_settings`: optional dict of device-specific settings passed via SoapySDR (e.g., `{biastee: "true"}`). Available settings are logged at DEBUG level on startup.
- `exclude_channel_indices`: 0-based indices to skip (no analysis, no recording).

## SoapySDR Installation (AirSpy and other devices)

AirSpy devices (and any other `soapy:<driver>` device) require SoapySDR, which is installed at the system level:

```bash
# Raspberry Pi OS / Debian
sudo apt install -y soapysdr-tools python3-soapysdr
sudo apt install -y soapysdr-module-airspy      # AirSpy R2
sudo apt install -y soapysdr-module-airspyhf    # AirSpy HF+ Discovery

# If soapysdr-module-airspyhf is not in your distro's repos (e.g., Raspberry Pi OS),
# build from source instead:
sudo apt install -y libairspyhf-dev libsoapysdr-dev cmake
git clone https://github.com/pothosware/SoapyAirspyHF.git
cd SoapyAirspyHF && mkdir build && cd build
cmake .. && make && sudo make install && cd ../..

# Verify SoapySDR can see connected devices
SoapySDRUtil --find
```

The Python virtual environment **must** be created with `--system-site-packages` to access the system-installed SoapySDR bindings:

```bash
python3 -m venv --system-site-packages /home/si/venvs/substation
```

## Broadcast WAV (BWF) & Metadata
Each recording captures industry-standard **Broadcast WAV (BWF)** metadata (EBU Tech 3285). This embeds technical details directly into the audio file, making it ideal for archival and automated post-processing.

**Compatibility**: These are standard `.wav` files. They will play perfectly in any normal audio player (VLC, Windows Media Player, Audacity, mobile devices, etc.).

### Metadata Example
If you open a recording in a professional audio tool or a BWF viewer, you will see fields like these:

| Field | Example Value | Description |
| :--- | :--- | :--- |
| **Description** | `{"band":"pmr","channel_index":0,"channel_freq":446006250.0}` | Machine-readable JSON with channel details |
| **Coding History** | `A=PCM,F=16000,W=16,M=mono,T=NFM;Frequency=446.00625MHz` | Technical signal chain (Algorithm, Rate, Modulation) |
| **Originator** | `Substation` | The software that created the file |
| **Origination Date** | `2026-01-27` | Date the recording started |
| **Time Reference** | `1152000` | Sample count since midnight (for precise timing) |


## AirSpy Examples

Scan PMR446 with an AirSpy R2 (higher dynamic range than RTL-SDR, with per-element gain control):

```bash
substation --band pmr --device-type airspy --device-index 0
```

To fine-tune the AirSpy R2's gain stages for best noise figure, set per-element gains in `config.yaml` instead of a single `sdr_gain_db` value. Available element names and their ranges are logged at INFO level on startup — use those to guide your values:

```yaml
bands:
  pmr:
    type: PMR
    freq_start: 446.00625e+6
    freq_end: 446.19375e+6
    sample_rate: 2.5e6
    sdr_gain_elements:
      LNA: 10     # Adjust based on ranges shown in startup log
      MIX: 5
      VGA: 12
```

Scan HF shortwave bands with an AirSpy HF+ Discovery:

```bash
substation --band amateur_hf_20m --device-type airspyhf --device-index 0
```

The HF+ Discovery has a maximum bandwidth of 768 kHz, so `sample_rate` must be set accordingly:

```yaml
bands:
  amateur_hf_20m:
    freq_start: 14.0e+6
    freq_end: 14.35e+6
    channel_spacing: 3.0e+3
    sample_rate: 768.0e+3
    modulation: AM
    recording_enabled: true
    snr_threshold_db: 6.0
    sdr_gain_db: auto
```

## Gain Tuning

SDR gain controls how much the received signal is amplified before digitisation. Too little gain and weak signals are lost in the noise floor; too much and strong signals overdrive the ADC, causing distortion and spurious detections.

**Simple approach (recommended starting point)**: set `sdr_gain_db` to a numeric value or `auto`. When set to a single number, SoapySDR distributes the gain across the device's internal stages automatically — this produces good results for most setups without any per-element knowledge. Start here and only move to per-element tuning if you want to squeeze out the last bit of performance.

**Per-element tuning (advanced)**: devices with multiple gain stages (like the AirSpy R2) allow individual control via `sdr_gain_elements`. This can improve reception quality because the *order* of gain stages matters for noise performance:

| Stage | Role | Tuning guidance |
| :--- | :--- | :--- |
| **LNA** (Low-Noise Amplifier) | First amplifier in the chain. Has the greatest impact on overall noise figure. | Set as high as possible without overloading from strong nearby signals. This is where sensitivity is won or lost. |
| **Mixer** | Frequency conversion stage. | Moderate gain. Too high increases intermodulation distortion (ghost signals from mixing products of strong stations). |
| **VGA** (Variable Gain Amplifier) | Final gain stage before the ADC. | Use to bring the overall signal level into the ADC's optimal range. Boosting here amplifies noise from earlier stages equally, so it contributes the least to sensitivity. |

The general principle is: **maximise gain early in the chain** (LNA) and **minimise gain late** (VGA), within the limits of what doesn't cause overload. This keeps the signal-to-noise ratio as high as possible through the receive chain.

**Device-specific gain notes**:

*RTL-SDR*: Simple single-stage gain. `sdr_gain_db: auto` enables hardware AGC which works well for most bands. Manual values of 20-40 dB are typical.

*HackRF One*: No hardware AGC — `sdr_gain_db: auto` will set sensible defaults (LNA=32, VGA=30) and log a warning. For manual control, the value is clamped to hardware step sizes (LNA: 0-40 in 8 dB steps, VGA: 0-62 in 2 dB steps).

*AirSpy R2*: Three gain stages (LNA, Mixer, VGA). Start with `sdr_gain_db: auto` or a moderate overall value. For per-element control, maximise LNA first, set Mixer moderate, and use VGA to fine-tune.

*AirSpy HF+ Discovery*: Has an unusual gain architecture — the RF element is an **attenuator** (range -48 to 0 dB, where 0 means no attenuation) and the LNA is a simple on/off (0 or 6 dB). The `sdr_gain_db: auto` mode engages the device's built-in multi-loop AGC, which is a good starting point. For manual control:

```yaml
sdr_gain_elements:
  LNA: 6       # LNA on (maximum sensitivity)
  RF: 0        # No attenuation (maximum signal)
```

If you're getting too many false triggers on weak signals, you can add attenuation:

```yaml
sdr_gain_elements:
  LNA: 6
  RF: -10      # 10 dB attenuation — reduces noise triggers
```

**SNR threshold tuning**:

The `snr_threshold_db` setting controls how far above the noise floor a signal must be before it's detected. The right value depends on your device's sensitivity:

- **RTL-SDR**: 4-5 dB works well — the 8-bit ADC limits sensitivity naturally.
- **AirSpy R2 / HF+ Discovery**: Start at 6-8 dB. These devices see signals the RTL-SDR can't, so a higher threshold filters out weak transmissions that would produce noisy recordings.
- If you're getting recordings that are mostly noise, raise the threshold by 1-2 dB at a time.
- If you're missing transmissions you can hear on a handheld scanner, lower it.
- The OFF threshold is always 3 dB below the ON threshold (hysteresis) to prevent rapid toggling.

**General tips**:
- Available gain element names and their valid ranges are logged at INFO level on startup. Check these before setting values.
- Optimal values depend on your antenna, band, and local RF environment — a rooftop antenna in a city needs different gain from a small whip in a rural area.
- Airband (AM, 118-137 MHz) typically needs less gain than PMR (NFM, 446 MHz) because aircraft transmitters are more powerful (5-25W) than PMR handhelds (0.5W).

## Rejecting empty/noise recordings

### The problem

SNR thresholds detect any signal that's louder than the noise floor — but they can't distinguish a *real* signal from a *noisy* one. With sensitive receivers like the AirSpy HF+ Discovery, you'll often see channels register 6-10 dB SNR yet contain only hissing static when played back. Raising `snr_threshold_db` doesn't help: the SNR is genuinely high, because the noise in that channel really is louder than the band-wide noise floor.

What's needed is a way to tell **stationary noise** apart from **real, time-varying signals**.

### The solution: temporal variance

Real signals fluctuate over time:

- **Voice** (AM airband, NFM PMR/marine): syllables, gaps, attack and release create 5-15 dB power swings within a 200 ms detection window
- **TDMA data** (TETRA, DMR): timeslot structure produces sharp on/off transitions
- **Burst data** (ACARS, VDL Mode 2): the entire transmission is a short burst, with quiet on either side
- **Beacons / morse**: dot/dash patterns create clear modulation

Stationary noise — atmospheric, thermal, or computer-generated interference — produces **near-constant power** across the same window. Its temporal standard deviation is close to the natural sampling variance of an ideal PSD estimate (1-2 dB), regardless of how high its average power crosses the SNR threshold.

The scanner measures the standard deviation of each channel's power across the 8 segment PSDs that make up the Welch detection window. At the moment a channel transitions from inactive to active, if that standard deviation falls below `activation_variance_db`, the activation itself is suppressed — the channel never enters the active state, so no detection event fires and no recording is created. The check applies to every band regardless of whether recording is enabled.

### Example

Imagine a "noisy" channel with average power 9 dB above the noise floor and a real voice transmission also at 9 dB SNR:

| Source | Avg SNR | Per-segment power (dB above floor) | Std dev |
| :--- | :--- | :--- | :--- |
| Stationary noise | 9 dB | 9.1, 8.8, 9.0, 9.2, 8.9, 9.1, 8.7, 9.2 | **0.18 dB** |
| Voice transmission | 9 dB | 4.0, 12.5, 14.1, 7.0, 13.8, 11.2, 5.5, 3.9 | **4.3 dB** |

With `activation_variance_db: 3.0`, the noise is suppressed (0.18 < 3.0) and the voice is recorded (4.3 > 3.0). Both have the *same average SNR* — the variance check is what distinguishes them.

### Configuration

```yaml
bands:
  air_civil_bristol_airspyhf:
    type: AIR
    freq_start: 125.5e+6
    freq_end: 126.0e+6
    sample_rate: 0.912e6
    snr_threshold_db: 6
    activation_variance_db: 3.0  # Reject stationary-noise triggers
    sdr_gain_db: auto
```

The check runs **only at the moment a channel turns on**. Once a recording is in progress, brief gaps in the audio (silences between words) don't interrupt it — those are handled by the existing hold-time logic. Recordings end naturally when the SNR drops below the OFF threshold for longer than the hold time.

### How it interacts with other settings

| Setting | Relationship |
| :--- | :--- |
| `snr_threshold_db` | Runs first. Channels below the SNR threshold never get evaluated by the variance check. |
| `activation_variance_db` | Runs second, only on turn-on transitions, only when the SNR check passed. |
| Hysteresis (built-in 3 dB margin) | Unchanged. Once a recording starts, it continues until SNR drops below `snr_threshold_db - 3`. |
| Hold time (`recording_hold_time_ms`) | Unchanged. Brief drops in SNR during active recording are tolerated. |

The variance check is fundamentally a *gate*, not a *filter* — it decides whether to consider a channel active, then steps out of the way.

### Tuning guidance

| Symptom | Action |
| :--- | :--- |
| Default works | Leave it alone — the default of `3.0 dB` is chosen to cleanly separate noise from any modulated signal |
| Real signals (voice, data) being rejected | Lower the threshold: try `2.0` or `2.5` |
| Noise still triggers recordings | Raise the threshold: try `4.0` or `5.0` |
| Want to disable entirely | Set to `0` |
| Want to compare with/without | Run two scanner instances on different bands, one with the field set, one with `activation_variance_db: 0` |

### How to confirm it's working

Suppression events are logged at **DEBUG** level (they're frequent enough that logging them at INFO would clutter normal output). Run the scanner with debug logging enabled to see them:

```bash
PYTHONUNBUFFERED=1 substation --band air_civil_bristol_airspyhf --device-type airspyhf 2>&1 | grep suppressed
```

Or set the root log level to DEBUG by modifying the logging configuration. You'll then see lines like:

```
Channel 18 suppressed: power variance 0.4 dB below threshold 3.0 dB (likely noise)
```

This means the variance check rejected an activation that the SNR check would have accepted. Real signals don't produce this log line — they pass straight through to recording.

If you've enabled debug logging and see *no* suppression lines while still getting empty recordings, it means the noise has more variance than expected (e.g., bursty interference rather than constant hiss). Raise the threshold or investigate the interference source.

### Generality

The check operates on raw channel power computed from FFT bins, not on demodulated audio. This means it works identically for:

- AM airband voice
- NFM PMR, marine VHF, business radio, amateur 2m
- TDMA data (TETRA, DMR) — frame structure produces strong variance
- Burst data (ACARS, VDL Mode 2) — bursts produce maximum variance against silence
- Any future modulation type added to the scanner — no demodulator-specific tuning needed

## Parallel Scans (Multiple Devices)
Run one process per device:

```bash
substation --band air_civil_bristol --device-type rtlsdr --device-index 0
substation --band pmr --device-type rtlsdr --device-index 1
```

If you need stricter real-time behavior, you can pin each scan to a CPU core:

```bash
taskset -c 2 substation --band air_civil_bristol --device-index 0
taskset -c 3 substation --band pmr --device-index 1
```

## Resource and Performance Notes
- **Sample rate dominates CPU**. Large bands at high sample rates increase FFT/PSD load.
- **Overrun warnings** indicate the processing of a slice exceeded its real-time window. This can lead to dropped IQ blocks (`Sample queue full`).
- **Noise reduction** runs during write/flush if enabled (default). It uses `apply_spectral_subtraction` which is efficient and receives the band-wide noise floor for improved frame classification. The alternative `apply_noisereduce` implementation exists in `substation/dsp/noise_reduction.py` for reference but is not used by default as it is significantly more CPU-intensive.
- **Queue size** provides burst tolerance but uses RAM (each slice can be several MB).

If you see repeated `Sample queue full` warnings, reduce the band's `sample_rate`, exclude channels, or increase `sample_queue_maxsize`.

## Limitations
- Processing is slice-based; extremely wide bands or multiple high-rate scans can exceed real-time capacity on low-power CPUs.
- If you enable `apply_noisereduce` (requires code change), it is CPU-intensive for long chunks; on constrained devices, stick with the default `apply_spectral_subtraction` or reduce `disk_flush_interval_seconds`.

## Author
Written by Simon Holliday ([https://simonholliday.com/](https://simonholliday.com/))

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- **Copyleft**: Any modifications or improvements to this software must be shared back under the same license, even if used over a network.
- **Attribution**: You must give appropriate credit to the original author (Simon Holliday).
- **Commercial Use**: Permitted, provided you comply with the copyleft obligations of the AGPL-3.0.

See the [LICENSE](LICENSE) file for the full legal text.
