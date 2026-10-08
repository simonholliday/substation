# Architecture

This file is for people changing Substation's code, and subsystem.co does not
publish it. For using Substation, see [https://subsystem.co/substation/](https://subsystem.co/substation/).

Substation's signal processing is written in Python, with NumPy and SciPy doing the numerical work.

## Detection

The scanner divides the SDR's bandwidth into radio channels and analyses each one several times a second using Welch's Power Spectral Density method. Welch averaging across multiple overlapping FFT segments reduces noise variance, producing stable SNR measurements that don't jitter between slices. The noise floor tracks slowly via an exponential moving average, so brief transmissions stand out clearly against a stable background. A warmup period at startup absorbs the transient spikes that SDR hardware produces while its PLL and AGC settle.

The centre frequency is automatically shifted by half a radio channel spacing whenever a radio channel would fall on the DC spike - a common SDR artefact caused by LO leakage - so no radio channel is ever masked.

## Noise rejection

High-sensitivity receivers often trigger on noise that crosses the SNR threshold. Substation rejects these false activations with independent checks:

1. **RF power variance** - real signals (voice, data) fluctuate in power across the detection window; stationary noise does not. Radio channels with low variance are rejected before any demodulation occurs.
2. **Spectral flatness** - when a radio channel first activates, the audio is speculatively demodulated and its spectral flatness (Wiener entropy) is measured. Noise has a flat spectrum; any real signal has a peaked one. Flat-spectrum activations are rejected before a recording starts. At present this works only on AM bands: see [Limitations](../README.md#limitations).
3. **Post-recording checks** - after a recording finishes, it is discarded if its transmission was shorter than `min_recording_seconds`, not counting the hold time recorded after it, or if the complete file, analysed for spectral flatness, is predominantly noise (e.g. a brief signal followed by hold-timer padding). At present the flatness check works only on AM bands: see [Limitations](../README.md#limitations).

## Demodulation

Each modulation type has a dedicated, stateful demodulator that maintains phase and filter continuity across processing blocks, eliminating the pops and glitches that occur at block boundaries in stateless designs.

**NFM** - the most common mode for PMR, amateur, and public safety - runs through a complete processing chain: IF decimation, polar discriminator, Hampel impulse blanker (suppresses glitches from IQ samples dropped over USB by devices like the AirSpy R2), 300µs de-emphasis, DC blocking, voice bandpass filter (300-3400 Hz), and CTCSS/DCS subaudible tone detection. The voice bandpass reduces subaudible signalling in the recording: the lowest CTCSS tones strongly, and the highest, near 250 Hz, only slightly, so they can remain faintly audible. A Goertzel detector looks for CTCSS tones and a Golay decoder reads DCS codes. A tone found is embedded in the file's metadata and delivered live on the scanner's `channel_state` event (as `ctcss_hz` / `dcs_code` kwargs), so OSC or dashboard consumers see the tone as a property of the activation, with no file parsing required. Tone detection has not yet been thoroughly tested with real radios, so treat a reported tone as a guide rather than a certainty, and the absence of one as inconclusive.

**AM** - used for civil and military airband - uses envelope detection with an AGC that follows the audio's peaks, rising at once and releasing slowly, so it adapts to varying signal strength without pumping or clipping.

**SSB** (USB and LSB) - used for HF amateur and maritime - implements the Weaver method for clean sideband separation with real-valued Butterworth filters on I and Q, followed by voice AGC.

## Recording quality

Each recording passes through several stages between demodulation and disk:

- **Spectral subtraction** noise reduction estimates the background hiss from the quietest moments of each recording's first audio, and reduces it while preserving voice clarity. A 2D gain-mask smoothing kernel minimises musical noise artefacts.
- **Carrier transient trimming** (optional) detects and removes the sharp clicks that AM transmitters produce at key-on and key-off, using shape-based detection that distinguishes carrier transients from voice plosives.
- **Half-cosine fades** at recording boundaries prevent clicks from sudden onset or cutoff.
- **Soft limiting** via a tanh waveshaper rounds off peaks as they near full scale: audio up to full scale comes out at no more than 0.98 of it (-0.18 dBFS), leaving headroom for the small overshoot between audio samples that voice-band audio produces.
- **Broadcast WAV metadata** (BEXT, EBU Tech 3285) embeds each recording's start time, frequency, and modulation directly in the file, with any CTCSS tone or DCS code detected. Audio editors like Audacity, Reaper, and iZotope RX can place recordings on a timeline at their real capture time.
- **FLAC output** (optional) compresses recordings losslessly, to a size that depends on the band and the signal, with metadata stored as Vorbis comments. Its compression level was chosen by encoding real PMR recordings on a Raspberry Pi at every level: the highest levels gave almost no further reduction and cost noticeably more CPU time.

## Efficiency

The scanner is designed for 24/7 operation on low-power hardware. All DSP runs through NumPy and SciPy's compiled backends. FFT segments use zero-copy memory stride tricks. Expensive work runs only when it is needed: the segment PSD only when a radio channel changes state, and demodulation only while a radio channel records, or briefly when one turns on, to check it for noise. Audio buffering for each radio channel uses a pre-allocated ring buffer with modulo wrap-around, avoiding per-flush memory allocation. IIR filter states use float64 precision to prevent rounding drift in long-running sessions.
