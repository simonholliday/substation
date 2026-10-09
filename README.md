# Substation

**An SDR band scanner that detects, demodulates, and records radio transmissions automatically.**

Connect a USB SDR receiver, point it at a frequency band - Amateur, CB, Airband, PMR, Maritime, or any conventional analogue band - and Substation monitors every radio channel simultaneously, detecting each transmission and recording it to its own audio file with full metadata. Out of the box it records only the bands that UK law opens to anyone, such as amateur and CB radio, and detects activity on the rest: see [Reception and the law](#reception-and-the-law).

The scanner is designed for unattended, long-running operation. It handles the entire signal processing chain from raw IQ samples through to clean, archive-ready audio files: signal detection, demodulation (NFM, AM, USB, LSB), noise reduction, carrier transient removal, soft limiting, and automatic file management. Noise rejection checks each activation's RF power variance and audio spectral flatness, and each finished recording's length and spectral flatness, and discards what looks like hiss. At present the flatness checks reject hiss only on AM bands: see [Limitations](#limitations). Recordings include embedded metadata - frequency, timestamp, modulation, and any CTCSS or DCS tone detected - so every file is self-documenting.

Substation runs as a command-line tool or as a Python module in your own applications, including on low-power hardware such as a Raspberry Pi scanning a narrower band. The widest shipped bands, at 12.5 MHz, have not yet been shown to keep up in real time, and CTCSS and DCS tone detection has not yet been thoroughly tested with real radios: see [Limitations](#limitations).

**Full documentation: [https://subsystem.co/substation/](https://subsystem.co/substation/)**

- Guide: [https://subsystem.co/substation/guide](https://subsystem.co/substation/guide)
- Configuration reference: [https://subsystem.co/substation/configuration](https://subsystem.co/substation/configuration)
- Command-line reference: [https://subsystem.co/substation/command-line](https://subsystem.co/substation/command-line)
- Python reference: [https://subsystem.co/substation/reference](https://subsystem.co/substation/reference)

The guide starts from what a software-defined radio is and goes as far as a scanner left running, feeding recordings to Subsample and events to Subsequence. The references are generated from Substation's own code, so they describe the release the guide installs.

For changing Substation's code, see [docs/architecture.md](docs/architecture.md).

## Supported receivers

- **RTL-SDR Blog V4 or V3**: low cost, for general VHF and UHF listening.
- **HackRF One**: wideband, from 1 MHz to 6 GHz.
- **Airspy R2**: high-quality VHF and UHF.
- **Airspy HF+ Discovery**: HF, and VHF from 64 to 260 MHz, with high sensitivity.
- **Any other receiver with a SoapySDR driver module**, as `--device-type soapy:<driver>`.

[Your first receiver](https://subsystem.co/substation/guide/your-first-receiver) compares them, and [Setting up your receiver](https://subsystem.co/substation/guide/setting-up-your-receiver) shows how to set each one up.

## Quick start

Substation needs your receiver's driver installed first, which the guide's [install chapter](https://subsystem.co/substation/guide/installing-substation) shows for each receiver. Then:

```bash
pip install substation       # or "substation[hackrf]" for a HackRF One
substation --init            # optional: writes ./config.yaml, the fully commented defaults
substation --band amateur_2m # scans the 2m amateur band with an RTL-SDR
```

The scanner logs each radio channel on the 2m amateur band as it becomes active, and records each transmission to its own file. The scan is running once the log says `Detection enabled`. The 2m band can be quiet, so a first recording may take a while. Out of the box only amateur and CB bands record: the rest, such as airband and PMR446, only detect, as the next section explains. Recordings are written to:
```
./audio/YYYY-MM-DD/<band>/<date>_<time>_<band>_<channel>_<freq>_<snr>dB_<device>_<index>.wav
```

To install the latest code from GitHub instead, run `pip install git+https://github.com/simonholliday/substation.git`. From here, the guide goes from [a first scan](https://subsystem.co/substation/guide/a-first-scan) to a scanner left running.

## Reception and the law

Many radio services may not lawfully be listened to without permission, and the law differs from country to country. Every band Substation ships carries a `reception_class` saying how UK law treats it, and the class decides whether the band records out of the box:

| Class | What it covers | Out of the box |
| :--- | :--- | :--- |
| `general` | What Ofcom calls general reception: licensed broadcasting, amateur and CB radio, and weather and navigation transmissions | Records |
| `not_general` | Services outside general reception, such as PMR446, business radio, marine, military airband, and emergency services, which Ofcom says it is illegal to listen to | Detects activity without recording |
| `unsettled` | Bands where the position is unclear, such as civil airband, where Ofcom will not say that listening is an offence | Detects activity without recording |

In the UK, using a receiver to learn the contents, sender, or addressee of a transmission that is not general reception is an offence under the Wireless Telegraphy Act 2006, even if you tell no one. Elsewhere the law differs: the United States, for example, allows receiving unencrypted public-safety, marine, and air radio, and Germany forbids it. The classes describe UK law only, and are not legal advice: the law where you are decides what you may receive and record.

Where your law allows it, switch recording on for a band in your `config.yaml`:

```yaml
bands:
  air_civil_bristol:
    recording_enabled: true
```

`--list-bands` shows each band's class and whether it records. A band you define yourself takes its template's class when it sets none of its own, so a band of `type: CB` records and one of `type: PMR` only detects. A band with no class records only if you give it `reception_class: general` or `recording_enabled: true`.

## Limitations
- Processing time grows with the number of IQ samples a second a band takes, and most of it runs on one CPU core. The three shipped bands at 12.5 MHz, `air_civil_1`, `air_civil_2`, and `dmr`, have not yet been shown to keep up in real time: on the one desktop computer they have been tested on, processing fell behind and IQ samples were dropped, and faster computers are still to be tested. On a Raspberry Pi, or wherever `Processing overrun` warnings appear, scan a narrower band, such as one of `dmr_1` to `dmr_5`.
- CTCSS and DCS tone detection has not yet been thoroughly tested with real radios, so treat a reported tone as a guide rather than a certainty, and the absence of one as inconclusive.
- The spectral flatness checks reject hiss only on AM bands at present. The NFM, USB, and LSB demodulators filter their audio to the voice band before the checks measure it, and the checks measure the whole audio spectrum, so the empty frequencies outside the voice band make hiss measure as peaked, far below the threshold of 0.15. On those bands, which include every band that records out of the box, both checks keep hiss, and noise rejection rests on the SNR threshold and the RF power variance check. Fixing this needs a measure and a threshold chosen against real recordings of voice and hiss. [Recordings of nothing](https://subsystem.co/substation/guide/recordings-of-nothing) explains each check.
- If you enable `apply_noisereduce` (requires a code change and the `noisereduce` extra), it is CPU-intensive for long chunks; on constrained devices, stick with the default `apply_spectral_subtraction` or reduce `disk_flush_interval_seconds`.

## Author
Written by Simon Holliday ([https://simonholliday.com/](https://simonholliday.com/))

This project is managed with [Subroutine](https://github.com/simonholliday/subroutine).

## Licence

Substation is released under the [GNU Affero General Public License v3.0](https://github.com/simonholliday/substation/blob/main/LICENSE) (AGPLv3).

You are free to use, modify, and distribute this software under the terms of the AGPL. If you run a modified version of Substation as part of a network service, you must make the source code available to its users.
