<p align="center">
  <img src="https://audiocipher.app/og-image.png" alt="AudioCipher" width="600" />
</p>

<h1 align="center">AudioCipher — Python CLI Skill</h1>

<p align="center">
  Encode secret messages as audio. Decode what you find. Post anywhere.
</p>

<p align="center">
  <a href="https://audiocipher.app">audiocipher.app</a> &nbsp;·&nbsp;
  <a href="https://github.com/satoshi-owl/audiocipher-skill">GitHub</a> &nbsp;·&nbsp;
  <a href="https://audiocipher.app/llms.txt">llms.txt</a>
</p>

---

## What this skill does

A Python CLI and library for encoding and decoding hidden messages in audio — and for wrapping results as MP4 video for social posting.

- **Encode** text as cipher audio using 8 frequency-based modes
- **Decode** any WAV, MP4, OGG, or M4A back to plain text — auto-detects cipher mode
- **ABP mode** — OFDM QPSK codec that survives Telegram (OGG Opus) and X/Twitter (AAC) re-encoding
- **Analyze** any audio for hidden content: cipher tones, QR codes, embedded text, images
- **Spectrogram** — render a frequency/time PNG from any audio file
- **Image2audio** — hide an image inside audio so it appears in a spectrogram (Aphex Twin technique)
- **Video** — wrap audio in a branded MP4 for posting to X/Twitter
- **Self-update** — `python3 audiocipher.py update` pulls the latest version

Cross-platform with [audiocipher.app](https://audiocipher.app) — encode in the browser, decode in Python, or vice versa.

---

## Install

```bash
git clone https://github.com/satoshi-owl/audiocipher-skill.git
cd audiocipher-skill
bash install.sh
```

One-liner:

```bash
curl -sSL https://audiocipher.app/install | bash
```

**Requires:** Python 3.9+, ffmpeg, tesseract, zbar (installed automatically by `install.sh`)

---

## Quick start

```bash
# Encode a message
python3 audiocipher.py encode "HELLO WORLD" --output hello.wav

# Decode it back
python3 audiocipher.py decode hello.wav
# → HELLO WORLD

# ABP mode — survives Telegram/X re-encoding
python3 audiocipher.py encode "SECRET" --mode abp --output msg.wav
python3 audiocipher.py decode msg.wav --mode abp
# → SECRET

# ABP with passphrase (XChaCha20 encrypted)
python3 audiocipher.py encode "SECRET" --mode abp --passphrase "hunter2" --output enc.wav
python3 audiocipher.py decode enc.wav --mode abp --passphrase "hunter2"
# → SECRET

# Generate spectrogram
python3 audiocipher.py spectrogram hello.wav --output spec.png --labeled

# Post to X/Twitter — wrap as MP4 first
python3 audiocipher.py video hello.wav --output hello.mp4 --twitter
```

---

## Cipher modes

| Mode | Flag | Frequency range | Notes |
|------|------|----------------|-------|
| HZAlphabet | `hzalpha` | 220–8372 Hz | Chromatic scale; A–Z, 0–9, symbols |
| Morse Code | `morse` | Configurable | ITU timing; auto-detects tone on decode |
| DTMF | `dtmf` | 697–1633 Hz | Standard dual-tone; decodes phone/IVR audio |
| FSK Binary | `fsk` | 1000/1200 Hz | 45, 45.45 (RTTY), or 300 baud (Bell 103) |
| GGWave | `ggwave` | 1875–6562 Hz | RS ECC; ggwave-js compatible |
| AcDense | `acdense` | 300–8000 Hz | Multi-token OFDM bursts; high density |
| HyperDense | `achd` | 300–8000 Hz | Double-channel AcDense |
| **ABP** | `abp` | 300–8000 Hz | **OFDM QPSK + RS FEC + XChaCha20. Survives Telegram + X re-encoding.** |

Encoded WAVs embed all cipher settings as metadata — `decode --mode auto` restores them automatically.

---

## All commands

| Command | What it does |
|---------|-------------|
| `encode` | Encode text as cipher audio (WAV) |
| `decode` | Decode WAV/MP4/OGG → plain text (auto-detects mode) |
| `spectrogram` | Render frequency/time PNG from any audio |
| `image2audio` | Hide an image inside audio (visible in spectrogram) |
| `analyze` | Detect hidden content: cipher tones, QR codes, text, images |
| `video` | Wrap audio in branded MP4 (ALAC lossless or AAC for Twitter) |
| `update` | Pull latest version from GitHub + reinstall dependencies |

### encode

```bash
python3 audiocipher.py encode "HELLO" --output out.wav
python3 audiocipher.py encode "HELLO" --mode morse --output morse.wav
python3 audiocipher.py encode "HELLO" --mode abp --passphrase "secret" --output enc.wav
```

Options: `--mode hzalpha|morse|dtmf|fsk|ggwave|acdense|achd|abp` · `--passphrase PHRASE` (ABP only) · `--abp-profile social_safe|fast` · `--duration-ms 120` · `--waveform sine|square|sawtooth` · `--volume 0.8`

### decode

```bash
python3 audiocipher.py decode cipher.wav              # auto-detects mode
python3 audiocipher.py decode cipher.mp4              # works directly on ALAC video
python3 audiocipher.py decode received.ogg --mode abp # after Telegram re-encode
python3 audiocipher.py decode received.ogg --mode abp --passphrase "secret"
```

### spectrogram

```bash
python3 audiocipher.py spectrogram audio.wav --output spec.png --colormap green --labeled
```

Options: `--colormap green|inferno|viridis|amber|grayscale` · `--fmin` · `--fmax` · `--width 1200` · `--height 600` · `--gain-db 0` · `--labeled`

### image2audio

```bash
python3 audiocipher.py image2audio logo.png --output hidden.wav --fmin 1000 --fmax 8000 --duration 8
python3 audiocipher.py spectrogram hidden.wav --output spec.png  # your image is in here
```

### analyze

```bash
python3 audiocipher.py analyze mystery.wav --output-dir ./findings/
cat findings/results.json
```

Output:
```json
{
  "type": "cipher_hzalpha",
  "confidence": 0.92,
  "decoded_value": "HELLO WORLD",
  "freq_range_hz": [220, 8872],
  "time_range_s": [0.0, 2.4]
}
```

### video

```bash
# ALAC lossless — cipher decodes directly from the MP4
python3 audiocipher.py video cipher.wav --output out.mp4 --title "NULL"
python3 audiocipher.py decode out.mp4   # works

# AAC for X/Twitter
python3 audiocipher.py video cipher.wav --output out.mp4 --twitter
```

> Always use `video` before posting audio to X — agents cannot post raw audio files.

---

## Agent integration

### Install from an agent

```bash
git clone https://github.com/satoshi-owl/audiocipher-skill.git && bash audiocipher-skill/install.sh
# or:
curl -sSL https://audiocipher.app/install | bash
```

### Call from Python

```python
import subprocess

# Encode
subprocess.run(['python3', 'audiocipher.py', 'encode', 'HELLO', '--output', 'out.wav'])

# Decode → get text
result = subprocess.run(['python3', 'audiocipher.py', 'decode', 'out.wav'],
                        capture_output=True, text=True)
message = result.stdout.strip()  # → "HELLO"

# ABP encode with passphrase
subprocess.run(['python3', 'audiocipher.py', 'encode', 'SECRET',
                '--mode', 'abp', '--passphrase', 'hunter2', '--output', 'enc.wav'])

# Video for X/Twitter
subprocess.run(['python3', 'audiocipher.py', 'video', 'out.wav',
                '--output', 'out.mp4', '--twitter'])
```

### Onboarding hook (runs once per install)

```python
from onboard import is_complete, run_onboard

if not is_complete():
    result = run_onboard(
        operator_input=user_message,
        operator_attachment=wav_file_path,  # or None
    )
    await send(result['message'])
    if result['attachment']:
        await send_audio(result['attachment'])
    return
```

State saved to `~/.audiocipher/onboard_state.json` — never re-triggers after completion.

### Auto-update

The skill checks for updates silently after each command:

```
ℹ  Update available (v0.4.0) → python3 audiocipher.py update
```

```bash
python3 audiocipher.py update  # pulls latest + reinstalls deps
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, scipy | DSP, FFT, waveform synthesis |
| soundfile | WAV read/write |
| librosa | STFT for spectrogram |
| Pillow | Image I/O and rendering |
| opencv-python | Contour detection |
| pyzbar | QR / barcode decoding |
| pytesseract | OCR text extraction |
| zstandard | zstd compression (ABP) |
| argon2-cffi | Argon2id KDF (ABP encryption) |
| PyNaCl | XChaCha20-Poly1305 (ABP encryption) |
| ffmpeg (system) | Video generation |
| tesseract (system) | OCR engine |
| zbar (system) | QR library backend |

---

## Version

Current: **v0.4.0** — See [SKILL.md](SKILL.md) for the full capability manifest.
