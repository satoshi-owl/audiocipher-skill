<p align="center">
  <img src="https://audiocipher.app/og-image.png" alt="AudioCipher" width="600" />
</p>

<h1 align="center">AudioCipher — Python CLI Skill</h1>

<p align="center">
  Hide messages in audio. Decode what you find. Post to X.
</p>

<p align="center">
  <a href="https://audiocipher.app">audiocipher.app</a> &nbsp;·&nbsp;
  <a href="https://audiocipher.app/spectrogram">Spectrogram viewer</a>
</p>

---

## What this lets you do

- **Encode** a secret text message as an audio file using frequency-based ciphers
- **Decode** a received WAV or MP4 back to plain text — including auto-detecting the cipher mode
- **Hide an image inside audio** so it appears visually in a spectrogram (Aphex Twin technique)
- **Analyze any audio** for hidden content: QR codes, embedded text, images, cipher tones, anomalies
- **Render a spectrogram PNG** from any audio file for visual inspection
- **Generate a branded MP4** from audio for posting to X/Twitter — ALAC lossless by default so the cipher stays decodable from the video file
- **Self-update** — pull the latest version with one command; agents are notified automatically

Designed for use with OpenClaw agents. All decoders are direct Python ports of [audiocipher.app](https://audiocipher.app) — encode in the browser, decode in code, or vice versa.

---

## Quick Install

```bash
git clone https://github.com/satoshi-owl/audiocipher-skill.git
cd audiocipher-skill
bash install.sh
```

Or manually:

```bash
pip3 install -r requirements.txt
# System packages: ffmpeg, tesseract, zbar (see install.sh)
```

---

## Usage Examples

### Encode a secret message → decode it back

```bash
python3 audiocipher.py encode "HELLO WORLD" --output cipher.wav
python3 audiocipher.py decode cipher.wav
# → HELLO WORLD
```

### Hide an image in audio (visible in a spectrogram)

```bash
python3 audiocipher.py image2audio logo.png --output hidden.wav --fmin 1000 --fmax 8000 --duration 8
python3 audiocipher.py spectrogram hidden.wav --output spec.png --colormap green --labeled
# Open spec.png — your image is in the audio
```

### Analyze a mystery audio file for hidden content

```bash
python3 audiocipher.py analyze mystery.wav --output-dir ./findings/
cat findings/results.json
# Detects: QR codes, embedded text, images, cipher tones, anomalies
# Saves a cropped PNG per finding
```

### Record a message, post to X, decode the MP4 directly

```bash
python3 audiocipher.py encode "NULL" --output null.wav
# Default: ALAC lossless — cipher survives inside the video
python3 audiocipher.py video null.wav --output null.mp4 --title "NULL"
# Decode directly from the MP4 — no separate WAV needed
python3 audiocipher.py decode null.mp4
# → NULL

# Use --twitter for AAC social posting (HZAlpha only — other modes may not survive)
python3 audiocipher.py video null.wav --output null_tw.mp4 --title "NULL" --twitter
```

### Try every cipher mode

```bash
python3 audiocipher.py encode "AUDIOCIPHER" --mode morse  --output morse.wav
python3 audiocipher.py encode "AUDIOCIPHER" --mode dtmf   --output dtmf.wav
python3 audiocipher.py encode "AUDIOCIPHER" --mode fsk    --output fsk.wav

python3 audiocipher.py decode morse.wav   # → AUDIOCIPHER
python3 audiocipher.py decode dtmf.wav    # → AUDIOCIPHER
python3 audiocipher.py decode fsk.wav     # → AUDIOCIPHER
```

---

## All Commands

| Command | What it does |
|---------|-------------|
| `encode` | Encode a secret message as cipher audio (WAV) |
| `decode` | Decode a cipher WAV or MP4 back to text |
| `spectrogram` | Render a spectrogram PNG from any audio file |
| `image2audio` | Hide an image inside a spectrogram |
| `analyze` | Full detection pipeline — QR codes, text, images, cipher tones |
| `video` | Wrap audio in a branded MP4 (ALAC lossless; decodable) |
| `update` | Pull the latest version from GitHub |

### encode
```bash
python3 audiocipher.py encode "HELLO WORLD" --output cipher.wav --mode hzalpha
```
Options: `--mode hzalpha|morse|dtmf|fsk` · `--duration-ms 120` · `--waveform sine|square|sawtooth|triangle` · `--volume 0.8`

### decode
```bash
python3 audiocipher.py decode cipher.wav
# --mode auto reads embedded WAV metadata automatically
```

### spectrogram
```bash
python3 audiocipher.py spectrogram mystery.wav --output spec.png --colormap green --labeled
```
Options: `--colormap green|inferno|viridis|amber|grayscale` · `--fmin` · `--fmax` · `--width 1200` · `--height 600` · `--gain-db 0` · `--labeled`

### image2audio
```bash
python3 audiocipher.py image2audio logo.png --output hidden.wav --fmin 1000 --fmax 8000 --duration 8
```
Best clarity: `--fmin 1000 --fmax 8000 --duration 8`

### analyze
```bash
python3 audiocipher.py analyze mystery.wav --output-dir ./findings/
```

Outputs `findings/results.json` + a cropped PNG per finding:

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
# Default: ALAC lossless audio — cipher is decodable directly from the MP4
python3 audiocipher.py video cipher.wav --output cipher.mp4 --title "NULL"
python3 audiocipher.py decode cipher.mp4   # works!

# For Twitter/X: AAC 320k (HZAlpha survives; WaveSig/FSK/Morse do not)
python3 audiocipher.py video cipher.wav --output cipher.mp4 --twitter
```
Produces an animated 1280×720 H.264 MP4 — dark background, brand-green waveform glow, AUDIOCIPHER badge, `audiocipher.app` watermark. Twitter/X compatible.

Options: `--twitter` (AAC for social posting) · `--resolution 1280x720` · `--title TEXT` · `--verbose`

> **Always use `video` before posting audio to X — agents cannot post raw audio.**

---

## Encoding Modes

| Mode | Frequency range | Notes |
|------|----------------|-------|
| HZ Alpha | 220 Hz – 8872 Hz | Chromatic scale; supports A–Z, 0–9, symbols |
| Morse | User-defined tone | ITU timing; auto-detected on decode |
| DTMF | 697 – 1633 Hz | T9 letter mapping; standard dual-tone pairs |
| FSK | 1000 / 1200 Hz | ASCII → 8-bit; 300 baud default |
| WaveSig | 1875 – 6562 Hz | RS(12,8) over GF(16); 6 simultaneous tones per frame |

Encoded WAVs include embedded JSON metadata so `decode --mode auto` restores parameters automatically.

---

## Agent Integration (OpenClaw)

### Onboarding (runs exactly once)

```python
from onboard import is_complete, run_onboard

if not is_complete():
    result = run_onboard(
        operator_input=user_message,        # text the operator typed
        operator_attachment=wav_file_path,  # WAV they attached, or None
    )
    await send(result['message'])
    if result['attachment']:
        await send_audio(result['attachment'])
    return  # resume normal command handling next turn
```

State is persisted to `~/.audiocipher/onboard_state.json` — never re-triggers after completion.

### Calling commands from an agent

```python
import subprocess

# Encode
subprocess.run(['python3', 'audiocipher.py', 'encode', message, '--output', 'out.wav'])

# Decode WAV or MP4 (ALAC video files decode directly — no separate WAV needed)
result = subprocess.run(['python3', 'audiocipher.py', 'decode', 'out.wav'], capture_output=True, text=True)
decoded_text = result.stdout.strip()

# Video — ALAC lossless by default (cipher stays decodable from the MP4)
subprocess.run(['python3', 'audiocipher.py', 'video', 'out.wav', '--output', 'out.mp4', '--title', 'NULL'])

# For Twitter/X posting use --twitter (AAC, HZAlpha mode recommended)
subprocess.run(['python3', 'audiocipher.py', 'video', 'out.wav', '--output', 'out.mp4', '--twitter'])
```

### Keeping the skill up to date

The skill checks for updates silently after each command and prints a notification to stderr
if a newer version is available:

```
ℹ  Update available (v0.3.0) → python3 audiocipher.py update
```

Agents and operators can update immediately:

```bash
python3 audiocipher.py update
# Pulls latest from GitHub + reinstalls dependencies
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, scipy | DSP, FFT, waveform synthesis |
| soundfile | WAV read/write |
| librosa | High-quality STFT for spectrogram |
| Pillow | Image I/O and spectrogram rendering |
| opencv-python | Contour detection, morphological ops |
| pyzbar | QR / barcode decoding |
| pytesseract | OCR text extraction |
| ffmpeg (system) | Video generation |
| tesseract (system) | OCR engine |
| zbar (system) | QR library backend for pyzbar |
