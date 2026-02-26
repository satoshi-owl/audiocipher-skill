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
- **Decode** a received WAV back to plain text — including auto-detecting the cipher mode
- **Hide an image inside audio** so it appears visually in a spectrogram (Aphex Twin technique)
- **Analyze any audio** for hidden content: QR codes, embedded text, images, cipher tones, anomalies
- **Render a spectrogram PNG** from any audio file for visual inspection
- **Generate a branded MP4** from audio for posting to X/Twitter (agents can't post raw audio)

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

### Record a message and post it to X/Twitter

```bash
python3 audiocipher.py encode "NULL" --output null.wav
python3 audiocipher.py video null.wav --output null.mp4 --title "NULL"
# Upload null.mp4 to X — branded waveform video, audio-encoded message inside
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
| `decode` | Decode a received cipher WAV back to text |
| `spectrogram` | Render a spectrogram PNG from any audio file |
| `image2audio` | Hide an image inside a spectrogram |
| `analyze` | Full detection pipeline — QR codes, text, images, cipher tones |
| `video` | Wrap audio in a branded MP4 for posting to X/Twitter |

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
python3 audiocipher.py video cipher.wav --output cipher.mp4 --title "NULL"
```
Produces a 1280×720 H.264 MP4 — dark background, brand-green waveform glow, AUDIOCIPHER badge, `audiocipher.app` watermark. Twitter/X compatible.

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

# Decode
result = subprocess.run(['python3', 'audiocipher.py', 'decode', wav_path], capture_output=True, text=True)
decoded_text = result.stdout.strip()

# Video (always wrap before posting to X/Twitter)
subprocess.run(['python3', 'audiocipher.py', 'video', 'out.wav', '--output', 'out.mp4', '--title', 'NULL'])
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
