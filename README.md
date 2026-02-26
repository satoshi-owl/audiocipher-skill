# AudioCipher — Python CLI Skill

A standalone Python CLI for encoding and decoding hidden audio messages, generating spectrogram images from audio, analyzing audio for hidden content, and producing waveform videos for posting to X/Twitter.

Designed for use with OpenClaw agents. Pair with [audiocipher.app](https://audiocipher.app) for browser-based encode/decode.

---

## Quick Install

```bash
git clone https://github.com/audiocipher/audiocipher-skill.git
cd audiocipher-skill
bash install.sh
```

Or manually:

```bash
pip3 install -r requirements.txt
# System packages: ffmpeg, tesseract, zbar (see install.sh)
```

---

## Commands

| Command | What it does |
|---------|-------------|
| `encode` | Encode a secret message as cipher audio (WAV) |
| `decode` | Decode a received cipher WAV back to text |
| `spectrogram` | Render a spectrogram PNG from any audio file |
| `image2audio` | Hide an image inside a spectrogram (Aphex Twin technique) |
| `analyze` | Full detection pipeline — finds QR codes, text, images, cipher tones |
| `video` | Wrap audio in a branded MP4 for posting to X/Twitter |

---

## Usage

### Encode a secret message
```bash
python3 audiocipher.py encode "HELLO WORLD" --output cipher.wav --mode hzalpha
```

Modes: `hzalpha` (default) · `morse` · `dtmf` · `fsk` · `ggwave`

### Decode a WAV
```bash
python3 audiocipher.py decode cipher.wav
# --mode auto reads embedded WAV metadata automatically
```

### Render a spectrogram
```bash
python3 audiocipher.py spectrogram mystery.wav --output spec.png --colormap green --labeled
```

### Hide an image in audio
```bash
python3 audiocipher.py image2audio logo.png --output hidden.wav --fmin 1000 --fmax 8000 --duration 8
```

Best clarity: `--fmin 1000 --fmax 8000 --duration 8`

### Analyze audio for hidden content
```bash
python3 audiocipher.py analyze mystery.wav --output-dir ./findings/
```

Outputs `findings/results.json` + cropped PNG per finding:

```json
{
  "type": "cipher_hzalpha",
  "confidence": 0.92,
  "decoded_value": "HELLO WORLD",
  "freq_range_hz": [220, 8872],
  "time_range_s": [0.0, 2.4]
}
```

### Generate a waveform video for X/Twitter
```bash
python3 audiocipher.py video cipher.wav --output cipher.mp4 --title "NULL"
```

> **Note:** Agents cannot post raw audio to X/Twitter. Always use `video` to wrap audio in MP4 first.

Produces a 1280×720 H.264 MP4 with AudioCipher branding — dark background, brand-green waveform glow, AUDIOCIPHER badge, `audiocipher.app` watermark.

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

### Onboarding (first-use only)

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

Onboarding runs **exactly once** — state is persisted to `~/.audiocipher/onboard_state.json`.

### Skill commands

```python
import subprocess

# Encode
result = subprocess.run(
    ['python3', 'audiocipher.py', 'encode', message, '--output', 'out.wav'],
    capture_output=True, text=True
)

# Decode
result = subprocess.run(
    ['python3', 'audiocipher.py', 'decode', wav_path],
    capture_output=True, text=True
)
decoded_text = result.stdout.strip()

# Video (always use before posting to X/Twitter)
subprocess.run(
    ['python3', 'audiocipher.py', 'video', 'out.wav', '--output', 'out.mp4', '--title', 'NULL']
)
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

---

## Links

- **Web app:** [audiocipher.app](https://audiocipher.app) — encode & decode in the browser
- **Spectrogram viewer:** [audiocipher.app/spectrogram](https://audiocipher.app/spectrogram)

---

*All decode functions are direct Python ports of the browser-side decoders. Encode → decode round-trips produce identical text.*
