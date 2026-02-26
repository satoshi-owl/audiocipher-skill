# audiocipher — AudioCipher Python CLI Skill

A standalone Python CLI for encoding/decoding hidden audio messages, generating
spectrogram images from audio, analyzing audio for hidden content, and producing
waveform videos for posting to X/Twitter.

---

## Setup

```bash
cd skill/
bash install.sh
```

Or manually:
```bash
pip3 install -r requirements.txt
# Also install system packages: ffmpeg, tesseract, zbar (see install.sh)
```

---

## Onboarding (first-use only)

**Before handling any skill command, check whether onboarding has completed:**

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
    return  # swallow this turn; resume normal command handling next turn
```

The flow runs **exactly once** — state is persisted to `~/.audiocipher/onboard_state.json`.
After `is_complete()` returns `True` it stays True across restarts and re-imports.

**Phase sequence:**
1. Agent generates a mystery cipher WAV, sends it with a cryptic prompt
2. Operator uploads to **audiocipher.app → Decode**, tells the agent what they found
3. Agent asks operator to encode something on the site and send the WAV back
4. Agent decodes operator's WAV and reveals what they said → setup complete

Operator can type `"skip"` at any waiting phase to bypass and jump to complete.
Call `reset_onboard()` to wipe state and restart (testing / re-installation only).

---

## When to use this skill

| Task | Command |
|------|---------|
| Encode a secret message as audio | `encode` |
| Decode a received cipher WAV | `decode` |
| Hide an image in a spectrogram | `image2audio` |
| Find hidden content in audio | `analyze` |
| Visually inspect audio as a PNG | `spectrogram` |
| Create video for Twitter/X posting | `video` |

> **Important:** Agents cannot post raw audio to X/Twitter. Always use the `video`
> command to wrap audio in an MP4 before posting.

---

## Commands

### `encode` — Generate cipher audio

```bash
python3 audiocipher.py encode "HELLO WORLD" --output cipher.wav --mode hzalpha
```

**Options:**
- `--mode`         `hzalpha` | `morse` | `dtmf` | `fsk` | `ggwave` | `custom`  (default: `hzalpha`)
- `--output`       Output WAV path  (default: `cipher.wav`)
- `--duration-ms`  Tone duration ms  (default: 120)
- `--letter-gap`   Inter-letter silence ms  (default: 20)
- `--word-gap`     Inter-word silence ms  (default: 350)
- `--fade`         Fade-in/out ramp ms  (default: 10)
- `--volume`       Output amplitude 0–1  (default: 0.8)
- `--waveform`     `sine` | `square` | `sawtooth` | `triangle`  (default: `sine`)
- `--morse-freq`   Morse tone frequency Hz  (default: 700)
- `--morse-wpm`    Morse words per minute  (default: 20)
- `--fsk-baud`     FSK baud rate  (default: 300)

The output WAV contains embedded JSON metadata so `decode --mode auto` can
automatically restore mode and parameters.

---

### `decode` — Extract message from audio

```bash
python3 audiocipher.py decode cipher.wav
# or with explicit mode:
python3 audiocipher.py decode cipher.wav --mode hzalpha
```

**Options:**
- `--mode`      `auto` | `hzalpha` | `morse` | `dtmf` | `fsk` | `ggwave` | `custom`
  - `auto` reads embedded WAV metadata to restore mode + params automatically
- `--morse-wpm` Morse WPM (override metadata)
- `--fsk-baud`  FSK baud rate (override metadata)

Prints decoded text to **stdout**.

---

### `image2audio` — Hide image in spectrogram

```bash
python3 audiocipher.py image2audio logo.png \
    --output hidden.wav --fmin 1000 --fmax 8000 --duration 8
```

The image will be visible when the WAV is opened in a spectrogram viewer
(Sonic Visualiser, audiocipher.app/spectrogram, or Audacity).

**Options:**
- `--fmin`      Lowest frequency Hz  (default: 200)
- `--fmax`      Highest frequency Hz  (default: 16000)
- `--duration`  Audio duration seconds  (default: 6)
- `--sr`        Sample rate  (default: 44100)
- `--amplitude` Output amplitude 0–1  (default: 0.8)
- `--invert`    Invert brightness (white → silence, black → loud)

**Best results for image clarity:**  `--fmin 1000 --fmax 8000 --duration 8`

---

### `analyze` — Find hidden content in audio

```bash
python3 audiocipher.py analyze mystery.wav --output-dir ./findings/
```

Runs a full detection pipeline:
1. QR / barcode detection (pyzbar)
2. OCR text extraction (pytesseract)
3. Structured image regions (OpenCV contours)
4. Cipher tone auto-detection (HZAlpha, DTMF, FSK)
5. Anomaly / entropy analysis (narrow tones, repeating patterns)

**Output:**
- `./findings/results.json` — all findings as JSON array
- `./findings/crop_NNN_<type>.png` — cropped region per finding (noise-stripped)

**Options:**
- `--output-dir` Output directory  (default: `./findings/`)
- `--no-qr`      Skip QR detection
- `--no-ocr`     Skip OCR
- `--no-cipher`  Skip cipher tone detection
- `--no-anomaly` Skip anomaly detection
- `--no-crop`    Do not save crop PNGs

**Finding JSON structure:**
```json
{
  "type": "qr_code|text|image|cipher_hzalpha|cipher_dtmf|cipher_fsk|anomaly",
  "confidence": 0.85,
  "crop_path": "./findings/crop_001_qr.png",
  "bounding_box": {"x": 100, "y": 50, "w": 200, "h": 150},
  "freq_range_hz": [2000.0, 8000.0],
  "time_range_s": [1.2, 4.5],
  "decoded_value": "HELLO WORLD",
  "notes": "pyzbar detected QRCODE in raw variant"
}
```

---

### `spectrogram` — Render spectrogram PNG from audio

```bash
python3 audiocipher.py spectrogram mystery.wav --output spec.png
python3 audiocipher.py spectrogram mystery.wav --output spec.png --colormap green --labeled
```

Renders a high-resolution spectrogram image from any audio file using STFT.
Use this to visually inspect audio before running `analyze`, or to verify
`image2audio` output.

**Options:**
- `--output`     Output PNG path  (default: `spectrogram.png`)
- `--colormap`   `green` | `inferno` | `viridis` | `amber` | `grayscale`  (default: `green`)
- `--fmin`       Min display frequency Hz  (default: 0)
- `--fmax`       Max display frequency Hz  (default: Nyquist)
- `--width`      Image width px  (default: 1200)
- `--height`     Image height px  (default: 600)
- `--fft-size`   FFT window size — larger = better freq resolution  (default: 4096)
- `--gain-db`    Brightness gain dB, positive = brighter  (default: 0)
- `--labeled`    Add frequency (Hz) and time (s) axis labels to the image

---

### `video` — Generate waveform video for X/Twitter

```bash
python3 audiocipher.py video cipher.wav --output cipher.mp4 --title "AUDIOCIPHER"
```

Generates a 1280×720 H.264 MP4 with:
- `#0A0A0F` near-black background
- `#00FF88` brand-green waveform with three-layer glow (outer bloom → inner → core)
- CRT scanline overlay + subtle amplitude grid
- `AUDIOCIPHER` logo badge (top-right) + `audiocipher.app` watermark (bottom-right)
- Optional title text overlay (off-white, top-left)

Waveform is rendered by Python (PIL + numpy) as a static frame, then muxed
with audio via ffmpeg `-loop 1 -tune stillimage`. No ffmpeg filter quirks.

**Options:**
- `--output`     Output MP4 path  (default: `out.mp4`)
- `--resolution` `WxH`  (default: `1280x720`)
- `--title`      Optional title text overlay
- `--verbose`    Show ffmpeg output (useful for debugging)

**Twitter/X compatibility:** H.264 + AAC, `yuv420p`, `+faststart`, 1280×720.

---

## Algorithm notes

| Mode | Frequency range | Key detail |
|------|----------------|------------|
| HZ Alpha | 220 Hz – 8872 Hz | Chromatic scale; SHIFT marker (8869.84 Hz) for lowercase |
| Morse | User-defined tone | Auto-detects Morse frequency; ITU timing ratios |
| DTMF | 697–1633 Hz | T9 mapping for letters; standard ITU-T dual-tone pairs |
| FSK | 1000 Hz / 1200 Hz | ASCII → 8-bit big-endian; configurable baud (default 300); Goertzel with zero-padding for short windows |
| WaveSig | 1875–6562 Hz | RS(12,8) over GF(16); 6 simultaneous tones per frame |

All decode functions are direct ports of the browser-side JavaScript decoders
in `app.html`. Encode → decode round-trips produce identical text.

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
