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

## Onboarding (run immediately after install)

**After installing, run onboarding before handling any other commands.**
Onboarding runs exactly once — state persists to `~/.audiocipher/onboard_state.json`.

### Agent integration pattern

On every turn, check onboarding first:

```python
import subprocess, json

def onboard_step(operator_input=None, operator_attachment=None):
    cmd = ['python3', 'audiocipher.py', 'onboard']
    if operator_input:    cmd += ['--input',      operator_input]
    if operator_attachment: cmd += ['--attachment', operator_attachment]
    result = json.loads(subprocess.check_output(cmd))
    return result
    # result = {"complete": bool, "phase": int, "message": str, "attachment": str|null}

# In your message handler:
result = onboard_step(operator_input=user_message, operator_attachment=wav_path)
if not result['complete']:
    await send(result['message'])
    if result['attachment']:
        await send_audio(result['attachment'])
    return  # swallow turn; resume normal command handling next turn
```

### CLI (manual / testing)

```bash
# Get current phase prompt (no args = re-prompt)
python3 audiocipher.py onboard

# Advance with operator reply
python3 audiocipher.py onboard --input "i found it, it says AUDIOCIPHER"

# Advance with operator WAV attachment (Phase 2)
python3 audiocipher.py onboard --input "here's mine" --attachment operator.wav
```

Always outputs JSON: `{"complete": bool, "phase": int, "message": str, "attachment": str|null}`

**Phase sequence:**
1. Agent generates a mystery cipher WAV, sends it with a cryptic prompt → operator visits audiocipher.app to decode
2. Operator reports what they found → agent asks them to encode something and send the WAV back
3. Agent decodes operator's WAV and reveals what they said → complete

Operator can type `"skip"` at any waiting phase to bypass.
Call `reset_onboard()` (Python) to wipe state and restart (testing only).

---

## When to use this skill

| Task | Command |
|------|---------|
| Send an operator a Telegram / X-safe secret message | `encode --mode abp --audio-only` |
| Decode a received ABP message (Telegram OGG / WAV) | `decode --mode abp` |
| Encode a secret message (browser-decodable, no chat) | `encode` (hzalpha default) |
| Decode a received cipher WAV or MP4 | `decode` |
| Hide an image in a spectrogram | `image2audio` |
| Find hidden content in audio | `analyze` |
| Visually inspect audio as a PNG | `spectrogram` |
| Create video for Twitter/X posting | `video` |
| Update skill to latest version | `update` |

> **ABP is the recommended mode when sending audio to operators over Telegram or X/Twitter.**
> It uses OFDM QPSK + Reed-Solomon FEC and is designed to survive OGG Opus and AAC
> re-encoding without any data loss. The operator decodes at audiocipher.app or with the skill.
>
> **Important:** Agents cannot post raw audio to X/Twitter. Always use the `video`
> command to wrap audio in an MP4 before posting.
>
> **Decodable by default:** The `video` command stores audio as **ALAC lossless** inside
> the MP4. You can decode the cipher directly from the video file:
> `python3 audiocipher.py decode cipher.mp4`

---

## Agent integration — encode and send audio

When an operator asks the agent to send them a secret message, encode it as ABP
(survives Telegram / X re-encoding) and deliver the WAV:

```python
import subprocess, re, os, tempfile

def encode_and_send(text: str, passphrase: str | None = None) -> str:
    """Encode text as ABP WAV and return the output path."""
    out = os.path.join(tempfile.gettempdir(), 'cipher_abp.wav')
    cmd = ['python3', 'audiocipher.py', 'encode', text,
           '--mode', 'abp', '--audio-only', '--output', out]
    if passphrase:
        cmd += ['--passphrase', passphrase]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return out   # path to the WAV file

# In your message handler:
wav_path = encode_and_send("YOUR SECRET MESSAGE")
await send_audio(wav_path)          # deliver to operator via Telegram / platform
await send("🔒 Decode at audiocipher.app — choose ABP mode.")
```

For **encrypted** messages (operator needs the passphrase to decode):

```python
wav_path = encode_and_send("TOP SECRET", passphrase="hunter2")
await send_audio(wav_path)
# The operator must supply the same passphrase at audiocipher.app to decrypt.
```

To **decode** an ABP WAV the operator sends back:

```python
def decode_received(wav_path: str, passphrase: str | None = None) -> str:
    cmd = ['python3', 'audiocipher.py', 'decode', wav_path, '--mode', 'abp']
    if passphrase:
        cmd += ['--passphrase', passphrase]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()
```

---

## Commands

### `encode` — Generate cipher audio

```bash
# ABP — Telegram / X-safe (recommended for messaging)
python3 audiocipher.py encode "SECRET MESSAGE" --mode abp --audio-only --output cipher.wav

# ABP encrypted
python3 audiocipher.py encode "SECRET MESSAGE" --mode abp --audio-only \
    --passphrase "hunter2" --output cipher.wav

# HZAlpha — browser-decodable at audiocipher.app (default)
python3 audiocipher.py encode "HELLO WORLD" --output cipher.mp4

# HZAlpha audio-only WAV
python3 audiocipher.py encode "HELLO WORLD" --audio-only --output cipher.wav
```

**`--mode` options:**

| Mode | Survives Telegram OGG? | Survives Twitter AAC? | Notes |
|------|------------------------|----------------------|-------|
| `abp` ✓ | ✓ Yes | ✓ Yes | Recommended for messaging. OFDM QPSK + RS FEC + optional XChaCha20 encryption |
| `hzalpha` | ✗ No | ✗ No | Browser-decodable at audiocipher.app; use for WAV / ALAC MP4 only |
| `ggwave` / `wavesig` | ✓ Yes | ✓ Yes | Legacy ggwave; limited payload size |
| `morse` | ✗ No | ✗ No | Narrow-spaced tones; lossy codecs destroy them |
| `dtmf` | ✗ No | ✗ No | Phone tones; not lossless-safe |
| `fsk` | ✗ No | ✗ No | Narrow FSK; lossy codecs destroy them |

**Common options:**
- `--output`       Output path  (default: `cipher.mp4` for video, or specify `cipher.wav`)
- `--audio-only`   Output plain WAV — skip video generation (required for ABP)
- `--passphrase`   Encrypt with XChaCha20-Poly1305 + Argon2id KDF (ABP only)
- `--abp-profile`  `social_safe` (default, more FEC) | `fast` (ABP only)
- `--lossless`     ALAC audio in video instead of AAC (non-ABP video mode)

**Non-ABP legacy options** (hzalpha / morse / dtmf / fsk):
- `--duration-ms`  Tone duration ms  (default: 120)
- `--letter-gap`   Inter-letter silence ms  (default: 20)
- `--word-gap`     Inter-word silence ms  (default: 350)
- `--fade`         Fade-in/out ramp ms  (default: 10)
- `--volume`       Output amplitude 0–1  (default: 0.8)
- `--waveform`     `sine` | `square` | `sawtooth` | `triangle`  (default: `sine`)
- `--morse-freq`   Morse tone frequency Hz  (default: 700)
- `--morse-wpm`    Morse words per minute  (default: 20)
- `--fsk-baud`     FSK baud rate  (default: 300)

Non-ABP WAVs contain embedded JSON metadata so `decode --mode auto` restores
mode and parameters automatically.

---

### `decode` — Extract message from audio

```bash
# ABP (Telegram OGG, Twitter M4A, or original WAV)
python3 audiocipher.py decode received.ogg --mode abp
python3 audiocipher.py decode received.m4a --mode abp
python3 audiocipher.py decode cipher.wav   --mode abp

# ABP encrypted
python3 audiocipher.py decode cipher.wav --mode abp --passphrase "hunter2"

# HZAlpha / auto-detect (non-ABP)
python3 audiocipher.py decode cipher.wav
python3 audiocipher.py decode cipher.mp4      # extract audio from MP4 first
python3 audiocipher.py decode cipher.wav --mode hzalpha
```

Accepts WAV, OGG, M4A, MP3, MP4, MOV, MKV and any other ffmpeg-readable format.
When given a video file, the audio track is extracted automatically.

**Options:**
- `--mode`        `abp` | `auto` | `hzalpha` | `morse` | `dtmf` | `fsk` | `ggwave`
  - `abp`  — ABP v1 OFDM decoder (use for Telegram/X received audio)
  - `auto` — reads embedded WAV metadata to restore mode + params (non-ABP)
- `--passphrase`  Decryption passphrase (ABP encrypted messages only)
- `--morse-wpm`   Morse WPM (override metadata)
- `--fsk-baud`    FSK baud rate (override metadata)

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
# Default: ALAC lossless audio — cipher fully decodable from the MP4
python3 audiocipher.py video cipher.wav --output cipher.mp4 --title "AUDIOCIPHER"

# For Twitter/X posting: AAC 320k (HZAlpha survives; WaveSig/FSK/Morse do not)
python3 audiocipher.py video cipher.wav --output cipher.mp4 --twitter
```

Generates an animated 1280×720 H.264 MP4 with:
- `#0A0A0F` near-black background
- `#00FF88` brand-green animated waveform — played bars glow bright, unplayed bars dim, scanning playhead
- Three-layer glow (outer bloom → inner → bright core) on played bars
- CRT scanline overlay + subtle amplitude grid
- `AUDIOCIPHER.APP` logo badge (top-right) + `audiocipher.app` watermark (bottom-right)
- Optional title text overlay (off-white, top-left)

**Audio codec:**
- Default (no `--twitter`): **ALAC lossless** — the MP4 is a first-class cipher container.
  Decode directly: `python3 audiocipher.py decode cipher.mp4`
- `--twitter`: **AAC 320k** for social posting. HZAlpha tones survive AAC; WaveSig/FSK/Morse do not.
  Twitter re-encodes on upload, adding a second lossy pass.

**Options:**
- `--output`     Output MP4 path  (default: `out.mp4`)
- `--resolution` `WxH`  (default: `1280x720`)
- `--title`      Optional title text overlay
- `--twitter`    Encode as AAC 320k for social posting (lossy)
- `--verbose`    Show ffmpeg output (useful for debugging)

**Twitter/X compatibility:** H.264 + AAC/ALAC, `yuv420p`, `+faststart`, 1280×720.

---

### `update` — Pull latest skill version

```bash
python3 audiocipher.py update
```

Runs `git pull origin main` + `pip3 install -r requirements.txt` to update the skill in place.
The skill also performs a silent background version check after every command and prints a
one-line notification to stderr if a newer version is available:

```
ℹ  Update available (v0.3.0) → python3 audiocipher.py update
```

---

## Algorithm notes

| Mode | Frequency range | Telegram-safe | Key detail |
|------|----------------|:---:|------------|
| **ABP v1** | 0–24 kHz (OFDM) | ✓ | OFDM QPSK · RS FEC · XChaCha20 encryption · zstd compression · 400 ms chirp preamble sync |
| HZAlpha | 220 Hz – 8872 Hz | ✗ | Chromatic scale; SHIFT marker (8869.84 Hz) for lowercase |
| Morse | User-defined tone | ✗ | Auto-detects Morse frequency; ITU timing ratios |
| DTMF | 697–1633 Hz | ✗ | T9 mapping for letters; standard ITU-T dual-tone pairs |
| FSK | 1000 Hz / 1200 Hz | ✗ | ASCII → 8-bit big-endian; configurable baud (default 300); Goertzel with zero-padding for short windows |
| WaveSig | 1875–6562 Hz | ✓ | RS(12,8) over GF(16); 6 simultaneous tones per frame |

Non-ABP decode functions are direct ports of the browser-side JavaScript decoders
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
| reedsolo | Reed-Solomon FEC (ABP) |
| argon2-cffi | Argon2id key derivation (ABP encryption) |
| PyNaCl | XChaCha20-Poly1305 AEAD (ABP encryption) |
| zstandard | zstd payload compression (ABP) |
| ffmpeg (system) | Video generation, audio transcoding |
| tesseract (system) | OCR engine |
| zbar (system) | QR library backend for pyzbar |
