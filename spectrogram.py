"""
spectrogram.py — Image ↔ Audio spectrogram synthesis.

Functions:
  image_to_audio(image_path, ...) → np.ndarray (float32 PCM)
      Convert a greyscale image so it visually appears in the audio spectrogram
      (Aphex Twin / Mathematical Notation technique).

  audio_to_spectrogram_image(audio_path, ...) → PIL.Image
      Render a high-resolution spectrogram PNG from audio using STFT.
      Suitable for export or as input to analyzer.py.

Source reference:
  image_to_audio:  spectrogram.html i2aRender() lines 1178–1280
  audio_to_spec:   spectrogram.html _attachRealtimeScan() lines 929–1033
                   (adapted to offline batch processing with librosa)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from utils import read_wav, write_wav


# ─────────────────────────────────────────────────────────────────────────────
# Colour schemes for spectrogram rendering (match spectrogram.html COLOR_SCHEMES)
# Each returns (R, G, B) in 0–255 for a normalised value v ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def _color_green(v: float):
    """AudioCipher default: dark green → bright green → white."""
    if v < 0.5:
        g = int(v * 2 * 220)
        return (0, g, int(g * 0.4))
    else:
        w = int((v - 0.5) * 2 * 35)
        g = min(255, 220 + w)
        return (w * 3, g, int(g * 0.4) + w * 2)


def _color_inferno(v: float):
    """Approximation of matplotlib's inferno (black→purple→orange→yellow)."""
    stops = [
        (0.0,  (0,   0,   4)),
        (0.25, (87,  16,  110)),
        (0.5,  (188, 55,  84)),
        (0.75, (249, 142, 9)),
        (1.0,  (252, 255, 164)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= v <= t1:
            f = (v - t0) / (t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return (r, g, b)
    return (252, 255, 164)


def _color_viridis(v: float):
    stops = [
        (0.0,  (68,  1,   84)),
        (0.25, (59,  82,  139)),
        (0.5,  (33,  145, 140)),
        (0.75, (94,  201, 98)),
        (1.0,  (253, 231, 37)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= v <= t1:
            f = (v - t0) / (t1 - t0)
            return (
                int(c0[0] + f * (c1[0] - c0[0])),
                int(c0[1] + f * (c1[1] - c0[1])),
                int(c0[2] + f * (c1[2] - c0[2])),
            )
    return (253, 231, 37)


def _color_amber(v: float):
    r = min(255, int(v * 2 * 255)) if v < 0.5 else 255
    g = min(255, int((v - 0.5) * 2 * 180)) if v >= 0.5 else 0
    b = 0
    return (r, g, b)


def _color_grayscale(v: float):
    c = int(v * 255)
    return (c, c, c)


COLOR_SCHEMES = {
    'green':     _color_green,
    'inferno':   _color_inferno,
    'viridis':   _color_viridis,
    'amber':     _color_amber,
    'grayscale': _color_grayscale,
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE → AUDIO  (ported from i2aRender, spectrogram.html 1178–1280)
# ═══════════════════════════════════════════════════════════════════════════════

def image_to_audio(
    image_path: str,
    fmin:      float = 200.0,
    fmax:      float = 16000.0,
    duration:  float = 6.0,
    sr:        int   = 44100,
    amplitude: float = 0.8,
    invert:    bool  = False,
) -> np.ndarray:
    """
    Convert an image to audio so the image appears in the spectrogram.

    Maps pixel brightness → oscillator amplitude for each frequency band.
    Top of image = highest frequency (bin 0 = fmax), matching spectrogram
    display convention in spectrogram.html.

    Args:
        image_path: Path to image (PNG / JPG / any Pillow-readable format)
        fmin:       Lowest frequency to use (Hz)
        fmax:       Highest frequency to use (Hz)
        duration:   Audio duration in seconds
        sr:         Output sample rate
        amplitude:  Master output amplitude 0–1
        invert:     If True, invert pixel brightness (white → silence)

    Returns:
        np.ndarray, float32, mono, tanh soft-limited

    Source: spectrogram.html i2aRender() lines 1194–1253
    """
    from PIL import Image  # type: ignore

    img    = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size

    # Cap at 512 bins/cols for performance (matches JS: Math.min(..., 512))
    num_bins = min(img_h, 512)  # frequency axis
    num_cols = min(img_w, 512)  # time axis

    # Build brightness matrix [num_bins, num_cols] using ITU-R BT.601 luma
    pixels = np.array(img, dtype=np.float32)  # [H, W, 3]
    luma   = (0.299 * pixels[:, :, 0]
            + 0.587 * pixels[:, :, 1]
            + 0.114 * pixels[:, :, 2]) / 255.0  # [H, W]

    # Resample to [num_bins, num_cols] via nearest-neighbour (matches JS floor)
    row_idx = (np.arange(num_bins) * img_h / num_bins).astype(int)
    col_idx = (np.arange(num_cols) * img_w / num_cols).astype(int)
    brightness = luma[np.ix_(row_idx, col_idx)]  # [num_bins, num_cols]

    if invert:
        brightness = 1.0 - brightness

    # Frequency per bin: bin 0 = top of image = fmax, last bin = fmin
    # freq = fmax - (fmax - fmin) * bin / (num_bins - 1)
    freqs = fmax - (fmax - fmin) * np.arange(num_bins) / max(num_bins - 1, 1)

    total_samples = int(np.ceil(duration * sr))
    audio = np.zeros(total_samples, dtype=np.float64)

    # Column edges in sample space
    col_edges = np.linspace(0, total_samples, num_cols + 1).astype(int)

    # Master gain: volume * 0.9 / sqrt(num_bins), matching JS masterGain
    master = float(amplitude) * 0.9 / np.sqrt(num_bins)

    t = np.arange(total_samples, dtype=np.float64) / sr

    for bin_idx in range(num_bins):
        freq = float(freqs[bin_idx])
        if freq < 20.0 or freq > sr / 2.0:
            continue

        sine = np.sin(2.0 * np.pi * freq * t)

        # Build gain envelope: step function per time column
        env = np.zeros(total_samples, dtype=np.float64)
        for col_idx in range(num_cols):
            amp = float(brightness[bin_idx, col_idx])
            # Threshold 0.04 (< 4 % brightness → silence)  — matches JS
            if amp < 0.04:
                amp = 0.0
            env[col_edges[col_idx]:col_edges[col_idx + 1]] = amp

        audio += master * sine * env

    # tanh soft-limiting (matches JS: Math.tanh(samples[i]))
    audio = np.tanh(audio)
    return audio.astype(np.float32)


def write_image_audio(output_path: str, audio: np.ndarray, sr: int = 44100):
    """Write image-audio to WAV (no cipher metadata)."""
    write_wav(output_path, audio, sr=sr, mode=None)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO → SPECTROGRAM IMAGE
# ═══════════════════════════════════════════════════════════════════════════════

def audio_to_spectrogram_image(
    audio_path: str,
    fft_size:   int   = 4096,
    fmin:       float = 0.0,
    fmax:       float | None = None,
    colormap:   str   = 'green',
    width:      int   = 1200,
    height:     int   = 600,
    gain_db:    float = 0.0,
    log_scale:  bool  = False,
) -> 'PIL.Image.Image':  # type: ignore[name-defined]
    """
    .. deprecated::
        ``audio_to_spectrogram_image()`` contained a 720,000-iteration pure-Python
        pixel loop (``for py in range(height): for px in range(width)``), making it
        ~200× slower than the equivalent numpy path.

        This function is now a thin alias for :func:`audio_to_spectrogram_fast`,
        which uses a LUT colormap + PIL LANCZOS resize entirely at C speed.
        The ``log_scale`` parameter is silently ignored (not supported by the fast
        path); use a log-spaced ``fmin``/``fmax`` range instead.

    Render a high-resolution spectrogram PNG from audio using STFT.

    Args:
        audio_path: Path to audio file
        fft_size:   FFT window size (larger = better freq resolution)
        fmin:       Minimum display frequency (Hz)
        fmax:       Maximum display frequency (Hz); defaults to Nyquist
        colormap:   'green' | 'inferno' | 'viridis' | 'amber' | 'grayscale'
        width:      Output image width in pixels
        height:     Output image height in pixels
        gain_db:    Gain in dB applied to magnitude (positive = brighter)
        log_scale:  *Ignored* — kept for API compatibility only.

    Returns:
        PIL.Image in RGB mode
    """
    import warnings
    warnings.warn(
        "audio_to_spectrogram_image() is deprecated and was replaced with "
        "audio_to_spectrogram_fast() which uses a LUT colormap + PIL LANCZOS "
        "resize at C speed (~200× faster). The log_scale parameter is ignored.",
        DeprecationWarning,
        stacklevel=2,
    )
    return audio_to_spectrogram_fast(
        audio_path,
        fft_size=fft_size,
        fmin=fmin,
        fmax=fmax,
        colormap=colormap,
        width=width,
        height=height,
        gain_db=gain_db,
    )


def audio_to_spectrogram_fast(
    audio_path: str,
    fft_size:   int   = 4096,
    fmin:       float = 0.0,
    fmax:       float | None = None,
    colormap:   str   = 'green',
    width:      int   = 1200,
    height:     int   = 600,
    gain_db:    float = 0.0,
) -> 'PIL.Image.Image':  # type: ignore[name-defined]
    """
    Faster spectrogram rendering using numpy resize instead of per-pixel loop.
    Recommended for analyzer.py which needs large spectrograms quickly.

    Returns PIL.Image in RGB mode.
    """
    from PIL import Image  # type: ignore

    samples, sr = read_wav(audio_path)
    if fmax is None:
        fmax = float(sr // 2)

    try:
        import librosa  # type: ignore
        hop_length = fft_size // 4
        S = np.abs(librosa.stft(samples, n_fft=fft_size, hop_length=hop_length))
        freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=fft_size)
    except ImportError:
        from scipy.signal import spectrogram as _spect  # type: ignore
        freqs_stft, _, S = _spect(
            samples, fs=sr, nperseg=fft_size,
            noverlap=fft_size - fft_size // 4, scaling='spectrum',
        )
        S = np.sqrt(np.abs(S))

    S_db = 20.0 * np.log10(np.maximum(S, 1e-9)) + gain_db
    s_min, s_max = S_db.min(), S_db.max()
    if s_max > s_min:
        S_norm = np.clip((S_db - s_min) / (s_max - s_min), 0.0, 1.0).astype(np.float32)
    else:
        S_norm = np.zeros_like(S_db, dtype=np.float32)

    # Frequency crop
    bin_lo = int(np.searchsorted(freqs_stft, fmin))
    bin_hi = int(np.searchsorted(freqs_stft, fmax, side='right'))
    S_crop = S_norm[bin_lo:bin_hi, :]  # [freq_bins, n_frames]

    if S_crop.shape[0] < 1 or S_crop.shape[1] < 1:
        return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))

    # Flip so low freq at bottom (for PIL row 0 = top → high freq display)
    S_crop = S_crop[::-1, :]  # high freq at row 0

    # Resize to target [height, width] using PIL (fast C implementation)
    mag_img = Image.fromarray((S_crop * 255).astype(np.uint8), mode='L')
    mag_img = mag_img.resize((width, height), Image.LANCZOS)
    mag_norm = np.array(mag_img, dtype=np.float32) / 255.0  # [height, width]

    # Apply colormap using numpy vectorisation
    color_fn = COLOR_SCHEMES.get(colormap, _color_green)
    # Build LUT (256 entries)
    lut_r = np.zeros(256, dtype=np.uint8)
    lut_g = np.zeros(256, dtype=np.uint8)
    lut_b = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        r, g, b = color_fn(i / 255.0)
        lut_r[i], lut_g[i], lut_b[i] = r, g, b

    idx = (mag_norm * 255).astype(np.uint8)
    rgb = np.stack([lut_r[idx], lut_g[idx], lut_b[idx]], axis=-1)
    return Image.fromarray(rgb, 'RGB')


def save_spectrogram(
    audio_path: str,
    output_path: str,
    fft_size:    int   = 4096,
    fmin:        float = 0.0,
    fmax:        float | None = None,
    colormap:    str   = 'green',
    width:       int   = 1200,
    height:      int   = 600,
    gain_db:     float = 0.0,
    labeled:     bool  = False,
):
    """
    Render and save a spectrogram PNG.

    Args:
        audio_path:  Source audio
        output_path: Destination PNG path
        labeled:     If True, add frequency and time axis labels
        (remaining args: see audio_to_spectrogram_fast)
    """
    img = audio_to_spectrogram_fast(
        audio_path, fft_size=fft_size, fmin=fmin, fmax=fmax,
        colormap=colormap, width=width, height=height, gain_db=gain_db,
    )

    if labeled:
        from PIL import ImageDraw, ImageFont  # type: ignore
        PAD_L, PAD_B, PAD_T, PAD_R = 56, 28, 12, 12
        samples, sr = read_wav(audio_path)
        duration = len(samples) / sr
        fmax_eff = fmax if fmax is not None else sr / 2.0

        W = img.width + PAD_L + PAD_R
        H = img.height + PAD_T + PAD_B
        canvas = Image.new('RGB', (W, H), color=(10, 10, 15))
        canvas.paste(img, (PAD_L, PAD_T))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype('/System/Library/Fonts/Courier.dfont', 11)
        except Exception:
            font = ImageFont.load_default()

        # Freq axis labels (5 ticks, left side)
        for tick in range(6):
            freq = fmax_eff * tick / 5.0
            y    = PAD_T + img.height - int(tick * img.height / 5)
            label = f'{freq/1000:.1f}k Hz' if freq >= 1000 else f'{freq:.0f} Hz'
            draw.text((PAD_L - 4, y + 2), label, fill=(180, 180, 200), font=font, anchor='rm')
            draw.line([(PAD_L, y), (W - PAD_R, y)], fill=(255, 255, 255, 20))

        # Time axis labels (5 ticks, bottom)
        for tick in range(5):
            x = PAD_L + int(tick * img.width / 4)
            t_label = f'{duration * tick / 4:.1f}s'
            draw.text((x, H - 4), t_label, fill=(180, 180, 200), font=font, anchor='mb')

        # Watermark
        draw.text((W - PAD_R - 2, H - 4), 'audiocipher.app',
                  fill=(0, 255, 136, 64), font=font, anchor='rb')
        img = canvas

    img.save(output_path, 'PNG')
    return output_path
