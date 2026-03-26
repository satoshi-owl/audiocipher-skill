"""
video.py — AudioCipher-branded scrolling waveform video generation.

Generates a 1280×720 (or custom) MP4 with:
  - Signal Mode brand theme (#0A0A0F dark bg, #00FF88 green waveform)
    Matches the AudioCipher website Signal Mode aesthetic exactly.
  - SCROLLING waveform: the audio flows left past a fixed centre playhead
    · Left half  (played)    — #00FF88 bright green with three-layer glow, edge-tapered
    · Right half (upcoming)  — ghost/barely-visible dim green
    · Fixed centre playhead  — #00FF88 accent green with glow
    · Visible window ≈ 4 s of audio (zoom-in with --window-seconds)
  - Subtle amplitude grid (±25 / ±50 / ±75 % near-invisible green tint)
  - CRT scanline overlay (every 3rd row dimmed 8 % — subtle)
  - Accent green progress bar (3 px, very bottom of frame)
  - Optional title text (top-left, white, large monospace)
  - AUDIOCIPHER outlined pill badge (top-right, white text + green dot)
  - audiocipher.app watermark (bottom-right, dim green 18% opacity)
  - H.264 video + AAC audio (MP4 is fully decodable);
    add --twitter for AAC 320k social posting.

Rendering: frames piped to ffmpeg via stdin (rawvideo RGB24).
  - Pre-computes a normalised per-pixel amplitude array once.
  - Per-frame: vectorised numpy mask operations (no Python per-pixel loops).
  - No temp PNGs — memory-efficient streaming.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Brand palette — Signal Mode (matches audiocipher.app website exactly)
# ─────────────────────────────────────────────────────────────────────────────
_BG          = (10,  10,  15)   # #0A0A0F — Signal Mode dark background
_WAVE_CORE   = (0,  255, 136)   # #00FF88 — Signal Mode accent green
_WAVE_INNER  = (0,  178,  95)   # mid green — inner glow layer
_WAVE_OUTER  = (0,   80,  43)   # dark green — outer diffuse glow
_WAVE_DIM    = (0,   38,  21)   # barely-visible upcoming green
_GRID_LINE   = (0,   25,  14)   # amplitude grid — near-invisible green tint
_WHITE       = (200, 200, 220)  # slightly blue-white — text / badge label
_BADGE_BG    = (17,  17,  24)   # #111118 — badge background (card bg)
_PLAYHEAD    = (0,  255, 136)   # #00FF88 — playhead matches accent green
_GREEN_DIM   = (0,  100,  54)   # dim green — watermark / subtle UI elements

FPS            = 30     # output frame rate
WINDOW_SECONDS = 4.0    # seconds of audio visible in the scrolling window


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg availability
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    return shutil.which('ffmpeg') is not None


def check_or_install_ffmpeg() -> bool:
    if check_ffmpeg():
        return True
    system = platform.system()
    try:
        if system == 'Darwin':
            print('→ ffmpeg not found. Installing via Homebrew…')
            subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        elif system == 'Linux':
            print('→ ffmpeg not found. Installing via apt…')
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
        else:
            print(f'⚠ ffmpeg not found. Install from https://ffmpeg.org', file=sys.stderr)
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'⚠ ffmpeg install failed: {e}', file=sys.stderr)
        return False
    return check_ffmpeg()


# ─────────────────────────────────────────────────────────────────────────────
# Font helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int):
    from PIL import ImageFont
    candidates = [
        '/System/Library/Fonts/Courier.dfont',
        '/System/Library/Fonts/SFNSMono.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# Logo badge
# ─────────────────────────────────────────────────────────────────────────────

def _draw_logo_badge(draw, right_x: int, top_y: int, font_sm) -> None:
    """Outlined pill badge — 'AUDIOCIPHER' with accent green status dot."""
    BADGE_W, BADGE_H, RADIUS = 210, 28, 6
    bx = right_x - BADGE_W
    by = top_y
    # Filled background (very close to page bg — creates the "inset" look)
    try:
        draw.rounded_rectangle([(bx, by), (bx + BADGE_W, by + BADGE_H)],
                               radius=RADIUS, fill=_BADGE_BG)
    except AttributeError:
        r2 = RADIUS * 2
        draw.rectangle([(bx + RADIUS, by), (bx + BADGE_W - RADIUS, by + BADGE_H)], fill=_BADGE_BG)
        draw.rectangle([(bx, by + RADIUS), (bx + BADGE_W, by + BADGE_H - RADIUS)], fill=_BADGE_BG)
        for cx, cy in [(bx, by), (bx + BADGE_W - r2, by),
                       (bx, by + BADGE_H - r2), (bx + BADGE_W - r2, by + BADGE_H - r2)]:
            draw.ellipse([(cx, cy), (cx + r2, cy + r2)], fill=_BADGE_BG)
    # Outline — warm neutral, 18% opacity (simulated: blend toward bg)
    _BORDER = tuple(int(_BG[c] * 0.72 + _WHITE[c] * 0.28) for c in range(3))
    try:
        draw.rounded_rectangle([(bx, by), (bx + BADGE_W, by + BADGE_H)],
                               radius=RADIUS, outline=_BORDER, width=1)
    except TypeError:
        pass
    # Green status dot
    DOT_R  = 3
    dot_cx = bx + 16
    dot_cy = by + BADGE_H // 2
    draw.ellipse([(dot_cx - DOT_R, dot_cy - DOT_R), (dot_cx + DOT_R, dot_cy + DOT_R)],
                 fill=_WAVE_CORE)
    # Label — parchment text, 75% opacity (simulated)
    _LABEL = tuple(int(_BG[c] * 0.25 + _WHITE[c] * 0.75) for c in range(3))
    draw.text((bx + 26, by + BADGE_H // 2 - 7), 'AUDIOCIPHER.APP', fill=_LABEL, font=font_sm)


# ─────────────────────────────────────────────────────────────────────────────
# CRT scanlines
# ─────────────────────────────────────────────────────────────────────────────

def _make_scanline_mult(H: int) -> np.ndarray:
    """Float32 (H, 1, 1) multiplier: 0.92 every 3rd row (subtle), else 1.0."""
    m = np.ones((H, 1, 1), dtype=np.float32)
    m[::3] = 0.92
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_amps_scrolling(
    samples:       np.ndarray,
    sr:            int,
    WAVE_W:        int,
    window_seconds: float = WINDOW_SECONDS,
) -> tuple[np.ndarray, int, int]:
    """
    Build a zero-padded normalised amplitude array for scrolling playback.

    Layout of returned padded array (length = 2*WAVE_W + n_pixels):
      [0 .. WAVE_W-1]                 — left  zero-pad
      [WAVE_W .. WAVE_W+n_pixels-1]   — actual per-pixel amplitudes
      [WAVE_W+n_pixels .. end]        — right zero-pad

    Returns:
        bar_amps_padded   float32 (2*WAVE_W + n_pixels,)
        samples_per_px    int — audio samples per pixel column
        n_pixels          int — number of actual amplitude pixels
    """
    samples_per_px = max(1, int(window_seconds * sr / WAVE_W))
    n = len(samples)
    n_pixels = max(1, (n + samples_per_px - 1) // samples_per_px)

    raw = np.zeros(n_pixels, dtype=np.float32)
    for i in range(n_pixels):
        s0 = i * samples_per_px
        s1 = min(n, s0 + samples_per_px)
        if s0 < n:
            seg = samples[s0:s1]
            raw[i] = float(np.max(np.abs(seg.astype(np.float64))))

    peak = raw.max()
    if peak > 0:
        raw /= peak

    padded = np.zeros(2 * WAVE_W + n_pixels, dtype=np.float32)
    padded[WAVE_W: WAVE_W + n_pixels] = raw
    return padded, samples_per_px, n_pixels


def _precompute_ui(
    W:        int,
    H:        int,
    MARGIN_X: int,
    title:    str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render badge / title / watermark once on black canvas."""
    from PIL import Image, ImageDraw
    ui_arr  = np.zeros((H, W, 3), dtype=np.uint8)
    ui_img  = Image.fromarray(ui_arr, 'RGB')
    ui_draw = ImageDraw.Draw(ui_img)
    font_title = _load_font(30)
    font_sm    = _load_font(13)

    if title:
        ui_draw.text((MARGIN_X, 30), title.upper(), fill=_WHITE, font=font_title)

    _draw_logo_badge(ui_draw, right_x=W - MARGIN_X, top_y=28, font_sm=font_sm)

    wm_text = 'audiocipher.app'
    wm_x    = W - MARGIN_X - len(wm_text) * 7
    ui_draw.text((max(MARGIN_X, wm_x), H - 22), wm_text, fill=_GREEN_DIM, font=font_sm)

    ui_arr = np.array(ui_img)
    return ui_arr, ui_arr.any(axis=2)


def _make_fade_curve(WAVE_W: int) -> np.ndarray:
    """
    Raised-cosine taper (WAVE_W,) — matches JS pow(t, 0.65) cosine envelope.
    Fades amplitude smoothly to ~0 at both edges, full strength in the centre.
    """
    xs    = np.linspace(-1.0, 1.0, WAVE_W, dtype=np.float32)
    t     = np.abs(xs) ** 0.65
    return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)


def _smooth_amps(amps: np.ndarray, sigma: int = 4) -> np.ndarray:
    """
    Pure-numpy gaussian smoothing — no scipy dependency.
    Converts blocky per-pixel steps into a smooth continuous curve.
    """
    ks     = sigma * 6 + 1                                 # kernel size (odd)
    xs     = np.arange(ks, dtype=np.float32) - ks // 2
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    pad    = ks // 2
    padded = np.pad(amps, pad, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(amps)]


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame renderer (fully vectorised — no Python per-pixel loops)
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame_scrolling(
    bar_amps_padded:  np.ndarray,
    amp_offset:       int,
    fade_curve:       np.ndarray,
    ui_arr:           np.ndarray,
    ui_mask:          np.ndarray,
    scanline_mult:    np.ndarray,
    dist_from_center: np.ndarray,
    played_col:       np.ndarray,   # kept for signature compat (unused — gradient replaces it)
    upcoming_col:     np.ndarray,   # kept for signature compat
    MARGIN_X:         int,
    WAVE_W:           int,
    WAVE_CENTER:      int,
    WAVE_MAX_H:       int,
    H:                int,
    W:                int,
    progress:         float = 0.0,
) -> np.ndarray:
    """
    Render one smooth scrolling-waveform frame.

    Visual layers:
      1. BG fill
      2. Smooth gaussian-blurred waveform polygon (cream top/ghost upcoming)
      3. Vertical gradient overlay (dense at peak → transparent at centre)
      4. Playhead (green, 5-px glow)
      5. Progress bar
      6. CRT scanlines
      7. UI chrome (badge / title / watermark)
    """
    from PIL import Image, ImageDraw

    # ── 1. Background ──────────────────────────────────────────────────────────
    arr = np.full((H, W, 3), _BG, dtype=np.uint8)

    # ── 2. Smooth amplitude window ─────────────────────────────────────────────
    raw_amps  = bar_amps_padded[amp_offset: amp_offset + WAVE_W].copy()
    tapered   = raw_amps * fade_curve                            # cosine-tapered
    smoothed  = _smooth_amps(tapered, sigma=5)                   # gaussian smooth
    smoothed  = np.clip(smoothed, 0.0, 1.0)
    heights   = (smoothed * WAVE_MAX_H).astype(int)              # (WAVE_W,)

    # ── 3. Build waveform polygon and paint it via PIL ─────────────────────────
    img  = Image.fromarray(arr, 'RGB')
    draw = ImageDraw.Draw(img)

    # Centre baseline (barely visible)
    draw.line([(MARGIN_X, WAVE_CENTER), (MARGIN_X + WAVE_W, WAVE_CENTER)],
              fill=_WAVE_OUTER, width=1)

    # Build polygon (top edge left→right, bottom edge right→left)
    xs        = list(range(MARGIN_X, MARGIN_X + WAVE_W))
    top_pts   = [(x, WAVE_CENTER - h) for x, h in zip(xs, heights)]
    bot_pts   = [(x, WAVE_CENTER + h) for x, h in zip(xs, heights)]
    polygon   = top_pts + list(reversed(bot_pts))

    if len(polygon) >= 3:
        # Glow pass — slightly expanded polygon, dim colour
        glow_poly = [(x, WAVE_CENTER - max(0, h - 1) - 4)
                     for x, h in zip(xs, heights)] + \
                    list(reversed([(x, WAVE_CENTER + max(0, h - 1) + 4)
                                   for x, h in zip(xs, heights)]))
        draw.polygon(glow_poly, fill=tuple(
            int(_BG[c] * 0.35 + _WAVE_OUTER[c] * 0.65) for c in range(3)))
        # Main polygon
        draw.polygon(polygon, fill=_WAVE_CORE)

    arr = np.array(img)

    # ── 4. Horizontal played/upcoming gradient overlay ─────────────────────────
    # Left of playhead → full brightness; right → ghost
    # Build a 1-D alpha curve (WAVE_W,): 1.0 played, 0.0 upcoming
    play_px     = int(progress * WAVE_W)          # playhead column within wave area
    alpha_h     = np.zeros(WAVE_W, dtype=np.float32)
    if play_px > 0:
        alpha_h[:play_px] = 1.0
    # Soften the transition ±4 px
    blend_w = max(1, min(8, WAVE_W // 10))
    for k in range(blend_w):
        col_idx = play_px - blend_w + k
        if 0 <= col_idx < WAVE_W:
            alpha_h[col_idx] = k / blend_w

    # For each column apply: played → keep WAVE_CORE; upcoming → blend toward WAVE_DIM
    # wv view: (H, WAVE_W, 3)
    wv   = arr[:, MARGIN_X:MARGIN_X + WAVE_W].astype(np.float32)
    # alpha_h: (1, WAVE_W, 1) broadcast
    a    = alpha_h[np.newaxis, :, np.newaxis]
    dim  = np.array(_WAVE_DIM, dtype=np.float32)
    # Blend: result = a * played_pixel + (1-a) * dim_pixel
    # Only apply where pixel is not background
    bg_f = np.array(_BG, dtype=np.float32)
    is_wave = np.any(wv != bg_f, axis=2)            # (H, WAVE_W)
    played_wv = wv * a + np.full_like(wv, dim) * (1 - a)
    arr[:, MARGIN_X:MARGIN_X + WAVE_W] = np.where(
        is_wave[:, :, np.newaxis], played_wv.clip(0, 255).astype(np.uint8), arr[:, MARGIN_X:MARGIN_X + WAVE_W]
    )

    # ── 5. Vertical fade — dense at peak, wispy near the centre line ───────────
    # t[row]: 0 at WAVE_CENTER, 1 at WAVE_CENTER ± WAVE_MAX_H
    ys     = np.arange(H, dtype=np.float32)
    t_vert = np.clip(np.abs(ys - WAVE_CENTER) / max(1, WAVE_MAX_H * 0.85), 0, 1)
    # Where t_vert is low (near centre): blend toward BG
    # effect: opacity_multiplier(row) = t_vert.  Apply to wave pixels only
    t_col   = t_vert[:, np.newaxis, np.newaxis]           # (H, 1, 1)
    wave_px = arr[:, MARGIN_X:MARGIN_X + WAVE_W].astype(np.float32)
    bg_col  = np.array(_BG, dtype=np.float32)
    faded   = (wave_px * t_col + bg_col * (1 - t_col)).clip(0, 255).astype(np.uint8)
    # Apply only to non-background pixels
    is_w2   = np.any(wave_px.astype(np.uint8) != np.array(_BG, dtype=np.uint8), axis=2)
    arr[:, MARGIN_X:MARGIN_X + WAVE_W] = np.where(
        is_w2[:, :, np.newaxis], faded, arr[:, MARGIN_X:MARGIN_X + WAVE_W]
    )

    # ── 6. Playhead — accent green, 5-px glow envelope ────────────────────────
    ctr        = MARGIN_X + WAVE_W // 2
    row_start  = int(H * 0.05)
    row_end    = int(H * 0.95)
    glow_steps = [(-3, 0.10), (-2, 0.25), (-1, 0.50), (0, 1.0), (1, 0.50), (2, 0.25), (3, 0.10)]
    for dx, alpha in glow_steps:
        gx = ctr + dx
        if 0 <= gx < W:
            col = tuple(int(_BG[c] * (1 - alpha) + _PLAYHEAD[c] * alpha) for c in range(3))
            arr[row_start:row_end, gx] = col

    # Diamond marker at waveform centre
    d_size = 9
    for dy in range(-d_size, d_size + 1):
        dx_max = d_size - abs(dy)
        for dx in range(-dx_max, dx_max + 1):
            r, c2 = WAVE_CENTER + dy, ctr + dx
            if 0 <= r < H and 0 <= c2 < W:
                # blend toward white at centre of diamond
                t_d = 1.0 - (abs(dx) + abs(dy)) / (d_size + 1)
                arr[r, c2] = tuple(
                    int(_BG[c] * (1 - t_d) + _PLAYHEAD[c] * t_d) for c in range(3))

    # ── 7. Progress bar — 4 px at very bottom ─────────────────────────────────
    track_col = tuple(int(_BG[c] * 0.7 + _WAVE_CORE[c] * 0.06) for c in range(3))
    arr[H - 4:H, :]            = track_col
    fill_w = max(0, min(W, int(W * progress)))
    if fill_w > 0:
        arr[H - 4:H, :fill_w]  = _PLAYHEAD

    # ── 8. CRT scanlines ──────────────────────────────────────────────────────
    arr = (arr.astype(np.float32) * scanline_mult).clip(0, 255).astype(np.uint8)

    # ── 9. UI chrome ──────────────────────────────────────────────────────────
    arr[ui_mask] = ui_arr[ui_mask]

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_video(
    audio_path:     str,
    output_path:    str  = 'out.mp4',
    style:          str  = 'null',
    resolution:     str  = '1280x720',
    title:          str | None = None,
    twitter:        bool = False,   # True → AAC 320k; False → ALAC lossless (default)
    audio_bitrate:  str  = '320k',  # only used when twitter=True
    video_preset:   str  = 'fast',
    window_seconds: float = WINDOW_SECONDS,
    verbose:        bool = False,
) -> str:
    """
    Generate an animated AudioCipher waveform video from an audio file.

    The waveform scrolls left in real time: the centre playhead always marks
    "now", bars to the left (played) glow bright green with a three-layer bloom,
    bars to the right (upcoming) are dim green. Bars taper toward the edges of
    the visible window for a natural fade effect.

    Args:
        audio_path:     Input audio (WAV / MP3 / any ffmpeg-readable format)
        output_path:    Output MP4 path  (default: out.mp4)
        style:          Reserved for future visual styles
        resolution:     Output resolution as 'WxH'  (default: 1280x720)
        title:          Title text displayed top-left (e.g. "NULL")
        twitter:        If True, encode audio as AAC 320k for Twitter/X posting.
                        Default False = ALAC lossless — MP4 stays fully decodable.
        audio_bitrate:  AAC bitrate used only when twitter=True  (default: 320k)
        video_preset:   libx264 preset: ultrafast / fast / medium
        window_seconds: Seconds of audio visible in the scrolling window (default: 4)
        verbose:        Show ffmpeg stderr

    Returns:
        Absolute path to the generated MP4.

    Cipher mode compatibility with compressed audio:
        HZAlpha  — survives AAC, Opus, and Telegram/Twitter re-encoding. ✓
        WaveSig  — survives AAC/Opus (100 Hz bin spacing since v0.2.3). ✓
        FSK/Morse — NOT safe with lossy codecs (auto-transcoded to HZAlpha). ✗
    """
    if not check_ffmpeg():
        raise RuntimeError('ffmpeg not available. Run check_or_install_ffmpeg().')

    audio_path  = str(Path(audio_path).resolve())
    output_path = str(Path(output_path).resolve())

    try:
        W, H = map(int, resolution.lower().split('x'))
    except (ValueError, AttributeError):
        W, H = 1280, 720

    MARGIN_X    = 56
    WAVE_W      = W - 2 * MARGIN_X
    WAVE_CENTER = H // 2
    WAVE_MAX_H  = int(H * 0.30)

    # ── Load audio ────────────────────────────────────────────────────────────
    try:
        from utils import read_wav  # type: ignore
        samples, sr = read_wav(audio_path)
    except Exception:
        try:
            import soundfile as sf  # type: ignore
            samples, sr = sf.read(audio_path, dtype='float32', always_2d=False)
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
        except Exception as exc:
            raise RuntimeError(f'Could not read audio: {exc}') from exc

    # ── Pre-roll: prepend silence to absorb AAC/Opus priming delay ─────────────
    # AAC encoders have a ~46ms priming delay; two codec hops (e.g. Telegram
    # then Twitter) can trim the first ~100ms of audio.  A 400ms silent pre-roll
    # ensures the first cipher tone is always fully preserved.
    PRE_ROLL_MS = 400
    pre_roll_samples = int(PRE_ROLL_MS / 1000 * sr)
    samples = np.concatenate([
        np.zeros(pre_roll_samples, dtype=samples.dtype),
        samples,
    ])

    duration      = len(samples) / sr
    total_frames  = max(1, int(duration * FPS))

    print(f'→ {duration:.1f}s audio  ·  {total_frames} frames @ {FPS}fps', file=sys.stderr)

    # ── Cipher mode detection + MP4 metadata tag ──────────────────────────────
    # Read WAV header to detect mode and build an audiocipher comment tag.
    # The tag survives AAC encoding (stored in the MP4 container, not audio),
    # allowing `audiocipher decode` to auto-detect mode from the MP4 file.
    _cipher_comment = None
    _unsafe_mode = False
    try:
        from utils import parse_wav_metadata  # type: ignore
        _wav_meta = parse_wav_metadata(audio_path)
        if _wav_meta:
            _cipher_comment = json.dumps({'audiocipher': _wav_meta}, separators=(',', ':'))
            _mode_str = _wav_meta.get('mode', '')
            _unsafe_mode = _mode_str in ('fsk', 'morse')
        else:
            _md_raw = b''
            with open(audio_path, 'rb') as _f:
                _md_raw = _f.read(2048)
            _unsafe_mode = (
                b'"mode": "fsk"'    in _md_raw or b'"mode":"fsk"'   in _md_raw
                or b'"mode": "morse"' in _md_raw or b'"mode":"morse"' in _md_raw
            )
    except Exception:
        _unsafe_mode = False

    if _unsafe_mode:
        print(
            '⚠  FSK / Morse cipher detected.\n'
            '   These modes do NOT survive lossy codecs (AAC, Opus, Telegram re-encode).\n'
            '   Re-encode with --mode hzalpha for a decodable video:\n'
            '     python3 audiocipher.py encode "your message" --mode hzalpha',
            file=sys.stderr,
        )

    # ── Audio codec + decodability notice ─────────────────────────────────────
    if _unsafe_mode and not twitter:
        print(
            '⚠  AAC will corrupt FSK/Morse frequencies.\n'
            '   Cipher CANNOT be decoded from this MP4.\n'
            '   Use --mode hzalpha when encoding if you need a decodable video.',
            file=sys.stderr,
        )
    else:
        print(
            '→ AAC audio (lossy). HZAlpha and WaveSig survive; '
            'FSK/Morse do not.\n'
            '   Decode directly from this MP4 or from the original WAV.',
            file=sys.stderr,
        )

    # ── Pre-compute scrolling waveform data + UI ───────────────────────────────
    print('→ Pre-computing waveform…', file=sys.stderr)
    bar_amps_padded, samples_per_px, n_pixels = _precompute_amps_scrolling(
        samples, sr, WAVE_W, window_seconds=window_seconds,
    )
    ui_arr, ui_mask   = _precompute_ui(W, H, MARGIN_X, title)
    fade_curve        = _make_fade_curve(WAVE_W)
    scanline_mult     = _make_scanline_mult(H)

    # Precompute per-frame-invariant arrays (avoid recreating each iteration)
    dist_from_center  = np.abs(np.arange(H)[:, np.newaxis] - WAVE_CENTER).astype(np.int32)
    half              = WAVE_W // 2
    played_col        = np.zeros(WAVE_W, dtype=bool)
    played_col[:half] = True
    upcoming_col      = ~played_col

    # ── Open ffmpeg process (stdin = raw RGB24) ────────────────────────────────
    # AAC 320k by default — ABP survives it and browsers can decode AAC-MP4
    # natively (Chrome cannot decode ALAC via Web Audio API).
    # ALAC is only used when explicitly requested via twitter=False AND a
    # non-ABP cipher mode that requires lossless preservation.
    audio_args = ['-c:a', 'aac', '-b:a', audio_bitrate]

    # Embed cipher metadata as an MP4 comment tag so `audiocipher decode` can
    # auto-detect mode even after AAC re-encoding strips the WAV RIFF metadata.
    _meta_args = ['-metadata', f'comment={_cipher_comment}'] if _cipher_comment else []

    cmd = [
        'ffmpeg',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', 'rgb24',
        '-r', str(FPS),
        '-i', '-',          # read frames from stdin
        '-i', audio_path,   # audio track
        '-c:v', 'libx264',
        '-preset', video_preset,
        '-crf', '20',
        # Pre-roll: delay audio by PRE_ROLL_MS ms (matches samples pre-roll above).
        # Absorbs AAC/Opus encoder priming delay so the first cipher tone survives
        # two codec hops (e.g. Telegram re-encode then Twitter re-encode).
        '-af', f'adelay={PRE_ROLL_MS}:all=1',
        *audio_args,
        *_meta_args,
        '-pix_fmt', 'yuv420p',
        '-shortest',
        '-movflags', '+faststart',
        '-y', output_path,
    ]

    stderr_dest = None if verbose else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=stderr_dest,
    )

    # ── Stream frames ─────────────────────────────────────────────────────────
    print('→ Rendering frames…', file=sys.stderr)
    try:
        for fi in range(total_frames):
            frac = fi / max(1, total_frames - 1)

            # Map current playback position → slice offset in the padded amp array.
            # At frac=0:   amp_offset = WAVE_W//2  (centre = padded[WAVE_W] = first sample)
            # At frac=1:   amp_offset = WAVE_W//2 + n_pixels - 1
            current_px = int(frac * max(0, n_pixels - 1))
            amp_offset = half + current_px

            frame = _render_frame_scrolling(
                bar_amps_padded, amp_offset, fade_curve,
                ui_arr, ui_mask, scanline_mult,
                dist_from_center, played_col, upcoming_col,
                MARGIN_X, WAVE_W, WAVE_CENTER, WAVE_MAX_H, H, W,
                progress=frac,
            )
            proc.stdin.write(frame.tobytes())

            if (fi + 1) % 30 == 0 or fi == total_frames - 1:
                pct = (fi + 1) / total_frames * 100
                print(f'\r  frame {fi + 1}/{total_frames}  ({pct:.0f}%)',
                      end='', flush=True, file=sys.stderr)

        print(file=sys.stderr)  # newline after progress
        proc.stdin.close()
        proc.wait()

    except Exception:
        proc.stdin.close()
        proc.kill()
        raise

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    return output_path


def generate_video_safe(
    audio_path:  str,
    output_path: str = 'out.mp4',
    **kwargs,
) -> str | None:
    """Same as generate_video() but returns None on error instead of raising."""
    try:
        return generate_video(audio_path, output_path, **kwargs)
    except Exception as e:
        try:
            print(f'⚠ video generation failed: {e}', file=sys.stderr)
        except Exception:
            pass
        return None
