"""
video.py — AudioCipher-branded scrolling waveform video generation.

Generates a 1280×720 (or custom) MP4 with:
  - AudioCipher dark brand theme (#0A0A0F bg, #00FF88 waveform)
  - SCROLLING waveform: the audio flows left past a fixed centre playhead
    · Left half  (played)    — bright green with three-layer glow, tapers toward left edge
    · Right half (upcoming)  — dim green, tapers toward right edge
    · Fixed centre playhead  — bright vertical line
    · Visible window ≈ 4 s of audio (zoom-in with --window-seconds)
  - Subtle amplitude grid (±25 / ±50 / ±75 % reference lines)
  - CRT scanline overlay (every 3rd row dimmed 18 %)
  - Optional title text (top-left, off-white, large monospace)
  - AUDIOCIPHER logo badge (top-right, green pill)
  - audiocipher.app watermark (bottom-right, dim green)
  - H.264 video + ALAC lossless audio by default (MP4 is fully decodable);
    add --twitter for AAC 320k social posting.

Rendering: frames piped to ffmpeg via stdin (rawvideo RGB24).
  - Pre-computes a normalised per-pixel amplitude array once.
  - Per-frame: vectorised numpy mask operations (no Python per-pixel loops).
  - No temp PNGs — memory-efficient streaming.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Brand palette
# ─────────────────────────────────────────────────────────────────────────────
_BG          = (10,  10,  15)   # #0A0A0F
_GREEN_CORE  = (0,  255, 136)   # #00FF88 — waveform core / playhead
_GREEN_INNER = (0,  150,  75)
_GREEN_OUTER = (0,   40,  20)
_GREEN_DIM   = (0,   55,  28)   # upcoming bars (pre-play)
_GRID_LINE   = (0,   22,  11)
_WHITE       = (218, 218, 228)
_BADGE_BG    = (14,  30,  21)
_PLAYHEAD    = (180, 255, 210)  # slightly cooler white-green for scan line

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
    BADGE_W, BADGE_H, RADIUS = 190, 30, 6
    bx = right_x - BADGE_W
    by = top_y
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
    try:
        draw.rounded_rectangle([(bx, by), (bx + BADGE_W, by + BADGE_H)],
                               radius=RADIUS, outline=_GREEN_DIM, width=1)
    except TypeError:
        pass
    DOT_R = 4
    dot_cx, dot_cy = bx + 16, by + BADGE_H // 2
    draw.ellipse([(dot_cx - DOT_R, dot_cy - DOT_R), (dot_cx + DOT_R, dot_cy + DOT_R)],
                 fill=_GREEN_CORE)
    draw.text((bx + 28, by + BADGE_H // 2 - 7), 'AUDIOCIPHER', fill=_GREEN_CORE, font=font_sm)


# ─────────────────────────────────────────────────────────────────────────────
# CRT scanlines
# ─────────────────────────────────────────────────────────────────────────────

def _make_scanline_mult(H: int) -> np.ndarray:
    """Float32 (H, 1, 1) multiplier: 0.82 every 3rd row, else 1.0."""
    m = np.ones((H, 1, 1), dtype=np.float32)
    m[::3] = 0.82
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
    1-D float32 (WAVE_W,): multiplied into bar heights so bars at the
    edges of the scrolling window taper gracefully (0.35 → 1.0 → 0.35).
    """
    half  = WAVE_W // 2
    left  = np.linspace(0.35, 1.0, half,           dtype=np.float32)
    right = np.linspace(1.0, 0.35, WAVE_W - half,  dtype=np.float32)
    return np.concatenate([left, right])


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame renderer (fully vectorised — no Python per-pixel loops)
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame_scrolling(
    bar_amps_padded:  np.ndarray,   # pre-computed amp array (padded)
    amp_offset:       int,          # slice start in bar_amps_padded
    fade_curve:       np.ndarray,   # (WAVE_W,) height taper
    ui_arr:           np.ndarray,   # (H, W, 3) UI chrome
    ui_mask:          np.ndarray,   # (H, W) bool
    scanline_mult:    np.ndarray,   # (H, 1, 1) CRT dimming
    dist_from_center: np.ndarray,   # (H, 1) precomputed |y - WAVE_CENTER|
    played_col:       np.ndarray,   # (WAVE_W,) bool — True for left half
    upcoming_col:     np.ndarray,   # (WAVE_W,) bool — True for right half
    MARGIN_X:         int,
    WAVE_W:           int,
    WAVE_CENTER:      int,
    WAVE_MAX_H:       int,
    H:                int,
    W:                int,
) -> np.ndarray:
    """Render one scrolling-waveform video frame as a (H, W, 3) uint8 array."""

    # ── Background + amplitude grid ───────────────────────────────────────────
    arr = np.full((H, W, 3), _BG, dtype=np.uint8)
    for pct in (0.25, 0.50, 0.75):
        for sign in (1, -1):
            gy = WAVE_CENTER + int(sign * pct * WAVE_MAX_H)
            if 0 <= gy < H:
                arr[gy, MARGIN_X:MARGIN_X + WAVE_W] = _GRID_LINE
    arr[WAVE_CENTER, MARGIN_X:MARGIN_X + WAVE_W] = _GREEN_INNER

    # ── Amplitude window + fade-tapered bar heights ───────────────────────────
    window_amps = bar_amps_padded[amp_offset: amp_offset + WAVE_W]   # (WAVE_W,)
    bh = np.maximum(1, (window_amps * fade_curve * WAVE_MAX_H).astype(int))  # (WAVE_W,)

    # ── Vectorised glow masks (H × WAVE_W) ────────────────────────────────────
    # dist_from_center is (H, 1); bh is (WAVE_W,) → both broadcast to (H, WAVE_W)
    outer_mask = dist_from_center <= bh[np.newaxis, :] + 4   # (H, WAVE_W)
    inner_mask = dist_from_center <= bh[np.newaxis, :] + 1
    core_mask  = dist_from_center <= bh[np.newaxis, :]

    # wv is a (H, WAVE_W, 3) view into arr — writes go directly into arr
    wv = arr[:, MARGIN_X:MARGIN_X + WAVE_W]

    # Upcoming bars — dim, no glow
    wv[core_mask  & upcoming_col] = _GREEN_DIM

    # Played bars — three-layer glow (outer painted first, core last)
    wv[outer_mask & played_col]   = _GREEN_OUTER
    wv[inner_mask & played_col]   = _GREEN_INNER
    wv[core_mask  & played_col]   = _GREEN_CORE

    # ── Centre playhead (fixed vertical line) ─────────────────────────────────
    ctr = MARGIN_X + WAVE_W // 2
    for dx, col in ((0, _PLAYHEAD), (-1, _GREEN_INNER), (1, _GREEN_INNER)):
        gx = ctr + dx
        if MARGIN_X <= gx < MARGIN_X + WAVE_W:
            arr[:, gx] = col

    # ── CRT scanlines ─────────────────────────────────────────────────────────
    arr = (arr.astype(np.float32) * scanline_mult).clip(0, 255).astype(np.uint8)

    # ── UI chrome (no scanlines on badge / title / watermark) ─────────────────
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

    ⚠  Cipher mode compatibility with compressed audio:
        HZAlpha  — survives AAC, Opus, and Telegram/Twitter re-encoding. ✓
        WaveSig  — NOT safe with any lossy codec (46.875 Hz bin spacing). ✗
        FSK/Morse — NOT safe with lossy codecs. ✗
        → Use encode --mode hzalpha for any cipher you plan to share as a video.
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

    duration      = len(samples) / sr
    total_frames  = max(1, int(duration * FPS))

    print(f'→ {duration:.1f}s audio  ·  {total_frames} frames @ {FPS}fps', file=sys.stderr)

    # ── Cipher mode warning ────────────────────────────────────────────────────
    try:
        _md_raw = b''
        with open(audio_path, 'rb') as _f:
            _md_raw = _f.read(2048)
        _unsafe_mode = (
            b'"mode": "ggwave"' in _md_raw or b'"mode":"ggwave"' in _md_raw
            or b'"mode": "fsk"'  in _md_raw or b'"mode":"fsk"'   in _md_raw
            or b'"mode": "morse"' in _md_raw or b'"mode":"morse"' in _md_raw
        )
    except Exception:
        _unsafe_mode = False

    if _unsafe_mode:
        print(
            '⚠  WaveSig / FSK / Morse cipher detected.\n'
            '   These modes do NOT survive lossy codecs (AAC, Opus, Telegram re-encode).\n'
            '   Re-encode with --mode hzalpha for a decodable video:\n'
            '     python3 audiocipher.py encode "your message" --mode hzalpha',
            file=sys.stderr,
        )

    # ── Audio codec + decodability notice ─────────────────────────────────────
    if not twitter:
        print(
            '→ Audio: ALAC lossless — decode this MP4 directly with:\n'
            '     python3 audiocipher.py decode ' + os.path.basename(output_path),
            file=sys.stderr,
        )
    else:
        if _unsafe_mode:
            print(
                '⚠  --twitter: AAC will corrupt WaveSig/FSK/Morse frequencies.\n'
                '   Cipher CANNOT be decoded from this MP4.\n'
                '   Use --mode hzalpha when encoding if you need a decodable video.',
                file=sys.stderr,
            )
        else:
            print(
                '→ --twitter: AAC audio (lossy). HZAlpha survives; '
                'WaveSig/FSK/Morse do not.\n'
                '   Decode from the original WAV for guaranteed accuracy.',
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
    audio_args = (
        ['-c:a', 'aac', '-b:a', audio_bitrate]
        if twitter else
        ['-c:a', 'alac']
    )

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
        *audio_args,
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
        print(f'⚠ video generation failed: {e}', file=sys.stderr)
        return None
