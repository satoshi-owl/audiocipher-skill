"""
video.py — AudioCipher-branded waveform video generation.

Generates a 1280×720 (or custom) MP4 with:
  - AudioCipher dark brand theme (#0A0A0F bg, #00FF88 waveform)
  - Three-layer waveform glow (outer bloom → inner → bright core)
  - Subtle amplitude grid (±25 / ±50 / ±75% reference lines)
  - CRT scanline overlay (every 3rd row dimmed 18%)
  - Optional title text (top-left, off-white, large monospace)
  - AUDIOCIPHER logo badge (top-right, green pill)
  - audiocipher.app watermark (bottom-right, dim green)
  - H.264 + AAC, +faststart for Twitter/X streaming

Rendering: waveform drawn in Python (PIL + numpy) → static frame PNG,
then muxed with audio via ffmpeg (-loop 1 -tune stillimage).
This avoids showwaves / geq filter quirks in various ffmpeg builds.
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
_BG          = (10,  10,  15)   # #0A0A0F — near-black background
_GREEN_CORE  = (0,  255, 136)   # #00FF88 — primary brand colour (waveform core)
_GREEN_INNER = (0,  150,  75)   # mid-tone glow layer
_GREEN_OUTER = (0,   40,  20)   # soft outer bloom
_GREEN_DIM   = (0,   90,  45)   # watermark / secondary text
_GRID_LINE   = (0,   22,  11)   # barely-visible amplitude reference lines
_WHITE       = (218, 218, 228)  # off-white for title text
_BADGE_BG    = (14,  30,  21)   # logo badge background


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg availability
# ─────────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which('ffmpeg') is not None


def check_or_install_ffmpeg() -> bool:
    """
    Check for ffmpeg; attempt to install via Homebrew (macOS) or apt (Linux)
    if not found.  Returns True if ffmpeg is available after the attempt.
    """
    if check_ffmpeg():
        return True

    system = platform.system()
    try:
        if system == 'Darwin':
            print('→ ffmpeg not found. Installing via Homebrew…')
            subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        elif system == 'Linux':
            print('→ ffmpeg not found. Installing via apt…')
            subprocess.run(
                ['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True,
            )
        else:
            print(
                f'⚠ ffmpeg not found and auto-install is not supported on {system}. '
                'Please install from https://ffmpeg.org',
                file=sys.stderr,
            )
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'⚠ ffmpeg install failed: {e}', file=sys.stderr)
        return False

    return check_ffmpeg()


# ─────────────────────────────────────────────────────────────────────────────
# Font helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int):
    """Return a PIL ImageFont at `size` pt, falling back to the bitmap default."""
    from PIL import ImageFont  # type: ignore
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
    """
    Render the AUDIOCIPHER logo badge, right-aligned at (right_x, top_y).

    Layout (190 × 30 px pill):
      ╭──────────────────────╮
      │  ●  AUDIOCIPHER      │
      ╰──────────────────────╯
    """
    BADGE_W, BADGE_H = 190, 30
    RADIUS  = 6
    bx = right_x - BADGE_W
    by = top_y

    # ── Rounded rect (Pillow 8.2+ or manual fallback) ────────────────────────
    try:
        draw.rounded_rectangle(
            [(bx, by), (bx + BADGE_W, by + BADGE_H)],
            radius=RADIUS, fill=_BADGE_BG,
        )
    except AttributeError:
        # Pillow < 8.2 — approximate with rectangle + corner ellipses
        r2 = RADIUS * 2
        draw.rectangle([(bx + RADIUS, by), (bx + BADGE_W - RADIUS, by + BADGE_H)],
                       fill=_BADGE_BG)
        draw.rectangle([(bx, by + RADIUS), (bx + BADGE_W, by + BADGE_H - RADIUS)],
                       fill=_BADGE_BG)
        for cx, cy in [
            (bx, by), (bx + BADGE_W - r2, by),
            (bx, by + BADGE_H - r2), (bx + BADGE_W - r2, by + BADGE_H - r2),
        ]:
            draw.ellipse([(cx, cy), (cx + r2, cy + r2)], fill=_BADGE_BG)

    # ── Thin green border ─────────────────────────────────────────────────────
    try:
        draw.rounded_rectangle(
            [(bx, by), (bx + BADGE_W, by + BADGE_H)],
            radius=RADIUS, outline=_GREEN_DIM, width=1,
        )
    except TypeError:
        pass  # older Pillow may not support outline on rounded_rectangle

    # ── Green dot accent ──────────────────────────────────────────────────────
    DOT_R = 4
    dot_cx = bx + 16
    dot_cy = by + BADGE_H // 2
    draw.ellipse(
        [(dot_cx - DOT_R, dot_cy - DOT_R), (dot_cx + DOT_R, dot_cy + DOT_R)],
        fill=_GREEN_CORE,
    )

    # ── "AUDIOCIPHER" text ────────────────────────────────────────────────────
    draw.text((bx + 28, by + BADGE_H // 2 - 7), 'AUDIOCIPHER',
              fill=_GREEN_CORE, font=font_sm)


# ─────────────────────────────────────────────────────────────────────────────
# Waveform frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame(
    samples: np.ndarray,
    sr:      int,
    W:       int = 1280,
    H:       int = 720,
    title:   str | None = None,
) -> 'PIL.Image.Image':  # type: ignore[name-defined]
    """
    Render a single waveform frame as a PIL RGB image.

    Visual layers (bottom → top):
      1. Near-black background (#0A0A0F)
      2. Subtle amplitude grid (±25 / ±50 / ±75 %, barely visible)
      3. Horizontal centre line (always visible, brand green mid-tone)
      4. Three-layer glow waveform: outer bloom → inner glow → bright core
      5. CRT scanline overlay (every 3rd row dimmed 18 %)
      6. Title text     — top-left, off-white, 30pt monospace
      7. AUDIOCIPHER badge — top-right, brand green pill
      8. Watermark       — bottom-right, dim green 'audiocipher.app'
    """
    from PIL import Image, ImageDraw  # type: ignore

    MARGIN_X    = 56
    WAVE_W      = W - 2 * MARGIN_X     # waveform horizontal span (pixels)
    WAVE_CENTER = H // 2               # vertical midpoint
    WAVE_MAX_H  = int(H * 0.30)        # max half-height of tallest bar

    # ── 1. Background ─────────────────────────────────────────────────────────
    arr = np.full((H, W, 3), _BG, dtype=np.uint8)

    # ── 2. Subtle amplitude grid ──────────────────────────────────────────────
    for pct in (0.25, 0.50, 0.75):
        for sign in (1, -1):
            gy = WAVE_CENTER + int(sign * pct * WAVE_MAX_H)
            if 0 <= gy < H:
                arr[gy, MARGIN_X:MARGIN_X + WAVE_W] = _GRID_LINE

    # ── 3. Centre line ────────────────────────────────────────────────────────
    arr[WAVE_CENTER, MARGIN_X:MARGIN_X + WAVE_W] = _GREEN_INNER

    # ── 4. Per-column peak amplitude → bar heights ────────────────────────────
    n = len(samples)
    col_edges = np.linspace(0, n, WAVE_W + 1, dtype=int)

    amps = np.zeros(WAVE_W, dtype=np.float32)
    for ci in range(WAVE_W):
        seg = samples[col_edges[ci]:col_edges[ci + 1]]
        if len(seg):
            amps[ci] = float(np.max(np.abs(seg.astype(np.float64))))

    peak_global = float(amps.max())
    if peak_global > 0:
        amps /= peak_global

    bar_heights = np.maximum(1, (amps * WAVE_MAX_H).astype(int))

    # Draw three-layer glow for each column
    for ci, bh in enumerate(bar_heights):
        px = MARGIN_X + ci

        # Outer bloom  (core ± 4 px each side)
        y1 = max(0, WAVE_CENTER - bh - 4)
        y2 = min(H, WAVE_CENTER + bh + 5)
        arr[y1:y2, px] = _GREEN_OUTER

        # Inner glow  (core ± 1 px)
        y1 = max(0, WAVE_CENTER - bh - 1)
        y2 = min(H, WAVE_CENTER + bh + 2)
        arr[y1:y2, px] = _GREEN_INNER

        # Core bar  (exact amplitude)
        y1 = max(0, WAVE_CENTER - bh)
        y2 = min(H, WAVE_CENTER + bh + 1)
        arr[y1:y2, px] = _GREEN_CORE

    # ── 5. CRT scanlines ──────────────────────────────────────────────────────
    arr[::3] = (arr[::3].astype(np.float32) * 0.82).clip(0, 255).astype(np.uint8)

    img  = Image.fromarray(arr, 'RGB')
    draw = ImageDraw.Draw(img)

    # ── Load fonts ─────────────────────────────────────────────────────────────
    font_title = _load_font(30)
    font_sm    = _load_font(13)

    # ── 6. Title (top-left) ────────────────────────────────────────────────────
    if title:
        draw.text((MARGIN_X, 30), title.upper(), fill=_WHITE, font=font_title)

    # ── 7. AUDIOCIPHER logo badge (top-right) ─────────────────────────────────
    _draw_logo_badge(draw, right_x=W - MARGIN_X, top_y=28, font_sm=font_sm)

    # ── 8. Watermark (bottom-right) ───────────────────────────────────────────
    # Estimate text width conservatively so it sits inside the margin
    wm_text = 'audiocipher.app'
    wm_x = W - MARGIN_X - len(wm_text) * 7   # ~7px per char at 13pt
    draw.text((max(MARGIN_X, wm_x), H - 22), wm_text,
              fill=_GREEN_DIM, font=font_sm)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_video(
    audio_path:    str,
    output_path:   str  = 'out.mp4',
    style:         str  = 'null',        # reserved for future styles
    resolution:    str  = '1280x720',
    title:         str | None = None,
    audio_bitrate: str  = '192k',
    video_preset:  str  = 'fast',
    verbose:       bool = False,
) -> str:
    """
    Generate an AudioCipher-branded waveform video from an audio file.

    Args:
        audio_path:     Path to input audio (WAV / MP3 / any ffmpeg-supported)
        output_path:    Output MP4 path  (default: out.mp4)
        style:          Reserved; only 'null' currently defined
        resolution:     Output resolution as 'WxH'  (default: 1280x720)
        title:          Optional title text displayed top-left (e.g. "NULL")
        audio_bitrate:  AAC audio bitrate  (default: 192k)
        video_preset:   ffmpeg libx264 preset: ultrafast / fast / medium
        verbose:        Show ffmpeg output

    Returns:
        Absolute path to the generated MP4 file.

    Raises:
        RuntimeError:                  ffmpeg not available
        subprocess.CalledProcessError: ffmpeg returned non-zero exit code

    Notes:
        - Twitter/X: H.264 + AAC, 1280×720, yuv420p, +faststart ✓
        - Waveform is rendered by Python (PIL + numpy) as a static frame,
          then muxed with audio via ffmpeg `-loop 1 -tune stillimage`.
          This avoids ffmpeg showwaves / geq filter incompatibilities.
    """
    if not check_ffmpeg():
        raise RuntimeError(
            'ffmpeg is not installed. Run check_or_install_ffmpeg() '
            'or install from https://ffmpeg.org'
        )

    audio_path  = str(Path(audio_path).resolve())
    output_path = str(Path(output_path).resolve())

    try:
        W, H = map(int, resolution.lower().split('x'))
    except (ValueError, AttributeError):
        W, H = 1280, 720

    # ── Load audio for waveform rendering ─────────────────────────────────────
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

    # ── Render waveform frame ──────────────────────────────────────────────────
    frame_img = _render_frame(samples, sr, W=W, H=H, title=title)

    # ── Write temp PNG, mux with audio via ffmpeg ─────────────────────────────
    tmp_fd, frame_path = tempfile.mkstemp(suffix='.png')
    os.close(tmp_fd)
    try:
        frame_img.save(frame_path, 'PNG')

        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i',    frame_path,     # static waveform image
            '-i',    audio_path,     # audio track
            '-c:v',  'libx264',
            '-preset', video_preset,
            '-crf',  '18',           # high quality — static frame compresses well
            '-tune', 'stillimage',   # x264 optimisation for non-moving content
            '-c:a',  'aac',
            '-b:a',  audio_bitrate,
            '-pix_fmt',  'yuv420p',      # required for Twitter/X
            '-shortest',                 # end when audio track ends
            '-movflags', '+faststart',   # Twitter-friendly streaming
            '-y', output_path,
        ]

        run_kw: dict = {'check': True}
        if not verbose:
            run_kw['stdout'] = subprocess.DEVNULL
            run_kw['stderr'] = subprocess.DEVNULL

        subprocess.run(cmd, **run_kw)
    finally:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    return output_path


def generate_video_safe(
    audio_path:  str,
    output_path: str = 'out.mp4',
    **kwargs,
) -> str | None:
    """
    Same as generate_video() but returns None on any error instead of raising.
    Prints a warning to stderr on failure.
    """
    try:
        return generate_video(audio_path, output_path, **kwargs)
    except Exception as e:
        print(f'⚠ video generation failed: {e}', file=sys.stderr)
        return None
