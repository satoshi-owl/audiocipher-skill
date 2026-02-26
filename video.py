"""
video.py — AudioCipher-branded animated waveform video generation.

Generates a 1280×720 (or custom) MP4 with:
  - AudioCipher dark brand theme (#0A0A0F bg, #00FF88 waveform)
  - ANIMATED waveform: bright played-region + dim upcoming-region + scanning playhead
  - Three-layer glow (outer bloom → inner → bright core) on played bars
  - Subtle amplitude grid (±25 / ±50 / ±75 % reference lines)
  - CRT scanline overlay (every 3rd row dimmed 18 %)
  - Optional title text (top-left, off-white, large monospace)
  - AUDIOCIPHER logo badge (top-right, green pill)
  - audiocipher.app watermark (bottom-right, dim green)
  - H.264 + AAC, +faststart for Twitter/X streaming

Rendering: frames piped to ffmpeg via stdin (rawvideo RGB24).
  - Pre-renders played/unplayed waveform arrays once.
  - Per-frame: numpy column-slice blend + playhead line + scanlines.
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

FPS = 30  # output frame rate


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
# Pre-render animation layers
# ─────────────────────────────────────────────────────────────────────────────

def _build_layers(
    samples: np.ndarray,
    sr:      int,
    W:       int = 1280,
    H:       int = 720,
    title:   str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Pre-render three numpy arrays (H, W, 3) uint8:
      played_arr   — bars in bright green with three-layer glow
      unplayed_arr — bars in dim green  +  badge / title / watermark overlaid
      ui_arr       — badge / title / watermark on black (for compositing)

    Also returns (MARGIN_X, WAVE_W, WAVE_CENTER).
    """
    from PIL import Image, ImageDraw

    MARGIN_X    = 56
    WAVE_W      = W - 2 * MARGIN_X
    WAVE_CENTER = H // 2
    WAVE_MAX_H  = int(H * 0.30)

    # ── Background + grid + centre line ──────────────────────────────────────
    base = np.full((H, W, 3), _BG, dtype=np.uint8)
    for pct in (0.25, 0.50, 0.75):
        for sign in (1, -1):
            gy = WAVE_CENTER + int(sign * pct * WAVE_MAX_H)
            if 0 <= gy < H:
                base[gy, MARGIN_X:MARGIN_X + WAVE_W] = _GRID_LINE
    base[WAVE_CENTER, MARGIN_X:MARGIN_X + WAVE_W] = _GREEN_INNER

    # ── Bar heights ───────────────────────────────────────────────────────────
    n = len(samples)
    col_edges = np.linspace(0, n, WAVE_W + 1, dtype=int)
    amps = np.zeros(WAVE_W, dtype=np.float32)
    for ci in range(WAVE_W):
        seg = samples[col_edges[ci]:col_edges[ci + 1]]
        if len(seg):
            amps[ci] = float(np.max(np.abs(seg.astype(np.float64))))
    peak = float(amps.max())
    if peak > 0:
        amps /= peak
    bar_heights = np.maximum(1, (amps * WAVE_MAX_H).astype(int))

    # ── played_arr: bright bars with three-layer glow ─────────────────────────
    played = base.copy()
    for ci, bh in enumerate(bar_heights):
        px = MARGIN_X + ci
        played[max(0, WAVE_CENTER - bh - 4):min(H, WAVE_CENTER + bh + 5), px] = _GREEN_OUTER
        played[max(0, WAVE_CENTER - bh - 1):min(H, WAVE_CENTER + bh + 2), px] = _GREEN_INNER
        played[max(0, WAVE_CENTER - bh)    :min(H, WAVE_CENTER + bh + 1), px] = _GREEN_CORE

    # ── unplayed_arr: dim bars ────────────────────────────────────────────────
    unplayed = base.copy()
    for ci, bh in enumerate(bar_heights):
        px = MARGIN_X + ci
        unplayed[max(0, WAVE_CENTER - bh):min(H, WAVE_CENTER + bh + 1), px] = _GREEN_DIM

    # ── UI overlay (badge, title, watermark) on black ─────────────────────────
    # Render on pure black so non-black pixels = UI pixels (mask via .any(axis=2))
    ui_arr = np.zeros((H, W, 3), dtype=np.uint8)
    ui_img  = Image.fromarray(ui_arr, 'RGB')
    ui_draw = ImageDraw.Draw(ui_img)
    font_title = _load_font(30)
    font_sm    = _load_font(13)

    if title:
        ui_draw.text((MARGIN_X, 30), title.upper(), fill=_WHITE, font=font_title)

    _draw_logo_badge(ui_draw, right_x=W - MARGIN_X, top_y=28, font_sm=font_sm)

    wm_text = 'audiocipher.app'
    wm_x = W - MARGIN_X - len(wm_text) * 7
    ui_draw.text((max(MARGIN_X, wm_x), H - 22), wm_text, fill=_GREEN_DIM, font=font_sm)

    ui_arr = np.array(ui_img)

    return played, unplayed, ui_arr, MARGIN_X, WAVE_W, WAVE_CENTER


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def _make_scanline_mult(H: int) -> np.ndarray:
    """Float32 (H, 1, 1) multiplier: 0.82 every 3rd row, else 1.0."""
    m = np.ones((H, 1, 1), dtype=np.float32)
    m[::3] = 0.82
    return m


def _render_frame_animated(
    played:   np.ndarray,
    unplayed: np.ndarray,
    ui_arr:   np.ndarray,
    ui_mask:  np.ndarray,
    scanline_mult: np.ndarray,
    playhead_px: int,
    MARGIN_X: int,
    WAVE_W:   int,
    H:        int,
    W:        int,
) -> np.ndarray:
    """Render one video frame as a (H, W, 3) uint8 numpy array."""

    # Blend played (left) + unplayed (right)
    arr = unplayed.copy()
    if playhead_px > MARGIN_X:
        arr[:, MARGIN_X:playhead_px] = played[:, MARGIN_X:playhead_px]

    # Playhead scan line — 3px wide, centre is brightest
    for dx, col in ((0, _PLAYHEAD), (-1, _GREEN_INNER), (1, _GREEN_INNER)):
        gx = playhead_px + dx
        if MARGIN_X <= gx < MARGIN_X + WAVE_W:
            arr[:, gx] = col

    # CRT scanlines
    arr = (arr.astype(np.float32) * scanline_mult).clip(0, 255).astype(np.uint8)

    # Composite UI (badge / title / watermark) — no scanlines on top chrome
    arr[ui_mask] = ui_arr[ui_mask]

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_video(
    audio_path:    str,
    output_path:   str  = 'out.mp4',
    style:         str  = 'null',
    resolution:    str  = '1280x720',
    title:         str | None = None,
    twitter:       bool = False,   # True → AAC 320k (social posting); False → ALAC lossless (default)
    audio_bitrate: str  = '320k',  # only used when twitter=True
    video_preset:  str  = 'fast',
    verbose:       bool = False,
) -> str:
    """
    Generate an animated AudioCipher waveform video from an audio file.

    The waveform plays through in real time: bars to the left of the playhead
    glow bright green; bars to the right are dim. A scanning line marks the
    current position.

    Args:
        audio_path:    Input audio (WAV / MP3 / any ffmpeg-readable format)
        output_path:   Output MP4 path  (default: out.mp4)
        style:         Reserved for future visual styles
        resolution:    Output resolution as 'WxH'  (default: 1280x720)
        title:         Title text displayed top-left (e.g. "NULL")
        twitter:       If True, encode audio as AAC 320k for Twitter/X posting.
                       Default False = ALAC lossless — MP4 stays fully decodable.
        audio_bitrate: AAC bitrate used only when twitter=True  (default: 320k)
        video_preset:  libx264 preset: ultrafast / fast / medium
        verbose:       Show ffmpeg stderr

    Returns:
        Absolute path to the generated MP4.

    Notes:
        Default (twitter=False): ALAC lossless audio — decode directly with:
            python3 audiocipher.py decode cipher.mp4
        twitter=True: AAC lossy — HZAlpha survives; WaveSig/FSK/Morse do not.
            Twitter also re-encodes on upload, adding a second lossy pass.
    """
    if not check_ffmpeg():
        raise RuntimeError('ffmpeg not available. Run check_or_install_ffmpeg().')

    audio_path  = str(Path(audio_path).resolve())
    output_path = str(Path(output_path).resolve())

    try:
        W, H = map(int, resolution.lower().split('x'))
    except (ValueError, AttributeError):
        W, H = 1280, 720

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

    # ── Audio codec + decodability notice ─────────────────────────────────────
    if not twitter:
        print(
            '→ Audio: ALAC lossless — decode this MP4 directly with:\n'
            '     python3 audiocipher.py decode ' + os.path.basename(output_path),
            file=sys.stderr,
        )
    else:
        # Warn if source mode won't survive AAC
        try:
            _md_raw = b''
            with open(audio_path, 'rb') as _f:
                _md_raw = _f.read(2048)
            _lossy_unsafe = (
                b'"mode": "ggwave"' in _md_raw or b'"mode":"ggwave"' in _md_raw
                or b'"mode": "fsk"' in _md_raw or b'"mode": "morse"' in _md_raw
            )
        except Exception:
            _lossy_unsafe = False

        if _lossy_unsafe:
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

    # ── Pre-render waveform layers ─────────────────────────────────────────────
    print('→ Pre-rendering waveform layers…', file=sys.stderr)
    played, unplayed, ui_arr, MARGIN_X, WAVE_W, WAVE_CENTER = _build_layers(
        samples, sr, W=W, H=H, title=title,
    )
    ui_mask      = ui_arr.any(axis=2)          # True where UI pixels exist
    scanline_mult = _make_scanline_mult(H)

    # ── Open ffmpeg process (stdin = raw RGB24) ────────────────────────────────
    # Audio: ALAC lossless by default (MP4 stays decodable); AAC when --twitter
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
            frac        = fi / total_frames
            playhead_px = MARGIN_X + int(frac * WAVE_W)

            frame = _render_frame_animated(
                played, unplayed, ui_arr, ui_mask, scanline_mult,
                playhead_px, MARGIN_X, WAVE_W, H, W,
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
