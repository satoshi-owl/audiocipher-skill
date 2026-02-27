"""
abp_bridge.py — Thin bridge between AudioCipher skill and the ABP v1 codec.

Locates the sibling abp/ package (../abp relative to skill/) and exposes
two functions that skill/audiocipher.py can call directly:

    encode_abp(text, *, passphrase=None, profile='social_safe') -> np.ndarray
        Returns float32 mono 48 kHz audio samples.

    decode_abp(audio_path, *, passphrase=None) -> str
        Accepts any format ffmpeg can open (WAV, OGG, M4A, MP3 …).
        Returns the decoded plaintext string.
        Raises RuntimeError on decode failure (failure code included in message).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as _wavfile

# ── locate abp package ────────────────────────────────────────────────────────
# Preferred layout (bundled install):
#   skill/
#     abp/            ← abp package bundled alongside the skill files
#     abp_bridge.py   ← __file__
#
# Fallback layout (local dev monorepo):
#   audiocipher/
#     skill/          ← __file__ lives here
#     abp/
#       abp/          ← the importable package
#
# We first try a direct `import abp` (works when skill/ is in sys.path, which
# Python guarantees when running any script inside skill/).  If that fails we
# add the sibling abp/ repo root to sys.path so `import abp` resolves from
# the dev layout.

_SKILL_DIR = Path(__file__).resolve().parent           # …/audiocipher/skill

try:
    import abp as _abp_pkg  # noqa: F401 — probe only
except ImportError:
    # Bundled package not importable yet — try the dev monorepo layout.
    _ABP_REPO = _SKILL_DIR.parent / 'abp'              # …/audiocipher/abp
    if not (_ABP_REPO / 'abp' / '__init__.py').exists():
        raise ImportError(
            "ABP package not found.\n"
            "Run the skill installer (bash install.sh) to set up all dependencies,\n"
            "or place the abp/ package alongside the skill files."
        ) from None
    if str(_ABP_REPO) not in sys.path:
        sys.path.insert(0, str(_ABP_REPO))

from abp import encode_text, decode_audio          # noqa: E402
from abp.profiles import SR, PROFILES              # noqa: E402

# ── sample rate constant (48 kHz — re-exported for convenience) ───────────────
ABP_SR = SR


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_ffmpeg() -> bool:
    r = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    return r.returncode == 0


def _load_any(path: str) -> np.ndarray:
    """Load audio from any format → float32 mono 48 kHz."""
    path = str(path)
    ext  = Path(path).suffix.lower()

    if ext == '.wav':
        try:
            rate, raw = _wavfile.read(path)
            if raw.ndim == 2:
                raw = raw.mean(axis=1)
            if raw.dtype == np.int16:
                samples = raw.astype(np.float32) / 32768.0
            elif raw.dtype == np.int32:
                samples = raw.astype(np.float32) / 2_147_483_648.0
            else:
                samples = raw.astype(np.float32)
            if rate == SR:
                return samples
            # Wrong sample rate — fall through to ffmpeg resampler
        except Exception:
            pass

    if not _check_ffmpeg():
        raise RuntimeError(
            'ffmpeg not found — required to decode non-WAV audio.\n'
            '  macOS:  brew install ffmpeg\n'
            '  Ubuntu: sudo apt install ffmpeg\n'
            '  Windows: https://ffmpeg.org/download.html'
        )

    fd, tmp = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        result = subprocess.run(
            [
                'ffmpeg', '-y', '-i', path,
                '-ac', '1',
                '-ar', str(SR),
                '-c:a', 'pcm_s16le',
                tmp,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace')[-400:]
            raise RuntimeError(f'ffmpeg failed:\n{err}')
        _, raw = _wavfile.read(tmp)
        if raw.ndim == 2:
            raw = raw.mean(axis=1)
        return raw.astype(np.float32) / 32768.0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _save_wav(path: str, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    _wavfile.write(path, SR, (pcm * 32767).astype(np.int16))


# ── public API ────────────────────────────────────────────────────────────────

def encode_abp(
    text: str,
    *,
    passphrase: str | None = None,
    profile: str = 'social_safe',
) -> np.ndarray:
    """Encode *text* to ABP audio.

    Args:
        text:       Plaintext to encode (Unicode OK, any length the profile supports).
        passphrase: Optional encryption passphrase (XChaCha20-Poly1305 + Argon2id).
        profile:    Codec profile — 'social_safe' (default, more FEC) or 'fast'.

    Returns:
        float32 numpy array, mono, 48 kHz.

    Raises:
        ValueError: unknown profile name.
        RuntimeError: encode failed.
    """
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown ABP profile '{profile}'. Choose from: {list(PROFILES)}"
        )
    return encode_text(text, passphrase=passphrase, profile=profile)


def decode_abp(
    audio_path: str,
    *,
    passphrase: str | None = None,
) -> str:
    """Decode ABP audio from *audio_path* (any ffmpeg-readable format).

    Args:
        audio_path: Path to WAV / OGG / M4A / MP3 / … file.
        passphrase: Decryption passphrase (required if message was encrypted).

    Returns:
        Decoded plaintext string.

    Raises:
        RuntimeError: decode failed — message includes the FailureCode name and
                      any available SNR / FEC diagnostic info.
    """
    samples = _load_any(audio_path)
    result  = decode_audio(samples, passphrase=passphrase)

    if result.success:
        return result.text

    diag = f"[{result.failure.value}]"
    if result.snr_db is not None:
        diag += f"  SNR={result.snr_db:.1f} dB"
    raise RuntimeError(f"ABP decode failed: {diag}")


def abp_profiles() -> list[str]:
    """Return the list of valid ABP profile names."""
    return list(PROFILES)
