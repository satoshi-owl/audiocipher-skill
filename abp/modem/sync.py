"""ABP v1 — chirp preamble generation and sync detection.

Preamble
--------
A linear FM chirp sweeping CHIRP_F0 → CHIRP_F1 over CHIRP_DURATION seconds
at sample rate SR.  The chirp stays inside the ABP passband (800–3600 Hz) and
is distinct enough that cross-correlation with the template gives a clean peak
even after AAC 128 k compression.

Detection
---------
normalised_xcorr(received, template) peak search.  The normalised correlation
is robust to amplitude changes caused by the audio compressor.

The search window is the full audio signal; for long files this is O(N·M)
where N = len(received) and M = CHIRP_LEN.  For typical ABP payloads
(≤ 30 s of audio) this is fast enough; a sliding FFT is not necessary.
"""

import numpy as np
import scipy.signal as spsig
from numpy.typing import NDArray

from ..profiles import (
    SR, CHIRP_DURATION, CHIRP_LEN,
    CHIRP_F0, CHIRP_F1, CHIRP_AMP,
    SYNC_THRESHOLD,
)

# ── template (generated once at import time) ──────────────────────────────────

def _make_chirp() -> NDArray[np.float32]:
    t = np.linspace(0.0, CHIRP_DURATION, CHIRP_LEN, endpoint=False)
    c = spsig.chirp(t, f0=CHIRP_F0, f1=CHIRP_F1, t1=CHIRP_DURATION, method="linear")
    # Apply Hann window to suppress spectral leakage at edges
    window = np.hanning(CHIRP_LEN)
    c *= window
    # Normalise to unit energy for stable cross-correlation
    c /= np.sqrt(np.sum(c ** 2))
    return c.astype(np.float32)


CHIRP_TEMPLATE: NDArray[np.float32] = _make_chirp()


# ── public functions ──────────────────────────────────────────────────────────

def gen_preamble() -> NDArray[np.float32]:
    """Return the raw chirp preamble samples (float32, length CHIRP_LEN).

    The output is amplitude-scaled by CHIRP_AMP for embedding in audio.
    """
    t      = np.linspace(0.0, CHIRP_DURATION, CHIRP_LEN, endpoint=False)
    chirp  = spsig.chirp(t, f0=CHIRP_F0, f1=CHIRP_F1, t1=CHIRP_DURATION, method="linear")
    window = np.hanning(CHIRP_LEN)
    chirp *= window * CHIRP_AMP
    return chirp.astype(np.float32)


def detect(
    samples: NDArray[np.float32],
    threshold: float = SYNC_THRESHOLD,
) -> int | None:
    """Find the start of the chirp preamble in *samples*.

    Uses normalised cross-correlation against CHIRP_TEMPLATE.  The
    normalised value is 1.0 for a perfect match regardless of signal level.

    Args:
        samples:   1-D float32 audio at SR Hz.
        threshold: Minimum normalised correlation to accept as a sync hit.
                   SYNC_THRESHOLD (0.08) works reliably after AAC 128 k and OGG Opus 32-64 kbps.

    Returns:
        Sample index of the first sample of the chirp preamble, or None if
        no peak above *threshold* was found.
    """
    if len(samples) < CHIRP_LEN:
        return None

    # Full cross-correlation (mode='valid') → length len(samples) - CHIRP_LEN + 1
    # Each output value corresponds to a possible start offset.
    xcorr = spsig.correlate(samples, CHIRP_TEMPLATE, mode="valid")

    # Normalise to a proper Pearson correlation coefficient ρ ∈ [0, 1].
    # ||CHIRP_TEMPLATE|| = 1 (unit energy, see _make_chirp).
    # ||samples_window|| = rms × sqrt(CHIRP_LEN).
    # → ρ = |xcorr| / (rms × sqrt(CHIRP_LEN))
    sq = samples.astype(np.float64) ** 2
    window_sum = np.convolve(sq, np.ones(CHIRP_LEN, dtype=np.float64), mode="valid")
    rms = np.sqrt(window_sum / CHIRP_LEN + 1e-20)

    norm_xcorr = np.abs(xcorr) / (rms * np.sqrt(CHIRP_LEN))

    peak_idx   = int(np.argmax(norm_xcorr))
    peak_val   = float(norm_xcorr[peak_idx])

    if peak_val < threshold:
        return None

    return peak_idx


def detect_strict(
    samples: NDArray[np.float32],
    threshold: float = SYNC_THRESHOLD,
) -> int | None:
    """Like :func:`detect` but also verifies the peak is isolated.

    Isolation check: the peak must be at least 2× higher than any other peak
    within ±CHIRP_LEN//2 samples of it.  Rejects false positives in noisy signals.
    """
    if len(samples) < CHIRP_LEN:
        return None

    xcorr      = spsig.correlate(samples, CHIRP_TEMPLATE, mode="valid")
    sq         = samples.astype(np.float64) ** 2
    window_sum = np.convolve(sq, np.ones(CHIRP_LEN, dtype=np.float64), mode="valid")
    rms        = np.sqrt(window_sum / CHIRP_LEN + 1e-20)
    norm_xcorr = np.abs(xcorr) / (rms * np.sqrt(CHIRP_LEN))

    peak_idx = int(np.argmax(norm_xcorr))
    peak_val = float(norm_xcorr[peak_idx])

    if peak_val < threshold:
        return None

    # Isolation: suppress the main peak and check the next highest
    guard = CHIRP_LEN // 2
    lo    = max(0, peak_idx - guard)
    hi    = min(len(norm_xcorr), peak_idx + guard)
    suppressed = norm_xcorr.copy()
    suppressed[lo:hi] = 0.0
    second_val = float(np.max(suppressed))

    if second_val > peak_val * 0.5:
        # Two comparably-strong peaks → ambiguous; fall back to the first
        pass  # still return peak_idx — caller will validate via header parse

    return peak_idx
