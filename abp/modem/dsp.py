"""ABP v1 — DSP helpers: channel estimation, equalization, SNR.

Channel model assumed
---------------------
The channel applies a complex multiplicative distortion H[k] to each
subcarrier k.  Pilots (known symbols at known positions) let us sample H at
pilot frequencies; we interpolate to obtain H at data subcarrier frequencies.

All operations are vectorised over a single OFDM symbol (60-element arrays).
"""

import numpy as np
from numpy.typing import NDArray

from ..profiles import (
    BIN_LO, BIN_HI, N_ACTIVE,
    PILOT_LOCAL, DATA_LOCAL,
    N_PILOTS, N_DATA,
    PILOT_VALUE,
)

# Pre-compute pilot / data absolute bin indices once at import time
PILOT_ABS  = np.array([BIN_LO + i for i in PILOT_LOCAL], dtype=np.intp)
DATA_ABS   = np.array([BIN_LO + i for i in DATA_LOCAL],  dtype=np.intp)
ALL_LOCAL  = np.arange(N_ACTIVE)


# ── channel estimation ────────────────────────────────────────────────────────

def estimate_channel(
    received_active: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Estimate the per-subcarrier channel response from pilots.

    Args:
        received_active: Complex array of shape (N_ACTIVE,) — the received
                         frequency-domain values at all active subcarriers
                         (indices BIN_LO..BIN_HI in the full FFT).

    Returns:
        H: Complex array of shape (N_ACTIVE,) — estimated channel response.
           H[k] ≈ 1.0 in the absence of channel distortion.
    """
    # Channel estimate at pilot positions: H_pilot[i] = received[pilot_i] / expected
    pilot_received = received_active[PILOT_LOCAL]
    H_pilots       = pilot_received / PILOT_VALUE   # shape (N_PILOTS,)

    # Linear interpolation of real and imaginary parts separately.
    # xp: pilot local indices;  x: all local indices 0..N_ACTIVE-1
    xp = np.array(PILOT_LOCAL, dtype=float)
    x  = ALL_LOCAL.astype(float)

    H_real = np.interp(x, xp, H_pilots.real)
    H_imag = np.interp(x, xp, H_pilots.imag)

    return H_real + 1j * H_imag


def equalize(
    received_active: NDArray[np.complexfloating],
    H: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Zero-force equalise all active subcarriers using channel estimate H.

    Divides received[k] by H[k].  Subcarriers where |H[k]| < 0.05 are
    zeroed rather than amplified (noise magnification guard).

    Args:
        received_active: Complex array (N_ACTIVE,).
        H:               Channel estimate from :func:`estimate_channel`.

    Returns:
        Equalised complex array (N_ACTIVE,).
    """
    mask      = np.abs(H) > 0.05
    equalised = np.where(mask, received_active / np.where(mask, H, 1.0), 0.0 + 0.0j)
    return equalised


# ── SNR estimation ────────────────────────────────────────────────────────────

def estimate_snr(
    received_active: NDArray[np.complexfloating],
    H: NDArray[np.complexfloating],
) -> float:
    """Estimate SNR (dB) in the ABP band using pilot residuals.

    Signal power:  mean(|H_pilots × PILOT_VALUE|^2)
    Noise power:   mean(|received_pilots - H_pilots × PILOT_VALUE|^2)

    A well-conditioned channel at 0 dB SNR would give ~0 dB; AAC typically
    leaves >20 dB SNR in-band.

    Returns +inf if the noise power is essentially zero.
    """
    pilot_rx   = received_active[PILOT_LOCAL]
    H_at_pilot = H[PILOT_LOCAL]

    signal     = H_at_pilot * PILOT_VALUE         # what we expect
    noise      = pilot_rx - signal                 # residual

    sig_pwr    = float(np.mean(np.abs(signal) ** 2)) + 1e-20
    noise_pwr  = float(np.mean(np.abs(noise)  ** 2)) + 1e-20

    return 10.0 * np.log10(sig_pwr / noise_pwr)


# ── QPSK constellation ────────────────────────────────────────────────────────

# QPSK constellation: index = b0*2 + b1 (natural binary)
#   00 → (1+1j)/√2,   01 → (-1+1j)/√2
#   10 → (-1-1j)/√2,  11 → (1-1j)/√2
# QPSK_BITS[idx] must match the inverse of idx = b0*2+b1, i.e. bits [b0, b1].
QPSK_TABLE = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex) / np.sqrt(2)
QPSK_BITS  = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.uint8)  # idx = b0*2+b1


def qpsk_modulate(bits: NDArray[np.uint8]) -> NDArray[np.complexfloating]:
    """Map pairs of bits to QPSK symbols.

    Args:
        bits: 1-D uint8 array of length 2*N — must be even.

    Returns:
        Complex array of length N.
    """
    assert len(bits) % 2 == 0, "bit count must be even for QPSK"
    n      = len(bits) // 2
    pairs  = bits.reshape(n, 2)
    # Dibit → index: b[0]*2 + b[1]
    idx    = pairs[:, 0].astype(np.intp) * 2 + pairs[:, 1].astype(np.intp)
    return QPSK_TABLE[idx]


def qpsk_demodulate(symbols: NDArray[np.complexfloating]) -> NDArray[np.uint8]:
    """Hard-decision QPSK demodulation — minimum Euclidean distance.

    Args:
        symbols: Complex array of length N.

    Returns:
        uint8 array of length 2*N.
    """
    # Broadcast distance: (N, 1) vs (1, 4)
    dist   = np.abs(symbols[:, np.newaxis] - QPSK_TABLE[np.newaxis, :])
    idx    = np.argmin(dist, axis=1)          # shape (N,)
    bits   = QPSK_BITS[idx]                   # shape (N, 2)
    return bits.flatten()
