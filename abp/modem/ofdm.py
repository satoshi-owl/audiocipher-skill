"""ABP v1 — OFDM modem (modulate / demodulate).

PHY summary
-----------
  Sample rate    : 48 000 Hz
  FFT size       : 1024      → bin width = 46.875 Hz
  Cyclic prefix  : 256 samples = 5.33 ms
  Symbol length  : 1280 samples = 26.67 ms
  Symbol rate    : 37.5 sym/s
  Band           : bins 17–76  (≈ 800–3600 Hz)
  Active carriers: 60  (50 data QPSK + 10 pilots BPSK = +1+0j)
  Bits/symbol    : 100  (50 × 2-bit QPSK)

Real-valued OFDM
----------------
We use numpy.fft.irfft / rfft so only positive-frequency bins need to be set.
irfft automatically enforces conjugate symmetry → real time-domain output.
Bin 0 (DC) and bins above BIN_HI are left at zero.

Encoding pipeline
-----------------
  bits (1-D uint8)
    → pad to multiple of BITS_PER_SYMBOL
    → reshape (n_symbols, BITS_PER_SYMBOL)
    → per symbol: QPSK → place in freq domain → irfft → add CP
    → concatenate → float32 audio

Decoding pipeline
-----------------
  float32 audio  (post-preamble slice, length = n_symbols × SYMBOL_LEN)
    → split into symbols (strip CP) → rfft → extract active bins
    → estimate channel from pilots → equalize → QPSK hard-decision
    → concatenate bits → trim to n_fec_bits → bytes
"""

import math

import numpy as np
from numpy.typing import NDArray

from ..profiles import (
    SR, FFT_N, CP_LEN, SYMBOL_LEN,
    BIN_LO, BIN_HI, N_ACTIVE,
    PILOT_LOCAL, DATA_LOCAL, N_DATA,
    BITS_PER_SYMBOL, PILOT_VALUE,
    OFDM_AMPLITUDE,
)
from .dsp import (
    qpsk_modulate, qpsk_demodulate,
    estimate_channel, equalize, estimate_snr,
    PILOT_ABS, DATA_ABS,
)

# Number of rfft output bins for FFT_N real input
_RFFT_BINS = FFT_N // 2 + 1   # 513


# ── modulate ──────────────────────────────────────────────────────────────────

def modulate(bits: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert a bit stream to OFDM audio samples.

    Args:
        bits: 1-D uint8 array (values 0 or 1).  Will be padded with zeros to
              the next multiple of BITS_PER_SYMBOL.

    Returns:
        float32 array of length n_symbols × SYMBOL_LEN.
    """
    # Pad to symbol boundary
    n_pad    = (-len(bits)) % BITS_PER_SYMBOL
    bits_pad = np.concatenate([bits, np.zeros(n_pad, dtype=np.uint8)])
    n_sym    = len(bits_pad) // BITS_PER_SYMBOL

    output = np.empty(n_sym * SYMBOL_LEN, dtype=np.float64)

    for i in range(n_sym):
        sym_bits  = bits_pad[i * BITS_PER_SYMBOL : (i + 1) * BITS_PER_SYMBOL]
        # Map bits to QPSK constellation (N_DATA symbols)
        data_syms = qpsk_modulate(sym_bits)          # shape (N_DATA,)

        # Build frequency-domain frame (rfft output: 513 complex bins)
        freq = np.zeros(_RFFT_BINS, dtype=complex)

        # Insert pilot symbols at pilot absolute bins
        for local_i, abs_bin in zip(PILOT_LOCAL, PILOT_ABS):
            freq[abs_bin] = PILOT_VALUE

        # Insert data symbols at data absolute bins
        for d_i, abs_bin in enumerate(DATA_ABS):
            freq[abs_bin] = data_syms[d_i]

        # IFFT → real time-domain symbol (FFT_N samples)
        time = np.fft.irfft(freq, n=FFT_N)

        # Normalise each symbol to OFDM_AMPLITUDE
        peak = np.max(np.abs(time))
        if peak > 1e-9:
            time *= OFDM_AMPLITUDE / peak

        # Add cyclic prefix (last CP_LEN samples prepended)
        symbol = np.concatenate([time[-CP_LEN:], time])   # length SYMBOL_LEN

        output[i * SYMBOL_LEN : (i + 1) * SYMBOL_LEN] = symbol

    return output.astype(np.float32)


# ── demodulate ────────────────────────────────────────────────────────────────

def demodulate(
    samples: NDArray[np.float32],
) -> tuple[NDArray[np.uint8], int, float | None]:
    """Demodulate OFDM audio samples to a bit stream.

    The number of OFDM symbols is inferred from len(samples) // SYMBOL_LEN.
    Any trailing samples (< SYMBOL_LEN) are ignored.

    Args:
        samples: float32 audio, post-preamble, length ≈ n_symbols × SYMBOL_LEN.

    Returns:
        (bits, n_symbols_decoded, mean_snr_db)
        bits: 1-D uint8 array of length n_symbols × BITS_PER_SYMBOL.
        n_symbols_decoded: number of complete symbols processed.
        mean_snr_db: average per-symbol SNR estimate (None if 0 symbols).
    """
    n_sym = len(samples) // SYMBOL_LEN
    if n_sym == 0:
        return np.array([], dtype=np.uint8), 0, None

    all_bits = np.empty(n_sym * BITS_PER_SYMBOL, dtype=np.uint8)
    snr_sum  = 0.0

    for i in range(n_sym):
        raw    = samples[i * SYMBOL_LEN : (i + 1) * SYMBOL_LEN]

        # Strip cyclic prefix
        body   = raw[CP_LEN:]                          # FFT_N samples

        # FFT → frequency domain
        freq   = np.fft.rfft(body, n=FFT_N)            # 513 complex bins

        # Extract all active bins
        active = freq[BIN_LO : BIN_HI + 1]             # 60 complex values

        # Channel estimation from pilots
        H      = estimate_channel(active)

        # Zero-force equalisation
        eq     = equalize(active, H)

        # SNR estimate
        snr_sum += estimate_snr(active, H)

        # Extract data subcarriers (local indices from DATA_LOCAL)
        data_eq = eq[DATA_LOCAL]                        # 50 complex QPSK symbols

        # QPSK hard decision → 100 bits
        bits_sym = qpsk_demodulate(data_eq)
        all_bits[i * BITS_PER_SYMBOL : (i + 1) * BITS_PER_SYMBOL] = bits_sym

    mean_snr = snr_sum / n_sym if n_sym > 0 else None
    return all_bits, n_sym, mean_snr


# ── bit ↔ byte helpers ────────────────────────────────────────────────────────

def bits_to_bytes(bits: NDArray[np.uint8]) -> bytes:
    """Pack a 1-D bit array (MSB first per byte) into bytes.

    Trailing bits that don't fill a full byte are discarded.
    """
    n_bytes = len(bits) // 8
    packed  = np.packbits(bits[:n_bytes * 8])
    return packed.tobytes()


def bytes_to_bits(data: bytes) -> NDArray[np.uint8]:
    """Unpack bytes to a 1-D bit array (MSB first per byte)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# ── frame count helpers ───────────────────────────────────────────────────────

def n_symbols_for_bytes(n_bytes: int) -> int:
    """Number of OFDM symbols needed to carry *n_bytes* bytes."""
    n_bits = n_bytes * 8
    return math.ceil(n_bits / BITS_PER_SYMBOL)


def audio_duration_s(n_fec_bytes: int) -> float:
    """Total audio duration (seconds) for a given FEC-encoded byte count."""
    n_sym = n_symbols_for_bytes(n_fec_bytes)
    return n_sym * SYMBOL_LEN / SR
