"""ABP v1 — Reed-Solomon FEC + block interleaver.

RS codec
--------
Uses `reedsolo.RSCodec(nsym)` over GF(2^8):
  social_safe: RSCodec(32) → RS(255, 223), corrects ≤ 16 byte errors / block
  fast:        RSCodec(16) → RS(255, 239), corrects ≤  8 byte errors / block

Input is pre-padded to a multiple of k bytes so reedsolo always produces
complete 255-byte codewords.  Zero-padding is stripped on decode via orig_len.

Block interleaver
-----------------
RS-encoded bytes are processed in groups of min(depth, n_blocks_in_group) blocks.
Within each group the blocks are treated as a G×BLOCK_SIZE matrix (G = effective
depth for that group) and written out column by column (interleaved).  The output
is the same total length as the input — no expansion, no trimming.

Burst tolerance per group:
  errors corrected per block = ceil(B / G) ≤ floor(nsym / 2)
  → max burst B = G × floor(nsym / 2)

  social_safe full group (G=8):  8 × 16 = 128-byte burst
  fast        full group (G=4):  4 ×  8 =  32-byte burst

  For the last (partial) group, effective G = remaining blocks < depth,
  so burst tolerance is proportionally lower — acceptable for v1.
"""

import math
from typing import Tuple

import numpy as np
from reedsolo import RSCodec, ReedSolomonError

from .profiles import PROFILES

BLOCK_SIZE = 255   # RS codeword length for GF(2^8)


# ── codec factory ─────────────────────────────────────────────────────────────

def _make_rsc(profile: dict) -> RSCodec:
    nsym = profile["rs_n"] - profile["rs_k"]
    return RSCodec(nsym)


# ── RS encode ────────────────────────────────────────────────────────────────

def rs_encode(data: bytes, profile: dict) -> bytes:
    """RS-encode *data* → exactly n_blocks × 255 bytes.

    Pre-pads *data* to a multiple of k so every reedsolo chunk is k bytes
    and every encoded block is exactly BLOCK_SIZE bytes.
    """
    rsc = _make_rsc(profile)
    k   = profile["rs_k"]

    pad    = (-len(data)) % k
    padded = data + bytes(pad)

    encoded = bytes(rsc.encode(padded))
    assert len(encoded) % BLOCK_SIZE == 0
    return encoded


def rs_decode(encoded: bytes, orig_len: int, profile: dict) -> Tuple[bytes, int]:
    """RS-decode *encoded* → up to *orig_len* plaintext bytes.

    Returns (decoded_bytes[:orig_len], n_corrections).
    Raises ReedSolomonError if any block has too many errors.
    """
    rsc           = _make_rsc(profile)
    n_blocks      = len(encoded) // BLOCK_SIZE
    n_corrections = 0
    parts         = []

    for i in range(n_blocks):
        block = encoded[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
        dec, _, errata = rsc.decode(block)
        n_corrections += len(errata) if errata else 0
        parts.append(bytes(dec))

    raw = b"".join(parts)
    return raw[:orig_len], n_corrections


# ── block interleaver ────────────────────────────────────────────────────────

def _interleave_group(group_bytes: bytes) -> bytes:
    """Interleave a group of G complete RS blocks (G×BLOCK_SIZE bytes).

    Treats data as shape (G, BLOCK_SIZE), reads column by column.
    Output length == input length.
    """
    g = len(group_bytes) // BLOCK_SIZE
    assert g >= 2
    arr = np.frombuffer(group_bytes, dtype=np.uint8).reshape(g, BLOCK_SIZE)
    return arr.T.flatten().tobytes()


def _deinterleave_group(il_bytes: bytes) -> bytes:
    """Inverse of _interleave_group.

    Input shape is (BLOCK_SIZE, G) in column-major; transpose back.
    """
    g = len(il_bytes) // BLOCK_SIZE
    assert g >= 2
    arr = np.frombuffer(il_bytes, dtype=np.uint8).reshape(BLOCK_SIZE, g)
    return arr.T.flatten().tobytes()


def interleave(encoded: bytes, depth: int) -> bytes:
    """Interleave RS-encoded bytes in groups of up to *depth* blocks.

    The output is the same length as *encoded*.  No padding or trimming.
    Groups with only 1 block are passed through unchanged.
    """
    if depth <= 1:
        return encoded

    n_blocks = len(encoded) // BLOCK_SIZE
    result   = bytearray()

    for g in range(math.ceil(n_blocks / depth)):
        lo   = g * depth * BLOCK_SIZE
        hi   = min(lo + depth * BLOCK_SIZE, len(encoded))
        grp  = encoded[lo:hi]
        n_g  = len(grp) // BLOCK_SIZE
        if n_g <= 1:
            result.extend(grp)
        else:
            result.extend(_interleave_group(grp))

    return bytes(result)


def deinterleave(interleaved: bytes, depth: int) -> bytes:
    """Inverse of interleave(). Must use the same depth."""
    if depth <= 1:
        return interleaved

    n_blocks = len(interleaved) // BLOCK_SIZE
    result   = bytearray()

    for g in range(math.ceil(n_blocks / depth)):
        lo   = g * depth * BLOCK_SIZE
        hi   = min(lo + depth * BLOCK_SIZE, len(interleaved))
        grp  = interleaved[lo:hi]
        n_g  = len(grp) // BLOCK_SIZE
        if n_g <= 1:
            result.extend(grp)
        else:
            result.extend(_deinterleave_group(grp))

    return bytes(result)


# ── convenience pipeline ──────────────────────────────────────────────────────

def fec_encode(payload: bytes, profile_name: str = "social_safe") -> bytes:
    """RS-encode then interleave *payload*.

    Returns interleaved bytes of length ceil(len(payload)/k) × BLOCK_SIZE.
    """
    prof    = PROFILES[profile_name]
    encoded = rs_encode(payload, prof)
    return interleave(encoded, prof["interleave_depth"])


def fec_decode(
    fec_bytes: bytes,
    orig_len: int,
    profile_name: str = "social_safe",
) -> Tuple[bytes, int]:
    """Deinterleave then RS-decode *fec_bytes*.

    Args:
        fec_bytes:    Interleaved FEC bytes (length multiple of BLOCK_SIZE).
        orig_len:     Trim target for decoded output; pass len(fec_bytes) to skip.
        profile_name: Must match the profile used during fec_encode.

    Returns: (decoded_bytes, n_corrections).
    Raises:  ReedSolomonError on uncorrectable block.
    """
    prof          = PROFILES[profile_name]
    deinterleaved = deinterleave(fec_bytes, prof["interleave_depth"])
    return rs_decode(deinterleaved, orig_len, prof)
