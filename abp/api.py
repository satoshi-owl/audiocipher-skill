"""ABP v1 — high-level encode / decode API.

encode_text(text, *, passphrase, profile) -> np.ndarray  (float32 audio at 48 kHz)
decode_audio(samples, *, passphrase)      -> DecodeResult

Full pipeline
=============

Encode
------
  text
    → UTF-8 bytes                                    (orig_len)
    → [optional] zstd compress                       (compress.compress)
    → build ABPFrame header (76 bytes)
    → [if passphrase] Argon2id KDF + XChaCha20-Poly1305 encrypt
      [else]          SHA-256 integrity suffix append
    → payload = header || body
    → RS FEC encode + block interleave               (fec.fec_encode)
    → bit stream → QPSK → OFDM modulate              (ofdm.modulate)
    → prepend chirp preamble + append post-roll silence

Decode
------
  float32 samples
    → chirp sync detect                              (sync.detect)
    → slice post-preamble data samples
    → OFDM demodulate → bit stream                  (ofdm.demodulate)
    → bit stream → bytes (trim to complete RS blocks)
    → RS FEC deinterleave + decode                   (fec.fec_decode)
    → parse ABPFrame header (first HEADER_LEN bytes)
    → [if encrypted] Argon2id KDF + AEAD decrypt
      [else]         SHA-256 integrity verify
    → [if compressed] zstd decompress
    → UTF-8 decode → DecodeResult
"""

import os
import unicodedata

import numpy as np
import scipy.signal as _spsig
from numpy.typing import NDArray
from reedsolo import ReedSolomonError

from .compress import compress, decompress
from .crypto import (
    derive_key, encrypt, decrypt,
    append_integrity, verify_integrity,
    new_salt, new_nonce,
    TAG_LEN,
)
from .diagnostics import DecodeResult, FailureCode
from .fec import fec_encode, fec_decode, BLOCK_SIZE
from .framing import ABPFrame, HEADER_LEN
from .modem.ofdm import (
    modulate, demodulate,
    bytes_to_bits, bits_to_bytes,
    n_symbols_for_bytes,
)
from .modem.sync import gen_preamble, detect as chirp_detect
from .profiles import (
    SR, SYMBOL_LEN, CHIRP_LEN, POST_ROLL_LEN,
    FLAG_ENCRYPTED, FLAG_COMPRESSED,
    CRYPTO_NONE, CRYPTO_XCHACHA20_ARGON2ID,
    COMPRESS_NONE, COMPRESS_ZSTD,
    PROFILES, DEFAULT_PROFILE,
    BITS_PER_SYMBOL,
)


# ── encode ────────────────────────────────────────────────────────────────────

def encode_text(
    text: str,
    *,
    passphrase: str | None = None,
    profile: str = DEFAULT_PROFILE,
) -> NDArray[np.float32]:
    """Encode *text* to float32 48 kHz mono audio samples.

    Args:
        text:       Plaintext to hide.  Will be NFC-normalised and UTF-8 encoded.
        passphrase: If given, encrypts with XChaCha20-Poly1305 (Argon2id KDF).
                    If None, appends a SHA-256 integrity tag instead.
        profile:    "social_safe" (default) or "fast".

    Returns:
        float32 ndarray at 48 kHz: [chirp preamble | OFDM data | post-roll silence].
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}': choose from {list(PROFILES)}")

    prof      = PROFILES[profile]
    encrypted = passphrase is not None

    # 1. Normalise + encode text
    text_norm = unicodedata.normalize("NFC", text)
    plaintext = text_norm.encode("utf-8")
    orig_len  = len(plaintext)

    # 2. Compress (skip if the profile sets compress=False — e.g. "browser_safe")
    compress_enabled = prof.get("compress", True)
    if compress_enabled and (encrypted or len(plaintext) > 64):
        body = compress(plaintext, level=prof["compress_level"])
        compressed = True
    else:
        body = plaintext
        compressed = False

    # 3. Pre-compute body_len so we can build the FINAL header before encrypting.
    #    (The AEAD tag covers the header as AAD — header and body_len must match.)
    flags = 0x00
    if encrypted:
        flags |= FLAG_ENCRYPTED
    if compressed:
        flags |= FLAG_COMPRESSED

    salt  = new_salt()  if encrypted else bytes(32)
    nonce = new_nonce() if encrypted else bytes(24)

    if encrypted:
        # ciphertext = plaintext || 16-byte Poly1305 tag
        body_len = len(body) + TAG_LEN
    else:
        # plaintext payload || SHA-256 integrity suffix (32 bytes)
        body_len = len(body) + 32   # crypto.SHA256_LEN

    # 4. Build FINAL frame header with correct body_len
    frame = ABPFrame(
        flags=flags,
        crypto_suite=CRYPTO_XCHACHA20_ARGON2ID if encrypted else CRYPTO_NONE,
        compress_algo=COMPRESS_ZSTD if compressed else COMPRESS_NONE,
        compress_level=prof["compress_level"],
        orig_len=orig_len,
        body_len=body_len,
        salt=salt,
        nonce=nonce,
    )
    header_bytes = frame.pack()
    assert len(header_bytes) == HEADER_LEN

    # 5. Encrypt (AAD = final header) or add integrity tag
    if encrypted:
        key  = derive_key(passphrase, salt)
        body = encrypt(body, key, nonce, aad=header_bytes)
    else:
        body = append_integrity(body)  # body || SHA-256(body)

    assert len(body) == body_len, f"body length mismatch: {len(body)} != {body_len}"

    # 6. Payload = header || body
    payload = header_bytes + body

    # 7. RS FEC encode + interleave
    fec_bytes = fec_encode(payload, profile)
    n_fec     = len(fec_bytes)   # multiple of BLOCK_SIZE (255)

    # 8. Convert to bits → OFDM modulate
    bits      = bytes_to_bits(fec_bytes)
    ofdm_pcm  = modulate(bits)   # float32, length = n_symbols × SYMBOL_LEN

    # 9. Assemble final audio: preamble + OFDM data + post-roll
    preamble  = gen_preamble()
    post_roll = np.zeros(POST_ROLL_LEN, dtype=np.float32)
    audio     = np.concatenate([preamble, ofdm_pcm, post_roll])

    return audio.astype(np.float32)


# ── decode ────────────────────────────────────────────────────────────────────

# Rate-correction hypotheses tried in order of decreasing real-world likelihood.
# Covers ±0.5% clock drift (≫ typical 50 ppm oscillator accuracy; handles
# aggressive resampling pipelines like Telegram→X).
_RATE_GRID = [
    1.000,           # nominal — no correction
    1.001, 0.999,    # ±0.1%
    1.002, 0.998,    # ±0.2%  ← full-model Telegram→X drift
    1.003, 0.997,    # ±0.3%
    1.004, 0.996,    # ±0.4%
    1.005, 0.995,    # ±0.5%
]

# FailureCodes that indicate a *transient* decode failure that may resolve
# with a different sample-rate hypothesis.  All other codes are definitive.
_TRANSIENT_FAILURES = frozenset({
    FailureCode.NO_SYNC,
    FailureCode.FEC_UNCORRECTABLE,
    FailureCode.TRUNCATED,
})


def decode_audio(
    samples: NDArray[np.float32],
    *,
    passphrase: str | None = None,
) -> DecodeResult:
    """Decode ABP audio to plaintext.

    Tries a small grid of sample-rate corrections (±0.5% in 0.1% steps)
    to handle clock drift introduced by streaming platforms such as
    Telegram and X.  The first successful decode is returned immediately.

    Args:
        samples:    float32 ndarray at SR (48 kHz).
        passphrase: Must match the one used during encode if the message is
                    encrypted.  Pass None for unencrypted messages.

    Returns:
        :class:`DecodeResult` — check ``.success`` before using ``.text``.
    """
    samples = np.asarray(samples, dtype=np.float32)
    last_result: DecodeResult | None = None

    for rate in _RATE_GRID:
        if rate == 1.0:
            s = samples
        else:
            new_n = max(1, int(round(len(samples) * rate)))
            s = _spsig.resample(samples, new_n).astype(np.float32)

        result = _decode_inner(s, passphrase=passphrase)

        if result.success:
            return result

        last_result = result

        # Definitive failures (wrong passphrase, corrupt header, etc.) — don't retry.
        if result.failure not in _TRANSIENT_FAILURES:
            return result

    return last_result or DecodeResult(success=False, failure=FailureCode.NO_SYNC)


def _decode_inner(
    samples: NDArray[np.float32],
    *,
    passphrase: str | None = None,
) -> DecodeResult:
    """Single-pass decode attempt at the given (already rate-corrected) sample rate."""
    samples = np.asarray(samples, dtype=np.float32)

    # ── 1. Chirp sync ─────────────────────────────────────────────────────────
    sync_offset = chirp_detect(samples)
    if sync_offset is None:
        return DecodeResult(success=False, failure=FailureCode.NO_SYNC)

    # Slice post-preamble data (everything after chirp + any AAC priming silence)
    data_start  = sync_offset + CHIRP_LEN
    data_samples = samples[data_start:]

    # ── 2. OFDM demodulate ────────────────────────────────────────────────────
    bits, n_sym, snr_db = demodulate(data_samples)
    if n_sym == 0:
        return DecodeResult(
            success=False,
            failure=FailureCode.TRUNCATED,
            snr_db=snr_db,
        )

    # ── 3. Bits → bytes → trim to complete RS blocks ──────────────────────────
    raw_bytes = bits_to_bytes(bits)
    n_blocks  = len(raw_bytes) // BLOCK_SIZE
    if n_blocks == 0:
        return DecodeResult(
            success=False,
            failure=FailureCode.TRUNCATED,
            symbols_decoded=n_sym,
            snr_db=snr_db,
        )
    fec_bytes = raw_bytes[:n_blocks * BLOCK_SIZE]

    # ── 4. RS FEC decode (deinterleave + RS decode) ───────────────────────────
    # We don't know the profile until we parse the header, but both profiles
    # use RS(255,k) with BLOCK_SIZE=255.  We need orig_len from the header to
    # trim reedsolo padding — use a two-pass approach:
    #   Pass A: decode enough bytes to parse the header (HEADER_LEN bytes).
    #           HEADER_LEN=76 fits in 1 RS block (k=223), so the first block
    #           always contains the full header.
    #   Pass B: decode the full payload using the profile from the header.
    #
    # For simplicity in v1 we decode everything with social_safe (the default)
    # and fall back to fast if that fails.  The profile is informational in the
    # header; we try both.

    payload       = None
    fec_corr      = 0
    used_profile  = None

    for try_profile in ("social_safe", "fast"):
        try:
            # Total bytes we expect: header + body.  We don't know body_len yet,
            # so pass n_blocks * BLOCK_SIZE as an upper bound for orig_len;
            # we'll trim after parsing the header.
            decoded, corrections = fec_decode(
                fec_bytes,
                orig_len=n_blocks * BLOCK_SIZE,
                profile_name=try_profile,
            )
            payload      = decoded
            fec_corr     = corrections
            used_profile = try_profile
            break
        except (ReedSolomonError, Exception):
            continue

    if payload is None:
        return DecodeResult(
            success=False,
            failure=FailureCode.FEC_UNCORRECTABLE,
            symbols_decoded=n_sym,
            bytes_recovered=len(fec_bytes),
            snr_db=snr_db,
        )

    # ── 5. Parse header ───────────────────────────────────────────────────────
    try:
        frame = ABPFrame.unpack(payload)
    except ValueError as exc:
        return DecodeResult(
            success=False,
            failure=FailureCode.HEADER_INVALID,
            symbols_decoded=n_sym,
            bytes_recovered=len(payload),
            fec_corrections=fec_corr,
            snr_db=snr_db,
        )

    # Sanity-check body_len
    expected_total = HEADER_LEN + frame.body_len
    if expected_total > len(payload):
        return DecodeResult(
            success=False,
            failure=FailureCode.TRUNCATED,
            symbols_decoded=n_sym,
            bytes_recovered=len(payload),
            fec_corrections=fec_corr,
            snr_db=snr_db,
        )

    body = payload[HEADER_LEN : HEADER_LEN + frame.body_len]

    # ── 6. Decrypt or verify integrity ────────────────────────────────────────
    if frame.encrypted:
        if passphrase is None:
            return DecodeResult(
                success=False,
                failure=FailureCode.DECRYPT_FAIL,
                symbols_decoded=n_sym,
                fec_corrections=fec_corr,
                snr_db=snr_db,
            )
        try:
            key  = derive_key(passphrase, frame.salt)
            # Rebuild header with body_len=0 placeholder — must match what encode used as AAD.
            # Encode used a header with body_len=0 as the AAD placeholder during key setup...
            # Actually encode computes the final header with correct body_len for AAD.
            # We have the correct frame, so use it directly.
            body = decrypt(body, key, frame.nonce, aad=frame.pack())
        except Exception:
            return DecodeResult(
                success=False,
                failure=FailureCode.DECRYPT_FAIL,
                symbols_decoded=n_sym,
                fec_corrections=fec_corr,
                snr_db=snr_db,
            )
    else:
        try:
            body = verify_integrity(body)
        except ValueError:
            return DecodeResult(
                success=False,
                failure=FailureCode.INTEGRITY_FAIL,
                symbols_decoded=n_sym,
                fec_corrections=fec_corr,
                snr_db=snr_db,
            )

    # ── 7. Decompress ─────────────────────────────────────────────────────────
    if frame.compressed:
        try:
            body = decompress(body)
        except Exception:
            return DecodeResult(
                success=False,
                failure=FailureCode.DECOMPRESS_FAIL,
                symbols_decoded=n_sym,
                fec_corrections=fec_corr,
                snr_db=snr_db,
            )

    # ── 8. UTF-8 decode ───────────────────────────────────────────────────────
    try:
        text = body[:frame.orig_len].decode("utf-8")
    except UnicodeDecodeError:
        return DecodeResult(
            success=False,
            failure=FailureCode.HEADER_INVALID,
            symbols_decoded=n_sym,
            fec_corrections=fec_corr,
            snr_db=snr_db,
        )

    return DecodeResult(
        success=True,
        text=text,
        failure=FailureCode.OK,
        snr_db=snr_db,
        symbols_decoded=n_sym,
        bytes_recovered=len(payload),
        fec_corrections=fec_corr,
    )
