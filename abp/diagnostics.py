"""ABP v1 — decode result type and failure codes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FailureCode(str, Enum):
    """Reason a decode attempt did not produce plaintext."""

    OK                  = "ok"
    NO_SYNC             = "no_sync"             # chirp preamble not found
    HEADER_INVALID      = "header_invalid"      # bad magic / version / body_len
    FEC_UNCORRECTABLE   = "fec_uncorrectable"   # RS block has > nsym/2 errors
    INTEGRITY_FAIL      = "integrity_fail"      # SHA-256 suffix mismatch (plaintext mode)
    DECRYPT_FAIL        = "decrypt_fail"        # AEAD tag verification failed
    DECOMPRESS_FAIL     = "decompress_fail"     # zstd decompression error
    TRUNCATED           = "truncated"           # audio ended before all symbols decoded
    WRONG_PASSPHRASE    = "wrong_passphrase"    # alias for DECRYPT_FAIL (user-visible)


@dataclass
class DecodeResult:
    """Full decode outcome returned by :func:`abp.decode_audio`.

    On success  : ``success=True``,  ``text`` is the recovered plaintext.
    On failure  : ``success=False``, ``failure`` explains why, ``text`` is None.
    """

    success:          bool
    text:             Optional[str]    = None
    failure:          Optional[FailureCode] = None

    # Diagnostics — always populated even on failure (0 / None if unavailable)
    snr_db:           Optional[float]  = None    # estimated signal-to-noise in ABP band
    symbols_decoded:  int              = 0       # OFDM symbols successfully demodulated
    bytes_recovered:  int              = 0       # payload bytes recovered before FEC
    fec_corrections:  int              = 0       # byte-error positions corrected by RS

    # Human-readable summary for logging / CLI output
    def summary(self) -> str:
        if self.success:
            chars = len(self.text) if self.text else 0
            snr   = f"  SNR≈{self.snr_db:.1f}dB" if self.snr_db is not None else ""
            return (
                f"[OK] {chars} chars decoded  "
                f"sym={self.symbols_decoded} fec_corr={self.fec_corrections}{snr}"
            )
        return (
            f"[FAIL:{self.failure.value}]  "
            f"sym={self.symbols_decoded} recovered={self.bytes_recovered}B"
        )

    def __repr__(self) -> str:
        return f"DecodeResult({self.summary()})"
