"""ABP v1 — 76-byte fixed frame header.

Layout (all multi-byte integers little-endian):
  [0:4]   magic          = 0xAC 0xAB 0x01 0x00
  [4]     version        = 0x01
  [5]     flags          = bit0:ENCRYPTED | bit1:COMPRESSED | bits2-7:reserved
  [6]     crypto_suite   = 0x00:none | 0x01:XChaCha20-Poly1305+Argon2id
  [7]     compress_algo  = 0x00:none | 0x01:zstd
  [8]     compress_level (profile-derived, informational)
  [9:12]  reserved       = 0x00 0x00 0x00
  [12:16] orig_len       uint32 LE — original plaintext byte count
  [16:20] body_len       uint32 LE — protected payload byte count (after encrypt/compress)
  [20:52] salt           32 bytes  — Argon2id salt (zeros if not encrypted)
  [52:76] nonce          24 bytes  — XChaCha20 nonce (zeros if not encrypted)
  ── 76 bytes total ──
"""

import struct
from dataclasses import dataclass, field

from .profiles import (
    MAGIC, VERSION, HEADER_LEN,
    FLAG_ENCRYPTED, FLAG_COMPRESSED,
    CRYPTO_NONE, COMPRESS_NONE,
)

# struct format: <4s B  B     B            B             B              3s       I        I        32s   24s
# Field:          mag  ver  flags  crypto_suite  compress_algo  compress_level  reserved  orig_len  body_len  salt   nonce
# Sizes:           4   1    1      1             1              1               3         4         4         32     24  = 76
_STRUCT = struct.Struct("<4sBBBBB3sII32s24s")
assert _STRUCT.size == 76, f"Header struct size mismatch: {_STRUCT.size}"


@dataclass
class ABPFrame:
    """Parsed representation of the 76-byte ABP frame header."""

    # Required fields with defaults that produce a valid "empty" frame
    flags:          int   = 0x00
    crypto_suite:   int   = CRYPTO_NONE
    compress_algo:  int   = COMPRESS_NONE
    compress_level: int   = 0x00
    orig_len:       int   = 0
    body_len:       int   = 0
    salt:           bytes = field(default_factory=lambda: bytes(32))
    nonce:          bytes = field(default_factory=lambda: bytes(24))

    # ── derived helpers ───────────────────────────────────────────────────────

    @property
    def encrypted(self) -> bool:
        return bool(self.flags & FLAG_ENCRYPTED)

    @property
    def compressed(self) -> bool:
        return bool(self.flags & FLAG_COMPRESSED)

    # ── pack / unpack ─────────────────────────────────────────────────────────

    def pack(self) -> bytes:
        """Serialise to a 76-byte bytes object."""
        if len(self.salt) != 32:
            raise ValueError(f"salt must be 32 bytes, got {len(self.salt)}")
        if len(self.nonce) != 24:
            raise ValueError(f"nonce must be 24 bytes, got {len(self.nonce)}")

        return _STRUCT.pack(
            MAGIC,
            VERSION,
            self.flags,
            self.crypto_suite,
            self.compress_algo,
            self.compress_level,
            bytes(3),          # reserved
            self.orig_len,
            self.body_len,
            self.salt,
            self.nonce,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ABPFrame":
        """Deserialise from the first 76 bytes of *data*.

        Raises ValueError on bad magic or unsupported version.
        """
        if len(data) < HEADER_LEN:
            raise ValueError(
                f"Data too short for ABP header: {len(data)} < {HEADER_LEN}"
            )

        (
            magic,
            version,
            flags,
            crypto_suite,
            compress_algo,
            compress_level,
            _reserved,
            orig_len,
            body_len,
            salt,
            nonce,
        ) = _STRUCT.unpack(data[:HEADER_LEN])

        if magic != MAGIC:
            raise ValueError(
                f"Bad ABP magic: {magic.hex()} (expected {MAGIC.hex()})"
            )
        if version != VERSION:
            raise ValueError(
                f"Unsupported ABP version: {version} (expected {VERSION})"
            )

        return cls(
            flags=flags,
            crypto_suite=crypto_suite,
            compress_algo=compress_algo,
            compress_level=compress_level,
            orig_len=orig_len,
            body_len=body_len,
            salt=salt,
            nonce=nonce,
        )

    def __repr__(self) -> str:
        enc  = "enc"  if self.encrypted  else "plain"
        comp = "zstd" if self.compressed else "raw"
        return (
            f"ABPFrame(orig={self.orig_len}B body={self.body_len}B "
            f"{enc} {comp} salt={self.salt[:4].hex()}… nonce={self.nonce[:4].hex()}…)"
        )
