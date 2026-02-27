"""ABP v1 — cryptographic primitives.

  KDF:  Argon2id  (argon2-cffi)  → 32-byte key
  AEAD: XChaCha20-Poly1305       (PyNaCl raw bindings)

Key derivation parameters are defined in profiles.py and locked for v1.

Conventions:
  - passphrase : str  (UTF-8 text; NOT bytes — normalisation is caller's job)
  - salt       : bytes, exactly 32 bytes  (from frame header)
  - nonce      : bytes, exactly 24 bytes  (from frame header)
  - aad        : bytes  — additional authenticated data; must be the 76-byte
                          packed frame header so the AEAD tag covers the header.
"""

import hashlib
import os

from argon2.low_level import Type, hash_secret_raw
from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_encrypt,
    crypto_aead_xchacha20poly1305_ietf_decrypt,
    crypto_aead_xchacha20poly1305_ietf_KEYBYTES,    # 32
    crypto_aead_xchacha20poly1305_ietf_NPUBBYTES,   # 24
)

from .profiles import (
    ARGON2_TIME_COST,
    ARGON2_MEMORY_COST,
    ARGON2_PARALLELISM,
    ARGON2_HASH_LEN,
)

# Sanity-check library constants against what we hard-coded in profiles
assert crypto_aead_xchacha20poly1305_ietf_KEYBYTES  == ARGON2_HASH_LEN == 32
assert crypto_aead_xchacha20poly1305_ietf_NPUBBYTES == 24

SALT_LEN  = 32   # matches header salt field
NONCE_LEN = 24   # matches header nonce field
TAG_LEN   = 16   # Poly1305 tag appended by AEAD encrypt
SHA256_LEN = 32  # integrity suffix used in plaintext mode


# ── random material generation ────────────────────────────────────────────────

def new_salt() -> bytes:
    """Return 32 cryptographically-random bytes for Argon2id salt."""
    return os.urandom(SALT_LEN)


def new_nonce() -> bytes:
    """Return 24 cryptographically-random bytes for XChaCha20 nonce."""
    return os.urandom(NONCE_LEN)


# ── key derivation ────────────────────────────────────────────────────────────

def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte key from *passphrase* and *salt* using Argon2id.

    Parameters are locked to the values in profiles.py so v1 output is
    always reproducible from the same (passphrase, salt) pair.

    Args:
        passphrase: Unicode string (will be encoded as UTF-8).
        salt:       Exactly 32 bytes (from frame header).

    Returns:
        32-byte raw key suitable for XChaCha20-Poly1305.
    """
    if len(salt) != SALT_LEN:
        raise ValueError(f"salt must be {SALT_LEN} bytes, got {len(salt)}")

    return hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=ARGON2_MEMORY_COST,
        parallelism=ARGON2_PARALLELISM,
        hash_len=ARGON2_HASH_LEN,
        type=Type.ID,
    )


# ── AEAD encrypt / decrypt ────────────────────────────────────────────────────

def encrypt(plaintext: bytes, key: bytes, nonce: bytes, aad: bytes) -> bytes:
    """XChaCha20-Poly1305 authenticated encryption.

    The frame header (76 bytes) is passed as *aad* so the tag also
    authenticates header fields (version, flags, body_len, etc.).

    Returns ciphertext || 16-byte Poly1305 tag  (len = len(plaintext) + 16).
    """
    if len(key)   != 32: raise ValueError(f"key must be 32 bytes")
    if len(nonce) != 24: raise ValueError(f"nonce must be 24 bytes")

    return crypto_aead_xchacha20poly1305_ietf_encrypt(
        message=plaintext,
        aad=aad,
        nonce=nonce,
        key=key,
    )


def decrypt(ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes) -> bytes:
    """XChaCha20-Poly1305 authenticated decryption.

    Raises nacl.exceptions.CryptoError if the tag does not verify
    (wrong passphrase, corrupted ciphertext, or tampered header).

    Returns plaintext (len = len(ciphertext) - 16).
    """
    if len(key)   != 32: raise ValueError(f"key must be 32 bytes")
    if len(nonce) != 24: raise ValueError(f"nonce must be 24 bytes")
    if len(ciphertext) < TAG_LEN:
        raise ValueError("ciphertext shorter than AEAD tag length")

    return crypto_aead_xchacha20poly1305_ietf_decrypt(
        ciphertext=ciphertext,
        aad=aad,
        nonce=nonce,
        key=key,
    )


# ── plaintext-mode integrity ──────────────────────────────────────────────────

def append_integrity(data: bytes) -> bytes:
    """Append a 32-byte SHA-256 digest to *data* for unencrypted integrity check.

    Layout: data || SHA-256(data)
    """
    return data + hashlib.sha256(data).digest()


def verify_integrity(data_with_tag: bytes) -> bytes:
    """Verify and strip the 32-byte SHA-256 integrity suffix.

    Args:
        data_with_tag: payload || SHA-256(payload)

    Returns:
        The original payload (without suffix).

    Raises:
        ValueError if the digest does not match.
    """
    if len(data_with_tag) < SHA256_LEN:
        raise ValueError("Payload too short to contain integrity suffix")

    payload = data_with_tag[:-SHA256_LEN]
    tag     = data_with_tag[-SHA256_LEN:]
    expected = hashlib.sha256(payload).digest()

    if tag != expected:
        raise ValueError("Integrity check failed — payload has been modified")

    return payload
