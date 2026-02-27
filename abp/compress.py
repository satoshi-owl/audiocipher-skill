"""ABP v1 — zstd compression wrapper.

ABP uses zstd for all compression (legacy skill modes keep zlib).
Level 3 (social_safe) and level 1 (fast) are the only values used by profiles,
but the functions accept any valid zstd level (1–22).
"""

import zstandard as zstd


def compress(data: bytes, level: int = 3) -> bytes:
    """Compress *data* with zstd at *level*.

    Returns the compressed bytes.  Never raises on valid input.
    """
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress zstd-compressed *data*.

    Raises zstandard.ZstdError on corrupt input.
    """
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data)
