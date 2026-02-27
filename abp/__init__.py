"""ABP — Audio Binary Protocol v1.

Public API:
    encode_text(text, *, passphrase=None, profile="social_safe") -> np.ndarray
    decode_audio(samples, *, passphrase=None)                    -> DecodeResult
"""

from .api import encode_text, decode_audio
from .diagnostics import DecodeResult, FailureCode

__version__ = "1.0.0"
__all__ = ["encode_text", "decode_audio", "DecodeResult", "FailureCode"]
