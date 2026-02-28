"""ABP v1 — all compile-time constants, keyed in one place.

Nothing here is computed at runtime except the derived lists at the bottom.
Change a value here and it propagates everywhere.
"""

# ── Argon2id KDF ──────────────────────────────────────────────────────────────
# OWASP "interactive" tier, tuned for ~0.5 s on a 4-core desktop.
# These are LOCKED for v1 — changing them breaks compatibility with existing ciphers.
ARGON2_TIME_COST   = 3
ARGON2_MEMORY_COST = 65536   # KiB = 64 MiB
ARGON2_PARALLELISM = 4
ARGON2_HASH_LEN    = 32      # bytes → 256-bit key for XChaCha20-Poly1305

# ── Frame header ──────────────────────────────────────────────────────────────
MAGIC      = bytes([0xAC, 0xAB, 0x01, 0x00])
VERSION    = 0x01
HEADER_LEN = 76   # bytes — fixed for v1

# flags byte bitmask
FLAG_ENCRYPTED  = 0x01
FLAG_COMPRESSED = 0x02

# crypto_suite values
CRYPTO_NONE               = 0x00
CRYPTO_XCHACHA20_ARGON2ID = 0x01   # v1 symmetric; X25519 reserved for 0x02+

# compress_algo values
COMPRESS_NONE = 0x00
COMPRESS_ZSTD = 0x01

# ── FEC + interleaving profiles ───────────────────────────────────────────────
# interleave_depth: number of RS(255,k) blocks interleaved together.
# A depth-D interleave converts a burst error of ≤ D×(nsym//2) bytes
# into ≤ nsym//2 scattered byte errors per RS block — exactly what RS can correct.
PROFILES: dict[str, dict] = {
    "social_safe": {
        "rs_n":             255,
        "rs_k":             223,   # RS(255,223): nsym=32, corrects ≤16 byte errors
        "interleave_depth": 8,     # burst tolerance: 8 × 16 = 128 bytes
        "compress_level":   3,     # zstd level
    },
    "fast": {
        "rs_n":             255,
        "rs_k":             239,   # RS(255,239): nsym=16, corrects ≤8 byte errors
        "interleave_depth": 4,     # burst tolerance: 4 × 8 = 32 bytes
        "compress_level":   1,
    },
    # browser_safe: same strong FEC as social_safe but zstd compression is disabled.
    # Use this when the recipient will decode in the AudioCipher browser app, which
    # does not ship a zstd decompressor.  Longer WAV file, but fully browser-decodable.
    "browser_safe": {
        "rs_n":             255,
        "rs_k":             223,   # RS(255,223): nsym=32, corrects ≤16 byte errors
        "interleave_depth": 8,     # burst tolerance: 8 × 16 = 128 bytes
        "compress_level":   0,     # informational only — compress=False disables zstd
        "compress":         False, # ← skip zstd so browser can decode without WASM
    },
}
DEFAULT_PROFILE = "social_safe"

# ── OFDM PHY ──────────────────────────────────────────────────────────────────
SR         = 48_000          # sample rate (Hz)
FFT_N      = 1024            # FFT size → bin width = SR/FFT_N = 46.875 Hz
CP_LEN     = 256             # cyclic prefix length (samples) = 5.33 ms
SYMBOL_LEN = FFT_N + CP_LEN  # 1280 samples = 26.67 ms per OFDM symbol

# Band: 800–3600 Hz (telephone voice passband, AAC-safe mid-band)
#   bin 17 → 17 × 46.875 = 796.875 Hz ≈ 800 Hz
#   bin 76 → 76 × 46.875 = 3562.5  Hz ≈ 3600 Hz
BIN_LO   = 17
BIN_HI   = 76
N_ACTIVE = BIN_HI - BIN_LO + 1   # 60 active bins

# Pilot placement: every PILOT_PERIOD-th active carrier (0-indexed local offset)
# Local indices:  0  6  12  18  24  30  36  42  48  54  → 10 pilots
PILOT_PERIOD = 6
PILOT_LOCAL  = list(range(0, N_ACTIVE, PILOT_PERIOD))   # [0, 6, 12, ... 54]
N_PILOTS     = len(PILOT_LOCAL)                          # 10
DATA_LOCAL   = [i for i in range(N_ACTIVE) if i not in set(PILOT_LOCAL)]  # 50
N_DATA       = len(DATA_LOCAL)                           # 50
BITS_PER_SYMBOL = N_DATA * 2                             # 100 bits/symbol (QPSK)

# Pilot value: constant +1+0j — gives clean channel phase reference
PILOT_VALUE  = complex(1, 0)

# Target symbol rate: 37.5 sym/s → raw bitrate ≈ 3750 bits/s ≈ 468.75 B/s raw
# After RS(255,223): ≈ 410 B/s net.

# ── Preamble (chirp sync) ─────────────────────────────────────────────────────
CHIRP_DURATION = 0.5                           # seconds
CHIRP_LEN      = int(SR * CHIRP_DURATION)      # 24 000 samples
CHIRP_F0       = float(BIN_LO * SR / FFT_N)   # 796.875 Hz — aligned to BIN_LO
CHIRP_F1       = float(BIN_HI * SR / FFT_N)   # 3562.5  Hz — aligned to BIN_HI
CHIRP_AMP      = 1.0                           # chirp amplitude — raised for OGG Opus survival

# Normalised cross-correlation threshold for sync detection.
# 0.25 survived AAC 128k in bench testing; 0.15 is the hard floor.
SYNC_THRESHOLD = 0.08   # lowered from 0.25 — calibrated for OGG Opus 32-64 kbps (Telegram voice)

# ── Output audio ──────────────────────────────────────────────────────────────
POST_ROLL_LEN   = int(SR * 0.25)   # 12 000 samples of trailing silence
OFDM_AMPLITUDE  = 0.70             # peak amplitude of OFDM output (pre-normalise)
