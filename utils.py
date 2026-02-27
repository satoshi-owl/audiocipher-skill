"""
utils.py — Shared frequency maps, WAV I/O, FFT helpers
Ported from: app.html lines 2047–3617
"""
import struct
import json
import numpy as np

# ─────────────────────────────────────────────────────────────
# Frequency Maps (ported verbatim from app.html)
# ─────────────────────────────────────────────────────────────

HZALPHA_MAP = {
    'A':220.00,'B':233.08,'C':246.94,'D':261.63,'E':277.18,'F':293.66,'G':311.13,
    'H':329.63,'I':349.23,'J':369.99,'K':392.00,'L':415.30,'M':440.00,'N':466.16,
    'O':493.88,'P':523.25,'Q':554.37,'R':587.33,'S':622.25,'T':659.25,'U':698.46,
    'V':739.99,'W':783.99,'X':830.61,'Y':880.00,'Z':932.33,
    '0':987.77, '1':1046.50,'2':1108.73,'3':1174.66,'4':1244.51,
    '5':1318.51,'6':1396.91,'7':1480.00,'8':1567.98,'9':1661.22,
    ' ':0,
    '.':1760.00,'!':1864.66,',':1975.53,'?':2093.00,
    '-':2217.46,'_':2349.32,':':2489.02,';':2637.02,
    '/':2793.83,'\\':2959.96,'@':3135.96,'#':3322.44,
    '$':3520.00,'%':3729.31,'&':3951.07,'*':4186.01,
    '(':4434.92,')':4698.64,'+':4978.03,'=':5274.04,
    '"':5587.65,"'":5919.91,'[':6271.93,']':6644.88,
    '{':7040.00,'}':7458.62,'<':7902.13,'>':8372.02,
}

HZ_SHIFT_FREQ = 8869.84  # signals next letter is lowercase

MORSE_MAP = {
    'A':'.-',  'B':'-...','C':'-.-.','D':'-..', 'E':'.',   'F':'..-.',
    'G':'--.',  'H':'....','I':'..',  'J':'.---','K':'-.-', 'L':'.-..',
    'M':'--',   'N':'-.',  'O':'---', 'P':'.--.','Q':'--.-','R':'.-.',
    'S':'...',  'T':'-',   'U':'..-', 'V':'...-','W':'.--', 'X':'-..-',
    'Y':'-.--', 'Z':'--..',
    '0':'-----','1':'.----','2':'..---','3':'...--','4':'....-',
    '5':'.....','6':'-....','7':'--...','8':'---..','9':'----.',
    '.':'.-.-.-',',':'--..--','?':'..--..','!':'-.-.--',
    '/':'-..-.', '(':'-.--.', ')':'-.--.-',
    '&':'.-...',':':'---...',';':'-.-.-.','=':'-...-',
    '+':'.-.-.', '-':'-....-','_':'..--.-','"':'.-..-.',
    '$':'...-..-','@':'.--.-.', "'":'.----.'
}
MORSE_REVERSE = {v: k for k, v in MORSE_MAP.items()}

DTMF_KEY_MAP = {
    '1':[697,1209],'2':[697,1336],'3':[697,1477],'A':[697,1633],
    '4':[770,1209],'5':[770,1336],'6':[770,1477],'B':[770,1633],
    '7':[852,1209],'8':[852,1336],'9':[852,1477],'C':[852,1633],
    '*':[941,1209],'0':[941,1336],'#':[941,1477],'D':[941,1633],
}
DTMF_REVERSE = {tuple(v): k for k, v in DTMF_KEY_MAP.items()}

T9_MAP = {
    'A':'2','B':'22','C':'222','D':'3','E':'33','F':'333',
    'G':'4','H':'44','I':'444','J':'5','K':'55','L':'555',
    'M':'6','N':'66','O':'666','P':'7','Q':'77','R':'777','S':'7777',
    'T':'8','U':'88','V':'888','W':'9','X':'99','Y':'999','Z':'9999',
    ' ':'0',
}
T9_REVERSE = {}  # '2'→'A', '22'→'B', etc.
for _ch, _seq in T9_MAP.items():
    if _ch != ' ':
        T9_REVERSE[_seq] = _ch

FSK_F0   = 1000   # mark (bit 0)
FSK_F1   = 1200   # space (bit 1)
FSK_BAUD = 300    # bits per second (legacy JS default was 45; Python default is 300)

# WaveSig uses 100 Hz bin spacing (up from the original 46.875 Hz).
# 100 Hz is wide enough to survive AAC 128k and Opus 96k re-encoding
# (two codec hops, e.g. Telegram → Twitter) while keeping all 6×16 = 96
# frequency slots within 1000–10500 Hz — well inside AAC's reliable range.
GGWAVE_F0               = 1000    # start frequency (Hz)
GGWAVE_DF               = 100     # bin spacing (Hz)  — was 46.875 (too narrow for AAC)
GGWAVE_SAMPLES_PER_FRAME = 1024
GGWAVE_SAMPLE_RATE      = 48000
GGWAVE_FRAMES_PER_TX    = 9
GGWAVE_TONES_PER_FRAME  = 6
GGWAVE_MARKER_LO        = GGWAVE_F0 - 2 * GGWAVE_DF
GGWAVE_MARKER_HI        = GGWAVE_F0 + 98 * GGWAVE_DF


# ─────────────────────────────────────────────────────────────
# GF(16) Reed-Solomon (ported from app.html lines 2129–2248)
# RS(12,8): 8 data nibbles + 4 parity. Corrects up to 2 errors.
# ─────────────────────────────────────────────────────────────

class GF16:
    PRIM = 0x13
    SIZE = 16
    _exp = [0] * 32
    _log = [0] * 16

    @classmethod
    def _init(cls):
        # Build exp/log tables for GF(16) with primitive polynomial PRIM=0x13
        # (x^4 + x + 1).  After left-shifting, if bit 4 is set we reduce by
        # XORing with the full polynomial (0x13), NOT (PRIM & (SIZE-1) = 0x03).
        # Masking off the leading bit causes the result to stay ≥ 16 and then
        # _log[x] writes out-of-range — the classic GF reduction bug.
        x = 1
        for i in range(cls.SIZE - 1):
            cls._exp[i] = x
            cls._log[x] = i
            x <<= 1
            if x & cls.SIZE:           # bit 4 set → reduce mod PRIM
                x ^= cls.PRIM          # XOR full polynomial (keeps x in 0–15)
        cls._exp[cls.SIZE - 1] = 1
        for i in range(cls.SIZE, 32):
            cls._exp[i] = cls._exp[i - (cls.SIZE - 1)]

    @classmethod
    def mul(cls, a, b):
        if a == 0 or b == 0: return 0
        return cls._exp[cls._log[a] + cls._log[b]]

    @classmethod
    def add(cls, a, b):
        return a ^ b

    @classmethod
    def inv(cls, a):
        if a == 0: raise ValueError('GF16 inv(0)')
        return cls._exp[(cls.SIZE - 1 - cls._log[a]) % (cls.SIZE - 1)]

    @classmethod
    def _poly_mul(cls, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            for j, qj in enumerate(q):
                r[i + j] ^= cls.mul(pi, qj)
        return r

    @classmethod
    def _gen_poly(cls):
        g = [1]
        for i in range(4):
            g = cls._poly_mul(g, [cls._exp[i], 1])
        return g

    @classmethod
    def encode(cls, data8):
        """Append 4 parity nibbles to 8 data nibbles → 12 nibbles.

        Uses an LFSR (linear feedback shift register) to compute the remainder
        of (data_polynomial × x^4) mod g(x).  This is the standard systematic
        RS encoder: data nibbles are copied unchanged to positions [0..7], and
        parity nibbles go to [8..11].  All four syndromes of the result are
        guaranteed to be zero (valid codeword).

        The previous 'long-division in place' approach erroneously modified
        data positions during the loop, producing non-systematic codewords
        with non-zero syndromes, causing every round-trip to fail.
        """
        g = cls._gen_poly()   # g[0]=const term, g[4]=leading coeff=1
        # 4-stage LFSR; reg[0] feeds back first (highest-degree stage).
        reg = [0, 0, 0, 0]
        for d in data8:
            d = int(d) & 0xf
            feedback = d ^ reg[0]
            if feedback:
                reg[0] = reg[1] ^ cls.mul(feedback, g[3])
                reg[1] = reg[2] ^ cls.mul(feedback, g[2])
                reg[2] = reg[3] ^ cls.mul(feedback, g[1])
                reg[3] =          cls.mul(feedback, g[0])
            else:
                reg[0] = reg[1]
                reg[1] = reg[2]
                reg[2] = reg[3]
                reg[3] = 0
        return [int(d) & 0xf for d in data8] + reg

    @classmethod
    def decode(cls, received12):
        """Correct up to 2 errors in 12-nibble block → 8 data nibbles."""
        r = [x & 0xf for x in received12]
        S = []
        for i in range(4):
            val = 0
            for j in range(12):
                val = cls.add(cls.mul(val, cls._exp[i]), r[j])
            S.append(val)
        if all(s == 0 for s in S):
            return r[:8]
        # Berlekamp-Massey
        C, B, L, m, b = [1], [1], 0, 1, 1
        for n in range(4):
            d = S[n]
            for i in range(1, L + 1):
                d ^= cls.mul(C[i] if i < len(C) else 0, S[n - i])
            if d == 0:
                m += 1
                continue
            T = [0] * max(len(C), len(B) + m)
            for i in range(len(C)): T[i] ^= C[i]
            coeff = cls.mul(d, cls.inv(b))
            for i in range(len(B)): T[i + m] ^= cls.mul(coeff, B[i])
            if 2 * L <= n:
                B, L, b, m = C, n + 1 - L, d, 1
            else:
                m += 1
            C = T
        if L > 2:
            return r[:8]
        # Chien search
        err_locs = []
        for i in range(12):
            val = 0
            for j, cj in enumerate(C):
                val ^= cls.mul(cj, cls._exp[(i * j) % 15])
            if val == 0:
                err_locs.append(i)
        # Forney
        if len(err_locs) != L:
            return r[:8]
        for pos in err_locs:
            Xi = cls._exp[pos]
            num, denom = 0, 1
            for i in range(4):
                num ^= cls.mul(S[i], cls._exp[(pos * (i + 1)) % 15])
            for loc in err_locs:
                if loc != pos:
                    denom ^= cls.mul(Xi, cls._exp[loc])
            if denom == 0:
                return r[:8]
            r[pos] ^= cls.mul(num, cls.inv(denom))
        return r[:8]


GF16._init()


# ─────────────────────────────────────────────────────────────
# Parameterized RS Codec over GF(16) for arbitrary (n, k)
# Used by AcDense (RS(9,7)) — separate from the hardcoded GF16 class
# ─────────────────────────────────────────────────────────────

class RS_GF16:
    """
    Reed-Solomon codec over GF(16) for any valid (n, k) with n ≤ 15.
    Used by AcDense mode: RS(9, 7) — 7 data nibbles + 2 parity per frame.
      - corrects 1 symbol error per frame
      - 9 symbols total = ACDENSE_N_TONES (one RS block per frame)
    """

    def __init__(self, n: int, k: int):
        assert 1 <= k < n <= 15, f'RS({n},{k}) invalid for GF(16)'
        assert (n - k) % 2 == 0, f'n-k must be even (got n-k={n-k})'
        self.n        = n
        self.k        = k
        self.t        = (n - k) // 2   # error correction capability
        self._g       = self._make_gen_poly()

    def _make_gen_poly(self):
        """Generator polynomial: product of (x + α^i) for i = 0 .. n-k-1."""
        n_parity = self.n - self.k
        g = [1]
        for i in range(n_parity):
            g = GF16._poly_mul(g, [GF16._exp[i], 1])
        return g

    def encode(self, data):
        """
        Systematic RS encode.
        data: list of k GF(16) symbols (nibbles 0-15)
        returns: list of n symbols = data[:k] + parity
        """
        n_parity = self.n - self.k
        g = self._g
        reg = [0] * n_parity
        for d in data[:self.k]:
            d = int(d) & 0xF
            feedback = d ^ reg[0]
            if feedback:
                # In-place LFSR shift — safe because reg[i] reads reg[i+1]
                # which has not yet been overwritten (process left to right).
                for i in range(n_parity - 1):
                    reg[i] = reg[i + 1] ^ GF16.mul(feedback, g[n_parity - 1 - i])
                reg[n_parity - 1] = GF16.mul(feedback, g[0])
            else:
                for i in range(n_parity - 1):
                    reg[i] = reg[i + 1]
                reg[n_parity - 1] = 0
        return [int(d) & 0xF for d in data[:self.k]] + reg

    def decode(self, received):
        """
        RS decode. received: list of n GF(16) symbols.
        Returns corrected data[:k], or uncorrected data[:k] if > t errors.
        """
        n_parity = self.n - self.k
        r = [int(x) & 0xF for x in received[:self.n]]

        # Compute syndromes via Horner's method
        S = []
        for i in range(n_parity):
            val = 0
            for j in range(self.n):
                val = GF16.add(GF16.mul(val, GF16._exp[i]), r[j])
            S.append(val)

        if all(s == 0 for s in S):
            return r[:self.k]

        # Berlekamp-Massey
        C, B, L, m, b = [1], [1], 0, 1, 1
        for n_iter in range(n_parity):
            d = S[n_iter]
            for i in range(1, L + 1):
                d ^= GF16.mul(C[i] if i < len(C) else 0, S[n_iter - i])
            if d == 0:
                m += 1
                continue
            T = [0] * max(len(C), len(B) + m)
            for i in range(len(C)):
                T[i] ^= C[i]
            coeff = GF16.mul(d, GF16.inv(b))
            for i in range(len(B)):
                T[i + m] ^= GF16.mul(coeff, B[i])
            if 2 * L <= n_iter:
                B, L, b, m = C, n_iter + 1 - L, d, 1
            else:
                m += 1
            C = T

        if L > self.t:
            return r[:self.k]   # too many errors — return uncorrected

        # Chien search over the full GF(16) multiplicative group (order 15).
        #
        # With the Horner-reversed polynomial evaluation convention used here,
        # an error at array position pos_arr contributes:
        #   S_k = e * (alpha^k)^{n-1-pos_arr}   (Xi = alpha^{n-1-pos_arr})
        # The error locator sigma(z) = 1 + Xi*z has root at z = Xi^{-1}.
        # The Chien search evaluates sigma(alpha^chien_i) and finds the root
        # at chien_i where alpha^chien_i = Xi^{-1} = alpha^{pos_arr-(n-1)}.
        # This gives:   chien_i = (pos_arr - (n-1)) mod 15
        # Inverse:   pos_arr = (chien_i + (n-1)) mod 15
        #
        # We must search all 15 elements of GF(16)* (not just range(n)),
        # because for small n the valid roots may fall outside range(n).
        err_pairs: list[tuple[int, int]] = []   # [(pos_arr, chien_i), ...]
        for chien_i in range(15):
            actual_pos = (chien_i + self.n - 1) % 15
            if actual_pos >= self.n:
                continue    # root maps outside the valid codeword range
            val = 0
            for j, cj in enumerate(C):
                val ^= GF16.mul(cj, GF16._exp[(chien_i * j) % 15])
            if val == 0:
                err_pairs.append((actual_pos, chien_i))

        if len(err_pairs) != L:
            return r[:self.k]

        # Forney algorithm using the error evaluator polynomial Omega.
        #
        # Standard formula:  e_j = X_j * Omega(X_j^{-1}) / sigma'(X_j^{-1})
        #
        # where:
        #   X_j     = alpha^{n-1-p_j}  (error locator element, Xi in code)
        #   X_j^{-1} = alpha^{chien_i}  (Xi_inv)
        #   Omega(z) = S(z)*sigma(z) mod z^{n_parity}  (error evaluator)
        #   sigma'(z) = formal derivative of sigma (in GF(2^m): drop even-degree terms)
        #
        # NOTE: the naive formula  num = sum S[i]*alpha^{chien_i*(i+1)}  always
        # evaluates to 0 when n_parity is even, because S[i] = e*X^i causes
        # every pair of terms to cancel in GF(2^m).  Use Omega instead.

        # Step 1: compute Omega = conv(C, S) truncated to degree < n_parity
        Omega = [0] * n_parity
        for i in range(n_parity):
            for j in range(min(i + 1, len(C))):
                Omega[i] ^= GF16.mul(C[j], S[i - j])

        # Step 2: formal derivative of sigma (in GF(2^m), only odd-degree survive)
        #   sigma'[i] = C[i+1]  if  (i+1) is odd,  else 0
        sigma_prime = [
            C[i + 1] if (i + 1) < len(C) and (i + 1) % 2 == 1 else 0
            for i in range(len(C) - 1)
        ]

        # Step 3: evaluate Omega and sigma' at Xi_inv for each error, apply correction
        for actual_pos, chien_i in err_pairs:
            Xi_inv = GF16._exp[chien_i]          # X_j^{-1} = alpha^{chien_i}
            Xi     = GF16.inv(Xi_inv)             # X_j = alpha^{n-1-p_j}

            # Horner evaluation of Omega(Xi_inv)
            omega_val = 0
            for coeff in reversed(Omega):
                omega_val = GF16.add(GF16.mul(omega_val, Xi_inv), coeff)

            # Horner evaluation of sigma'(Xi_inv)
            sigma_p_val = 0
            for coeff in reversed(sigma_prime):
                sigma_p_val = GF16.add(GF16.mul(sigma_p_val, Xi_inv), coeff)

            if sigma_p_val == 0:
                return r[:self.k]

            e = GF16.mul(Xi, GF16.mul(omega_val, GF16.inv(sigma_p_val)))
            r[actual_pos] ^= e

        return r[:self.k]


# ─────────────────────────────────────────────────────────────
# AcDense — High-density cipher constants
#
# AcDense uses 9 simultaneous tones (vs 6 for WaveSig) with
# RS(9,7) per frame and zlib compression for Unicode/emoji support.
#
# Raw throughput:  7 nibbles / 192ms = 18.2 bytes/sec (compressed)
# vs WaveSig:      4 bytes   / 192ms = 10.4 bytes/sec (uncompressed)
# Effective gain:  ~6× for typical English text with zlib
#
# Frequency layout (9 tones × 16 positions × 100 Hz = 14.4 kHz range):
#   Tone 0:  1000–2500 Hz
#   Tone 1:  2600–4100 Hz
#   ...
#   Tone 8: 13800–15300 Hz
#   Marker:  15600 Hz  (200 Hz clear gap above all data tones)
# ─────────────────────────────────────────────────────────────

ACDENSE_F0      = 1000    # Hz — same base as WaveSig
ACDENSE_DF      = 100     # Hz — same bin spacing as WaveSig
ACDENSE_N_TONES = 9       # simultaneous tones (vs 6 for WaveSig)
ACDENSE_N_POS   = 16      # positions per tone (GF(16))
ACDENSE_MARKER  = 15600   # Hz — preamble/postamble marker (unique to AcDense)
# ACDENSE_MARKER = F0 + N_TONES*N_POS*DF + 200 = 1000 + 14400 + 200 = 15600 Hz

ACDENSE_RS = RS_GF16(9, 7)   # RS(9,7): 7 data + 2 parity nibbles per frame
#                                corrects 1 symbol (tone) error per frame


# ─────────────────────────────────────────────────────────────
# ACHD (AudioCipher HyperDense) constants
# ─────────────────────────────────────────────────────────────
# RS(15,11) over GF(16): 11 data + 4 parity nibbles per frame.
# n=15 is the maximum RS codeword length over GF(16) (group order = 15).
# Corrects up to 2 symbol errors per frame (vs 1 for AcDense).
#
# Frequency layout (15 tones × 16 positions × 75 Hz = 18000 Hz range):
#   Tone n, position p: freq = 900 + (n×16 + p) × 75 Hz
#   Tone 0:   900–2025 Hz
#   Tone 14: 16725–17850 Hz
#   Marker:  19200 Hz  (1350 Hz clear gap above data; < 22050 Hz Nyquist at 44.1 kHz)
#
# Throughput (typical English):
#   Raw:       28.6 bytes/sec compressed (vs AcDense 18.2 bytes/sec)
#   Effective: ~100 chars/sec after zlib (~1.57× AcDense)
# ─────────────────────────────────────────────────────────────

ACHD_F0      = 900     # Hz — base frequency
ACHD_DF      = 75      # Hz — tone spacing (75 Hz fits 15 tones within 44.1 kHz)
ACHD_N_TONES = 15      # simultaneous tones (maximum for GF(16) RS codeword)
ACHD_N_POS   = 16      # positions per tone (GF(16))
ACHD_MARKER  = 19200   # Hz — preamble/postamble marker (unique to ACHD)
# ACHD_MARKER = 900 + 15×16×75 + 1350 = 900 + 18000 + 300 ≈ 19200 Hz

ACHD_RS    = RS_GF16(15, 11)  # RS(15,11): 11 data + 4 parity; corrects 2 errors
ACHD_MAGIC = bytes([0xAC, 0x1D])  # magic header bytes (AcDense uses 0xAC 0xDE)


# ─────────────────────────────────────────────────────────────
# FFT Helpers (ported from app.html lines 3552–3617)
# ─────────────────────────────────────────────────────────────

def fft_peak(samples: np.ndarray, sr: int, fmin: float, fmax: float) -> dict:
    """Find peak frequency in [fmin, fmax] with parabolic interpolation.
    Returns {'freq': Hz, 'amp': magnitude}."""
    N = len(samples)
    windowed = samples * np.hanning(N)
    spectrum = np.fft.rfft(windowed)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)

    min_bin = max(0, int(fmin * N / sr))
    max_bin = min(len(mag) - 1, int(fmax * N / sr))
    if min_bin >= max_bin:
        return {'freq': 0.0, 'amp': 0.0}

    band = mag[min_bin:max_bin + 1]
    peak_idx = int(np.argmax(band)) + min_bin

    # Parabolic interpolation
    if 0 < peak_idx < len(mag) - 1:
        a, b, c = mag[peak_idx - 1], mag[peak_idx], mag[peak_idx + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        refined = peak_idx + delta
        freq = refined * sr / N
    else:
        freq = peak_idx * sr / N

    return {'freq': float(freq), 'amp': float(mag[peak_idx])}


def goertzel(samples: np.ndarray, sr: int, target_hz: float) -> float:
    """
    Goertzel algorithm — O(N) single-frequency energy at target_hz.

    Uses scipy.signal.lfilter (C-level, truly vectorised) when scipy is
    available.  Falls back to the numpy-array recurrence (still pure Python
    loop, but avoids Python float-object boxing) when scipy is absent.

    The Goertzel recurrence  s[n] = x[n] + coeff*s[n-1] - s[n-2]  is
    equivalent to the IIR filter:
        b = [1],  a = [1, -coeff, 1]
    so lfilter is a drop-in, correct, and ~20-50× faster on typical windows.
    """
    N = len(samples)
    if N == 0:
        return 0.0
    k = int(0.5 + N * target_hz / sr)
    omega = 2.0 * np.pi * k / N
    coeff = 2.0 * np.cos(omega)

    try:
        from scipy.signal import lfilter  # type: ignore
        b = np.array([1.0], dtype=np.float64)
        a = np.array([1.0, -coeff, 1.0], dtype=np.float64)
        y = lfilter(b, a, samples.astype(np.float64))
        s1 = float(y[-1])
        s2 = float(y[-2]) if len(y) > 1 else 0.0
    except ImportError:
        # Fallback: numpy array recurrence (avoids Python float boxing)
        s = np.zeros(N + 2, dtype=np.float64)
        x = samples.astype(np.float64)
        for i in range(N):
            s[i + 2] = x[i] + coeff * s[i + 1] - s[i]
        s1, s2 = float(s[N + 1]), float(s[N])

    return float(s1 * s1 + s2 * s2 - coeff * s1 * s2)


def rms_windows(samples: np.ndarray, sr: int, window_ms: float = 5.0) -> np.ndarray:
    """Compute RMS in non-overlapping windows of window_ms ms."""
    hop = max(1, int(sr * window_ms / 1000))
    n_windows = len(samples) // hop
    out = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        seg = samples[i * hop:(i + 1) * hop]
        out[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    return out


def on_off_events(rms: np.ndarray, hop_samples: int, sr: int,
                  threshold: float = 0.005, hysteresis: int = 3):
    """Convert RMS array to list of (type, sample_idx) on/off events with hysteresis."""
    events = []
    state = False
    count = 0
    for i, v in enumerate(rms):
        above = v > threshold
        if above != state:
            count += 1
            if count >= hysteresis:
                state = above
                events.append(('on' if state else 'off', i * hop_samples))
                count = 0
        else:
            count = 0
    return events


# ─────────────────────────────────────────────────────────────
# WAV I/O
# ─────────────────────────────────────────────────────────────

def read_wav(path: str):
    """Read WAV file. Returns (samples float32, sample_rate int)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)  # stereo → mono
    return data, sr


def _build_info_chunk(meta_json: str) -> bytes:
    """Build RIFF INFO chunk with ICMT sub-chunk containing JSON metadata."""
    meta_bytes = meta_json.encode('utf-8')
    if len(meta_bytes) % 2 != 0:
        meta_bytes += b'\x00'
    icmt_size = len(meta_bytes)
    # ICMT chunk: id(4) + size(4) + data
    icmt = b'ICMT' + struct.pack('<I', icmt_size) + meta_bytes
    # LIST/INFO wraps ICMT
    list_data = b'INFO' + icmt
    list_chunk = b'LIST' + struct.pack('<I', len(list_data)) + list_data
    return list_chunk


def write_wav(path: str, samples: np.ndarray, sr: int = 44100,
              mode: str = None, params: dict = None):
    """Write WAV file with optional AudioCipher metadata embedded in INFO chunk."""
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    pcm_bytes = pcm.tobytes()

    meta_chunk = b''
    if mode:
        meta = {
            'tool': 'AudioCipher',
            'version': 1,
            'mode': mode,
            'params': params or {}
        }
        meta_chunk = _build_info_chunk(json.dumps(meta))

    fmt_chunk = (b'fmt '
                 + struct.pack('<I', 16)    # chunk size
                 + struct.pack('<H', 1)     # PCM
                 + struct.pack('<H', 1)     # mono
                 + struct.pack('<I', sr)    # sample rate
                 + struct.pack('<I', sr * 2)  # byte rate
                 + struct.pack('<H', 2)     # block align
                 + struct.pack('<H', 16))   # bits per sample

    data_chunk = b'data' + struct.pack('<I', len(pcm_bytes)) + pcm_bytes

    # INFO chunk placed before data chunk so Audacity / SonicVisualiser find it
    riff_data = b'WAVE' + fmt_chunk + meta_chunk + data_chunk
    riff = b'RIFF' + struct.pack('<I', len(riff_data)) + riff_data

    with open(path, 'wb') as f:
        f.write(riff)


def parse_wav_metadata(path: str) -> dict | None:
    """Extract AudioCipher JSON metadata from WAV INFO/ICMT chunk."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
        idx = 12  # skip RIFF + size + WAVE
        while idx + 8 <= len(data):
            chunk_id = data[idx:idx + 4].decode('ascii', errors='replace')
            chunk_size = struct.unpack_from('<I', data, idx + 4)[0]
            if chunk_id == 'LIST' and idx + 12 <= len(data):
                list_type = data[idx + 8:idx + 12].decode('ascii', errors='replace')
                if list_type == 'INFO':
                    sub = idx + 12
                    end = idx + 8 + chunk_size
                    while sub + 8 <= end:
                        sub_id = data[sub:sub + 4].decode('ascii', errors='replace')
                        sub_size = struct.unpack_from('<I', data, sub + 4)[0]
                        if sub_id == 'ICMT':
                            raw = data[sub + 8:sub + 8 + sub_size].rstrip(b'\x00').decode('utf-8')
                            return json.loads(raw)
                        sub += 8 + sub_size + (sub_size % 2)
            idx += 8 + chunk_size + (chunk_size % 2)
    except Exception:
        pass
    return None
