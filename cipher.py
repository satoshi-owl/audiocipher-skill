"""
cipher.py — AudioCipher encode/decode for all cipher modes.

Modes:
  hzalpha  — HZ Alpha: chromatic frequency mapping (220 Hz – 8872 Hz)
  morse    — International Morse Code with ITU symbol extensions
  dtmf     — DTMF dual-tone multi-frequency (T9 mapping for letters)
  fsk      — Binary FSK: ASCII → 8-bit bitstream (F0=1000Hz, F1=1200Hz)
  ggwave   — WaveSig: multi-freq FSK with Reed-Solomon RS(12,8) ECC
  custom   — User-defined frequency map (pass freq_map=dict)

Source reference: app.html lines 2393–2544 (generateTokens),
                  2927–3022 (renderAudioBuffer / buildInfoChunk / encodeWAV),
                  3620–4193 (all decoder functions).
"""
from __future__ import annotations

import re
import json
import numpy as np

import struct

from utils import (
    HZALPHA_MAP, HZ_SHIFT_FREQ,
    MORSE_MAP, MORSE_REVERSE,
    DTMF_KEY_MAP, DTMF_REVERSE,
    T9_MAP, T9_REVERSE,
    FSK_F0, FSK_F1, FSK_BAUD,
    GGWAVE_F0, GGWAVE_DF, GGWAVE_SAMPLES_PER_FRAME, GGWAVE_SAMPLE_RATE,
    GGWAVE_FRAMES_PER_TX, GGWAVE_TONES_PER_FRAME,
    GGWAVE_MARKER_LO, GGWAVE_MARKER_HI,
    ACDENSE_F0, ACDENSE_DF, ACDENSE_N_TONES, ACDENSE_MARKER, ACDENSE_RS,
    ACHD_F0, ACHD_DF, ACHD_N_TONES, ACHD_MARKER, ACHD_RS, ACHD_MAGIC,
    GF16,
    fft_peak, goertzel,
    read_wav, write_wav, parse_wav_metadata,
)

# DTMF row and column standard frequencies
DTMF_ROWS = [697, 770, 852, 941]
DTMF_COLS = [1209, 1336, 1477, 1633]

# ─────────────────────────────────────────────────────────────
# Default encode parameters (match app.html slider defaults)
# ─────────────────────────────────────────────────────────────
DEFAULTS: dict = {
    'volume':        0.8,
    'duration_ms':   120.0,    # tone duration
    'letter_gap_ms':  20.0,    # inter-letter silence (hzalpha / dtmf)
    'word_gap_ms':   350.0,    # inter-word silence
    'fade_ms':        10.0,    # fade-in / fade-out ramp
    'waveform':      'sine',   # sine | square | sawtooth | triangle
    'morse_freq':    700.0,    # Hz for Morse tones
    'morse_wpm':      20.0,    # words per minute
    'fsk_baud':    300.0,            # bits per second (300 = practical; FSK_BAUD=45 matches JS default)
}


# ═══════════════════════════════════════════════════════════════════════════════
# WAVEFORM SYNTHESIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _oscillator(freq: float, n_samples: int, sr: int, waveform: str) -> np.ndarray:
    """Generate one period-correct oscillator at freq Hz for n_samples samples."""
    t = np.arange(n_samples, dtype=np.float64) / sr
    phi = 2.0 * np.pi * freq * t
    if waveform == 'square':
        return np.sign(np.sin(phi)).astype(np.float32)
    elif waveform in ('sawtooth', 'triangle'):
        try:
            from scipy.signal import sawtooth as _saw
            width = 1.0 if waveform == 'sawtooth' else 0.5
            return _saw(phi, width=width).astype(np.float32)
        except ImportError:
            # Fallback: pure numpy sawtooth
            frac = (freq * t) % 1.0
            if waveform == 'sawtooth':
                return (2.0 * frac - 1.0).astype(np.float32)
            else:  # triangle
                return (2.0 * np.abs(2.0 * frac - 1.0) - 1.0).astype(np.float32)
    else:  # sine (default)
        return np.sin(phi).astype(np.float32)


def _fade_envelope(n: int, fade_n: int) -> np.ndarray:
    """Linear fade-in / fade-out envelope matching Web Audio linearRampToValueAtTime."""
    env = np.ones(n, dtype=np.float32)
    fade_n = min(fade_n, n // 2)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        env[:fade_n] = ramp
        env[n - fade_n:] = ramp[::-1]
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN GENERATION (ported from generateTokens, app.html 2393–2544)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_tokens(text: str, mode: str, p: dict,
                     freq_map: dict | None = None) -> list[dict]:
    """
    Convert text to a list of token dicts:
      {'type': 'tone',    'char': str, 'freqs': [Hz, ...], 'ms': float}
      {'type': 'gap',     'ms': float}
      {'type': 'newline', 'ms': float}

    Direct port of generateTokens from app.html lines 2393–2544.
    """
    tokens: list[dict] = []

    # ── HZ Alpha / Custom ────────────────────────────────────────────────────
    if mode in ('hzalpha', 'custom'):
        fmap = freq_map if freq_map else HZALPHA_MAP
        in_word = False
        for ch in text:
            upper_ch = ch.upper()
            if ch in ('\n', '\r'):
                tokens.append({'type': 'newline', 'ms': p['word_gap_ms'] * 3.0})
                in_word = False
                continue
            if ch == ' ':
                if in_word:
                    tokens.append({'type': 'gap', 'ms': p['word_gap_ms']})
                in_word = False
                continue
            # Try uppercase first (letters), then original (symbols / digits)
            if fmap.get(upper_ch) is not None:
                key = upper_ch
            elif fmap.get(ch) is not None:
                key = ch
            else:
                key = None
            if key is not None and fmap.get(key, 0) != 0:
                if ch != upper_ch and ch.islower():
                    # Lowercase letter: emit SHIFT marker first
                    tokens.append({
                        'type': 'tone', 'char': '\x01',
                        'freqs': [HZ_SHIFT_FREQ], 'ms': p['duration_ms'],
                    })
                tokens.append({
                    'type': 'tone', 'char': key,
                    'freqs': [fmap[key]], 'ms': p['duration_ms'],
                })
                in_word = True

    # ── Morse ─────────────────────────────────────────────────────────────────
    elif mode == 'morse':
        dot_ms     = 1200.0 / p['morse_wpm']
        dash_ms    = dot_ms * 3.0
        el_gap_ms  = dot_ms
        char_gap_ms = dot_ms * 3.0
        word_gap_ms = dot_ms * 7.0

        for line_idx, line in enumerate(text.split('\n')):
            if line_idx > 0:
                tokens.append({'type': 'newline', 'ms': word_gap_ms * 3.0})
            words = line.split(' ')
            for word_idx, word in enumerate(words):
                if word_idx > 0:
                    tokens.append({'type': 'gap', 'ms': word_gap_ms})
                for ci, ch in enumerate(word):
                    code = MORSE_MAP.get(ch.upper()) or MORSE_MAP.get(ch)
                    if not code:
                        continue
                    for ei, el in enumerate(code):
                        tokens.append({
                            'type': 'tone',
                            'char': f'{ch}:{el}',
                            'freqs': [p['morse_freq']],
                            'ms': dot_ms if el == '.' else dash_ms,
                        })
                        if ei < len(code) - 1:
                            tokens.append({'type': 'gap', 'ms': el_gap_ms})
                    if ci < len(word) - 1:
                        tokens.append({'type': 'gap', 'ms': char_gap_ms})

    # ── DTMF ─────────────────────────────────────────────────────────────────
    elif mode == 'dtmf':
        # DTMF word gap must be clearly larger than the inter-char gap (letter_gap_ms*4).
        # Use max(word_gap_ms, letter_gap_ms*8) so the decoder can reliably separate
        # character boundaries from word gaps at any letterGap/wordGap setting.
        dtmf_word_gap_ms = max(p['word_gap_ms'], p['letter_gap_ms'] * 8)
        in_word = False
        for ch in text:
            upper_ch = ch.upper()
            if ch in ('\n', '\r'):
                tokens.append({'type': 'newline', 'ms': dtmf_word_gap_ms * 3.0})
                in_word = False
                continue
            if ch == ' ':
                if in_word:
                    tokens.append({'type': 'gap', 'ms': dtmf_word_gap_ms})
                in_word = False
                continue
            # Digits → direct; letters → T9; DTMF symbols (* # A-D) → direct
            if upper_ch in T9_MAP:
                t9_str = T9_MAP[upper_ch]
            elif upper_ch in DTMF_KEY_MAP:
                t9_str = upper_ch
            elif ch in DTMF_KEY_MAP:
                t9_str = ch
            else:
                t9_str = None
            if not t9_str:
                continue
            # char_gap_ms separates *different* characters; letter_gap_ms
            # separates repeated keypresses of the *same* T9 key (e.g. 'B'→'22').
            # The decoder needs char_gap_ms > letter_gap_ms to tell them apart.
            char_gap_ms = p.get('char_gap_ms', p['letter_gap_ms'] * 4)
            for di, digit in enumerate(t9_str):
                pair = DTMF_KEY_MAP.get(digit)
                if pair:
                    tokens.append({
                        'type': 'tone',
                        'char': f'{ch}({digit})',
                        'freqs': list(pair),
                        'ms': p['duration_ms'],
                    })
                    if di < len(t9_str) - 1:
                        tokens.append({'type': 'gap', 'ms': p['letter_gap_ms']})
            in_word = True
            tokens.append({'type': 'gap', 'ms': char_gap_ms})

    # ── FSK Binary ────────────────────────────────────────────────────────────
    elif mode == 'fsk':
        baud = p.get('fsk_baud', float(FSK_BAUD))
        bit_ms = 1000.0 / baud
        for ch in text:
            if ch in ('\n', '\r'):
                continue
            code = ord(ch)
            for b in range(7, -1, -1):
                bit = (code >> b) & 1
                tokens.append({
                    'type': 'tone',
                    'char': str(bit),
                    'freqs': [FSK_F1 if bit else FSK_F0],
                    'ms': bit_ms,
                })
            tokens.append({'type': 'gap', 'ms': p['letter_gap_ms']})

    # ── WaveSig / GGWave ──────────────────────────────────────────────────────
    elif mode == 'ggwave':
        frame_dur_ms = (
            GGWAVE_SAMPLES_PER_FRAME / GGWAVE_SAMPLE_RATE
        ) * 1000.0 * GGWAVE_FRAMES_PER_TX

        # Normalise common Unicode → ASCII before encoding.
        # WaveSig is byte-based (each char → two 4-bit nibbles); the decoder
        # only passes through ASCII 32-127 + whitespace, so anything outside
        # that range encodes to garbage and is silently dropped on decode.
        _UNICODE_NORM = {
            '\u2014': '--',   # em-dash        —  → --
            '\u2013': '-',    # en-dash        –  → -
            '\u201c': '"',    # left  "        "  → "
            '\u201d': '"',    # right "        "  → "
            '\u2018': "'",    # left  '        '  → '
            '\u2019': "'",    # right '        '  → '
            '\u2026': '...',  # ellipsis       …  → ...
            '\u2022': '*',    # bullet         •  → *
            '\u00b7': '.',    # middle dot     ·  → .
            '\u00a9': '(c)',  # copyright      ©  → (c)
            '\u00ae': '(r)',  # registered     ®  → (r)
            '\u2122': '(tm)',# trademark      ™  → (tm)
            '\u00b0': 'deg', # degree         °  → deg
        }
        norm_text = []
        for ch in text:
            if ch in _UNICODE_NORM:
                norm_text.append(_UNICODE_NORM[ch])
            elif ord(ch) < 128 or ch in ('\t', '\n', '\r'):
                norm_text.append(ch)
            # else: silently drop truly un-mappable non-ASCII
        text = ''.join(norm_text)

        # Text → bytes → nibbles
        raw_nibbles: list[int] = []
        for ch in text:
            code = ord(ch)
            raw_nibbles.append((code >> 4) & 0xF)
            raw_nibbles.append(code & 0xF)

        # RS(12,8): process in blocks of 8 data nibbles → 12 coded nibbles
        coded: list[int] = []
        for i in range(0, len(raw_nibbles), 8):
            block = raw_nibbles[i:i + 8]
            while len(block) < 8:
                block.append(0)
            coded.extend(GF16.encode(block))

        # Start marker (2 simultaneous tones, 2× frame duration)
        tokens.append({
            'type': 'tone', 'char': '▶',
            'freqs': [GGWAVE_MARKER_LO, GGWAVE_MARKER_HI],
            'ms': frame_dur_ms * 2.0,
        })
        tokens.append({'type': 'gap', 'ms': frame_dur_ms})

        # Data frames: 6 nibbles at a time → 6 simultaneous tones
        for i in range(0, len(coded), GGWAVE_TONES_PER_FRAME):
            group = coded[i:i + GGWAVE_TONES_PER_FRAME]
            while len(group) < GGWAVE_TONES_PER_FRAME:
                group.append(0)
            freqs = [
                GGWAVE_F0 + (nib + idx * 16) * GGWAVE_DF
                for idx, nib in enumerate(group)
            ]
            tokens.append({
                'type': 'tone', 'char': '·',
                'freqs': freqs, 'ms': frame_dur_ms,
            })

        # End marker
        tokens.append({'type': 'gap', 'ms': frame_dur_ms})
        tokens.append({
            'type': 'tone', 'char': '■',
            'freqs': [GGWAVE_MARKER_LO, GGWAVE_MARKER_HI],
            'ms': frame_dur_ms * 2.0,
        })
        # Post-roll: 400 ms silence after the end marker.
        # Mirrors the pre-roll in video.py.  AAC trims a few hundred samples
        # from the stream end on every encode hop; this ensures the end marker
        # always has a full frame group of breathing room regardless of message
        # length, so the 2nd footer tone is never clipped by codec alignment.
        tokens.append({'type': 'gap', 'ms': 400.0})

    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN RENDERING → PCM  (ported from renderAudioBuffer, app.html 2927–2964)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_tokens(tokens: list[dict], sr: int, p: dict, mode: str) -> np.ndarray:
    """
    Synthesise token list to float32 PCM.
    Matches the OfflineAudioContext rendering in app.html:
      - gain.gain.linearRampToValueAtTime(volume / num_freqs, t + fade)
      - osc.type = waveform
    """
    segments: list[np.ndarray] = []

    for i, tok in enumerate(tokens):
        if tok['type'] == 'tone':
            ms   = float(tok['ms'])
            n    = max(1, int(round(sr * ms / 1000.0)))
            fade_n = max(0, int(round(sr * p['fade_ms'] / 1000.0)))
            fade_n = min(fade_n, n // 2)
            amp  = p['volume'] / len(tok['freqs'])

            wave = np.zeros(n, dtype=np.float64)
            for freq in tok['freqs']:
                wave += amp * _oscillator(float(freq), n, sr, p['waveform'])
            wave = (wave * _fade_envelope(n, fade_n)).astype(np.float32)
            segments.append(wave)

            # Inter-letter gap after consecutive tones (hzalpha / custom only)
            if mode in ('hzalpha', 'custom'):
                if i < len(tokens) - 1 and tokens[i + 1]['type'] == 'tone':
                    gap_n = max(0, int(round(sr * p['letter_gap_ms'] / 1000.0)))
                    if gap_n:
                        segments.append(np.zeros(gap_n, dtype=np.float32))

        elif tok['type'] in ('gap', 'newline'):
            gap_n = max(0, int(round(sr * tok['ms'] / 1000.0)))
            if gap_n:
                segments.append(np.zeros(gap_n, dtype=np.float32))

    if not segments:
        return np.zeros(sr, dtype=np.float32)

    audio = np.concatenate(segments)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio * (0.9 / peak)
    return audio.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENCODE API
# ═══════════════════════════════════════════════════════════════════════════════

def encode(text: str, mode: str = 'hzalpha', sr: int = 44100,
           freq_map: dict | None = None, **kwargs) -> np.ndarray:
    """
    Encode text to float32 PCM audio.

    Args:
        text:     Input text to encode
        mode:     'hzalpha' | 'morse' | 'dtmf' | 'fsk' | 'ggwave' | 'acdense' | 'achd' | 'custom'
        sr:       Output sample rate (default 44100)
        freq_map: Frequency map for mode='custom' (defaults to HZALPHA_MAP)
        **kwargs: Override any DEFAULTS key, e.g. duration_ms=80, waveform='square'

    Returns:
        np.ndarray, float32, mono, normalised to ±0.9

    Notes:
        mode='acdense': full Unicode + emoji support via zlib compression.
          ~6× higher effective throughput than 'ggwave' for typical text.
          Marker at 15600 Hz enables lossless auto-detection from WAV/MP4.
    """
    p = {**DEFAULTS, **kwargs}
    if mode == 'acdense':
        return _encode_acdense(text, sr, volume=p.get('volume', 0.8))
    if mode == 'achd':
        return _encode_achd(text, sr, volume=p.get('volume', 0.8))
    tokens = _generate_tokens(text, mode, p, freq_map=freq_map)
    return _render_tokens(tokens, sr, p, mode)


def write_cipher_wav(path: str, text: str, mode: str = 'hzalpha', sr: int = 44100,
                     freq_map: dict | None = None, **kwargs):
    """
    Encode text and write a WAV file with embedded AudioCipher metadata.
    The metadata allows `decode(..., mode='auto')` to auto-restore mode+params.
    """
    p = {**DEFAULTS, **kwargs}
    samples = encode(text, mode=mode, sr=sr, freq_map=freq_map, **kwargs)
    meta_params = {
        k: p[k] for k in (
            'volume', 'duration_ms', 'letter_gap_ms', 'word_gap_ms',
            'fade_ms', 'waveform', 'morse_freq', 'morse_wpm', 'fsk_baud',
        ) if k in p
    }
    write_wav(path, samples, sr=sr, mode=mode, params=meta_params)


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — Shared on/off event builder
# (Phase-1 logic from decodeHZAlpha, app.html 3661–3706)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_run_events(samples: np.ndarray, sr: int,
                      threshold: float = 0.005,
                      rms_hop_ms: float = 5.0,
                      hysteresis: int = 3) -> list[dict]:
    """
    Slide an RMS window over `samples` and return run-length events:
      [{'type': 'on'|'off', 'start_sample': int, 'end_sample': int, 'frames': int}, ...]

    Ported verbatim from Phase-1 of decodeHZAlpha (app.html 3661–3706), used by
    all three tone-based decoders (hzalpha, dtmf, and as a helper for ggwave).
    """
    hop = max(1, int(round(sr * rms_hop_ms / 1000.0)))
    total_hops = (len(samples) - hop) // hop

    events: list[dict] = []
    cur_state  = False
    run_start  = 0
    pend_state = False
    pend_count = 0

    for h in range(total_hops + 1):
        is_last = (h == total_hops)
        si = h * hop
        seg = samples[si: min(si + hop, len(samples))]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) > 0 else 0.0
        is_on = cur_state if is_last else (rms >= threshold)

        if is_on == cur_state:
            pend_count = 0
        else:
            if is_on == pend_state:
                pend_count += 1
            else:
                pend_state = is_on
                pend_count = 1
            if pend_count >= hysteresis or is_last:
                trans = h - pend_count + 1
                events.append({
                    'type':         'on' if cur_state else 'off',
                    'start_sample': run_start * hop,
                    'end_sample':   trans * hop,
                    'frames':       trans - run_start,
                })
                cur_state  = pend_state
                run_start  = trans
                pend_count = 0

    if run_start < total_hops:
        events.append({
            'type':         'on' if cur_state else 'off',
            'start_sample': run_start * hop,
            'end_sample':   total_hops * hop,
            'frames':       total_hops - run_start,
        })
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — HZ Alpha  (ported from decodeHZAlpha, app.html 3620–3779)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_hzalpha(samples: np.ndarray, sr: int, p: dict,
                    freq_map: dict | None = None) -> str:
    fmap    = freq_map if freq_map else HZALPHA_MAP
    letters = [k for k, v in fmap.items() if v > 0]
    freqs   = [fmap[l] for l in letters]

    all_freqs = freqs + [HZ_SHIFT_FREQ]
    min_f = min(all_freqs) * 0.92
    max_f = min(max(all_freqs) * 1.08, sr / 2.0 - 100.0)

    sorted_all = sorted(all_freqs)
    min_gap   = min(sorted_all[i+1] - sorted_all[i] for i in range(len(sorted_all) - 1))
    tolerance = min_gap * 0.45

    rms_hop_ms = 5.0
    FFT_N      = 8192
    half_fft   = FFT_N // 2

    silence_for_letter = max(int(p['letter_gap_ms'] / rms_hop_ms * 0.5), 2)
    silence_for_word   = max(int(p['word_gap_ms']   / rms_hop_ms * 0.55), 5)
    silence_for_nl     = silence_for_word * 3

    events = _build_run_events(samples, sr, threshold=0.005,
                               rms_hop_ms=rms_hop_ms, hysteresis=3)

    detections:    list[str] = []
    last_letter:   str | None = None
    last_ws:       str | None = None   # 'space' | 'newline' | None
    consec_sil     = 0
    next_is_lower  = False

    for ev in events:
        if ev['type'] == 'on':
            consec_sil = 0
            last_ws    = None

            # Centre an FFT_N-sample window on the midpoint of this tone burst
            mid       = (ev['start_sample'] + ev['end_sample']) // 2
            fft_start = max(0, mid - half_fft)
            fft_end   = min(len(samples), fft_start + FFT_N)
            window    = samples[fft_start:fft_end]
            if len(window) < FFT_N:
                window = np.concatenate(
                    [window, np.zeros(FFT_N - len(window), dtype=np.float32)]
                )

            peak = fft_peak(window, sr, min_f, max_f)
            f    = peak['freq']

            # SHIFT marker?
            if abs(f - HZ_SHIFT_FREQ) < tolerance:
                next_is_lower = True
                last_letter   = '\x01'
                continue

            # Nearest letter
            best, best_diff = None, float('inf')
            for li, letter in enumerate(letters):
                d = abs(freqs[li] - f)
                if d < best_diff:
                    best_diff = d
                    best      = letter

            if best is not None and best_diff < tolerance and best != last_letter:
                out = best.lower() if (next_is_lower and best.isalpha()) else best
                detections.append(out)
                last_letter   = best
                next_is_lower = False

        else:  # off run
            consec_sil += ev['frames']

            if consec_sil >= silence_for_nl and last_ws != 'newline':
                # Upgrade space → newline
                if last_ws == 'space':
                    for d in range(len(detections) - 1, -1, -1):
                        if detections[d] == ' ':
                            detections.pop(d)
                            break
                detections.append('\n')
                last_ws     = 'newline'
                last_letter = None

            elif consec_sil >= silence_for_word and last_ws is None and last_letter is not None:
                detections.append(' ')
                last_ws     = 'space'
                last_letter = None

            elif consec_sil >= silence_for_letter:
                last_letter = None  # allow same letter to re-fire

    result = ''.join(detections)
    result = re.sub(r' {2,}', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — Morse  (ported from decodeMorse, app.html 3782–3922)
# ═══════════════════════════════════════════════════════════════════════════════

def _goertzel_amp(samples: np.ndarray, sr: int, target_hz: float) -> float:
    """Normalised Goertzel amplitude matching the JS formula: sqrt(power) / N."""
    N = len(samples)
    if N == 0:
        return 0.0
    k      = int(0.5 + N * target_hz / sr)
    omega  = 2.0 * np.pi * k / N
    coeff  = 2.0 * np.cos(omega)
    s0 = s1 = s2 = 0.0
    for s in samples:
        s0 = float(s) + coeff * s1 - s2
        s2 = s1
        s1 = s0
    power = max(0.0, s1 * s1 + s2 * s2 - s1 * s2 * coeff)
    return float(np.sqrt(power)) / N


def _decode_morse(samples: np.ndarray, sr: int, p: dict) -> str:
    dot_ms          = 1200.0 / p['morse_wpm']
    hop_ms          = 5.0
    hop_samples     = max(1, int(round(sr * hop_ms / 1000.0)))
    window_samples  = max(int(round(sr * dot_ms / 1000.0 * 0.4)), 64)
    silence_thresh  = 0.008
    AUTO_FFT_N      = 2048

    # ── Auto-detect dominant Morse tone (200–2000 Hz) ────────────────────────
    auto_skip = int(len(samples) * 0.05)
    auto_end  = int(len(samples) * 0.95)
    auto_hop  = max(1, (auto_end - auto_skip) // 100)
    freq_accum = np.zeros(AUTO_FFT_N // 2, dtype=np.float64)
    auto_frames = 0

    for si in range(auto_skip, auto_end - AUTO_FFT_N, auto_hop):
        seg = samples[si:si + window_samples]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if rms < silence_thresh:
            continue
        frame = samples[si:si + AUTO_FFT_N]
        if len(frame) < AUTO_FFT_N:
            continue
        spec = np.abs(np.fft.rfft(frame.astype(np.float64)))
        freq_accum += spec[1:AUTO_FFT_N // 2 + 1] ** 2
        auto_frames += 1

    detected_morse_freq = p['morse_freq']  # fallback
    if auto_frames >= 3:
        bin_hz  = sr / AUTO_FFT_N
        min_bin = max(1, int(np.ceil(200.0 / bin_hz)))
        max_bin = min(int(np.floor(2000.0 / bin_hz)), len(freq_accum) - 1)
        best_offset = int(np.argmax(freq_accum[min_bin:max_bin + 1]))
        detected_morse_freq = (min_bin + best_offset) * bin_hz

    # ── Goertzel baseline from first 5 % of audio (silent frames only) ──────
    # Only include frames whose RMS is below silence_thresh — this gives a true
    # noise-floor estimate.  If the Morse signal starts immediately at sample 0
    # (no pre-roll) the first 5 % will be all-tone; in that case baseline_count
    # stays 0, baseline_amp = 0.0, and goertzel_thresh falls back to the fixed
    # minimum (silence_thresh * 0.5 = 0.004).
    baseline_sum, baseline_count = 0.0, 0
    for i in range(0, int(len(samples) * 0.05), hop_samples):
        seg = samples[i:i + window_samples]
        if len(seg) < 16:
            continue
        if float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) >= silence_thresh:
            continue                            # skip active-tone frames
        baseline_sum   += _goertzel_amp(seg, sr, detected_morse_freq)
        baseline_count += 1
    baseline_amp    = baseline_sum / baseline_count if baseline_count > 0 else 0.0
    goertzel_thresh = max(silence_thresh * 0.5, baseline_amp * 3.0)

    # ── Build on/off stream via Goertzel ─────────────────────────────────────
    stream: list[dict] = []
    state, count = False, 0
    for i in range(0, len(samples) - window_samples, hop_samples):
        seg = samples[i:i + window_samples]
        amp = _goertzel_amp(seg, sr, detected_morse_freq)
        is_on = (amp >= goertzel_thresh)
        if is_on == state:
            count += 1
        else:
            if count > 0:
                stream.append({'type': 'on' if state else 'off', 'ms': count * hop_ms})
            state = is_on
            count = 1
    if count > 0:
        stream.append({'type': 'on' if state else 'off', 'ms': count * hop_ms})

    # ── Classify elements ─────────────────────────────────────────────────────
    dot_thresh  = dot_ms * 1.5
    char_gap    = dot_ms * 2.5
    word_gap    = dot_ms * 5.5
    newline_gap = dot_ms * 21.0

    code   = ''
    result: list[str] = []

    for el in stream:
        if el['type'] == 'on':
            code += '.' if el['ms'] < dot_thresh else '-'
        else:
            if el['ms'] >= newline_gap:
                letter = MORSE_REVERSE.get(code)
                if letter:
                    result.append(letter)
                code = ''
                if result and result[-1] == ' ':
                    result.pop()
                if not result or result[-1] != '\n':
                    result.append('\n')
            elif el['ms'] >= word_gap:
                letter = MORSE_REVERSE.get(code)
                if letter:
                    result.append(letter)
                code = ''
                if not result or result[-1] not in (' ', '\n'):
                    result.append(' ')
            elif el['ms'] >= char_gap:
                letter = MORSE_REVERSE.get(code)
                if letter:
                    result.append(letter)
                code = ''

    if code:
        letter = MORSE_REVERSE.get(code)
        if letter:
            result.append(letter)

    text = ''.join(result)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — DTMF  (ported from decodeDTMF, app.html 3925–4057)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_dtmf(samples: np.ndarray, sr: int, p: dict) -> str:
    DTMF_FFT_N = 1024
    half_dtmf  = DTMF_FFT_N // 2
    rms_hop_ms = 5.0

    # char_gap_ms is 4× letter_gap_ms by convention (set in the encoder).
    # silence thresholds (in RMS frames):
    #   silence_for_letter : resets last_digit so the same key can fire again
    #                        (intra-character multi-press gap, e.g. 'B' → '2','2')
    #   silence_for_char   : marks a T9 character boundary with '|' in detected
    #                        (fires at half of char_gap_ms, which is > intra gap)
    #   silence_for_word   : inserts a word space
    #   silence_for_nl     : inserts a newline
    char_gap_ms = p.get('char_gap_ms', p['letter_gap_ms'] * 4)
    # DTMF word gap must be clearly larger than the inter-character gap (letter_gap_ms*4).
    # The JS encoder uses max(wordGap, letterGap*8) so this decoder mirrors that rule,
    # ensuring cross-compatibility regardless of which side encoded the audio.
    dtmf_word_gap_ms = max(p['word_gap_ms'], p['letter_gap_ms'] * 8)

    silence_for_letter = max(int(p['letter_gap_ms']  / rms_hop_ms * 0.5),  2)
    silence_for_char   = max(int(char_gap_ms         / rms_hop_ms * 0.5),  4)
    silence_for_word   = max(int(dtmf_word_gap_ms    / rms_hop_ms * 0.55), silence_for_char + 2)
    silence_for_nl     = silence_for_word * 3

    events = _build_run_events(samples, sr, threshold=0.005,
                               rms_hop_ms=rms_hop_ms, hysteresis=3)

    detected: list[str] = []
    last_digit: str | None = None
    last_ws:    str | None = None
    consec_sil  = 0

    for ev in events:
        if ev['type'] == 'on':
            consec_sil = 0
            last_ws    = None

            mid       = (ev['start_sample'] + ev['end_sample']) // 2
            fft_start = max(0, mid - half_dtmf)
            window    = samples[fft_start: min(len(samples), fft_start + DTMF_FFT_N)]
            if len(window) < DTMF_FFT_N:
                window = np.concatenate(
                    [window, np.zeros(DTMF_FFT_N - len(window), dtype=np.float32)]
                )

            # Best row freq — select by highest amplitude.
            # Selecting by smallest frequency-diff can be fooled by parabolic-
            # interpolation artifacts that produce phantom peaks at near-zero
            # amplitude but suspiciously accurate frequency.
            best_row, best_row_diff, best_row_amp = None, float('inf'), 0.0
            for f in DTMF_ROWS:
                peak = fft_peak(window, sr, f - 50, f + 50)
                if peak['amp'] > best_row_amp:
                    best_row_amp  = peak['amp']
                    best_row_diff = abs(peak['freq'] - f)
                    best_row      = f

            # Best col freq — same amplitude-first strategy
            best_col, best_col_diff, best_col_amp = None, float('inf'), 0.0
            for f in DTMF_COLS:
                peak = fft_peak(window, sr, f - 70, f + 70)
                if peak['amp'] > best_col_amp:
                    best_col_amp  = peak['amp']
                    best_col_diff = abs(peak['freq'] - f)
                    best_col      = f

            if best_row and best_col and best_row_diff < 60 and best_col_diff < 100:
                key = DTMF_REVERSE.get((best_row, best_col))
                if key and key != last_digit:
                    detected.append(key)
                    last_digit = key

        else:
            consec_sil += ev['frames']

            if consec_sil >= silence_for_nl and last_ws != 'newline':
                if last_ws == 'space':
                    for d in range(len(detected) - 1, -1, -1):
                        if detected[d] == ' ':
                            detected.pop(d)
                            break
                detected.append('\n')
                last_ws    = 'newline'
                last_digit = None
            elif consec_sil >= silence_for_word and last_ws is None and last_digit is not None:
                detected.append(' ')
                last_ws    = 'space'
                last_digit = None
            elif consec_sil >= silence_for_char and last_ws is None:
                # T9 character boundary: separate the key-sequence for each
                # character with '|' so the T9 decoder can split correctly.
                # Example: 'LL' encodes as '555|555', not ambiguous '555555'.
                if detected and detected[-1] not in ('|', ' ', '\n'):
                    detected.append('|')
                last_digit = None
            elif consec_sil >= silence_for_letter:
                last_digit = None

    # T9 reverse decode: digit runs → letters
    # '|' separates different T9 characters; each part is decoded independently.
    # re.finditer(...).group(0) captures the FULL repeated-char match ('22','555')
    # rather than just the capture group ('2','5') that re.findall() returns.
    full_str = ''.join(detected)
    decoded_lines = []
    for line in full_str.split('\n'):
        decoded_words = []
        for word in line.split(' '):
            parts = word.split('|')
            decoded_chars: list[str] = []
            for part in parts:
                if not part:
                    continue
                groups = [m.group(0) for m in re.finditer(r'(.)\1*', part)]
                decoded_chars.extend(T9_REVERSE.get(g, g) for g in groups)
            decoded_words.append(''.join(decoded_chars))
        decoded_lines.append(re.sub(r' {2,}', ' ', ' '.join(decoded_words)).strip())

    result = '\n'.join(decoded_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — FSK Binary  (ported from decodeFSK, app.html 4060–4108)
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_fsk(samples: np.ndarray, sr: int, p: dict) -> str:
    baud         = float(p.get('fsk_baud', FSK_BAUD))
    bit_samples  = max(1, int(round(sr / baud)))   # exact samples per bit
    silence_thresh = 0.003

    # ── Frequency resolution fix ────────────────────────────────────────────
    # fft_peak with a ±80 Hz search band breaks when the FFT bin width
    # exceeds the search range (e.g. 300 baud → 118 samples → 374 Hz/bin,
    # so both 920–1080 Hz and 1120–1280 Hz collapse to a single bin and
    # fft_peak returns 0 for F0 every time).
    #
    # Fix: use goertzel() which evaluates energy at the *exact* target
    # frequency (k = round(N * f / sr)), then zero-pad the bit window so
    # that F0 and F1 map to *different* k values with clean separation.
    #
    # pad_len = 2 * ceil(sr / |F1-F0|) guarantees bin separation ≥ 1:
    #   At 44.1 kHz, F0=1000, F1=1200: pad_len = 2*221 = 442
    #   → k_F0 = round(442*1000/44100) = 10 → 1000 Hz exactly
    #   → k_F1 = round(442*1200/44100) = 12 → 1200 Hz exactly
    #
    # The bit window (1/baud seconds) is kept intact for time alignment;
    # trailing zeros are appended only for the frequency measurement.
    pad_len = max(bit_samples, int(np.ceil(2.0 * sr / abs(FSK_F1 - FSK_F0))))

    bits: list[int | None] = []
    for i in range(0, len(samples) - bit_samples + 1, bit_samples):
        frame = samples[i:i + bit_samples].astype(np.float64)
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < silence_thresh:
            bits.append(None)
            continue
        # Zero-pad to pad_len for adequate frequency resolution
        padded = np.zeros(pad_len, dtype=np.float64)
        padded[:len(frame)] = frame
        e0 = goertzel(padded, sr, FSK_F0)
        e1 = goertzel(padded, sr, FSK_F1)
        bits.append(1 if e1 > e0 else 0)

    result: list[str] = []
    i = 0
    while i < len(bits):
        if bits[i] is None:
            i += 1
            continue
        byte_bits: list[int] = []
        j = i
        while len(byte_bits) < 8 and j < len(bits):
            if bits[j] is not None:
                byte_bits.append(bits[j])  # type: ignore[arg-type]
            j += 1
        if len(byte_bits) == 8:
            code = sum(b << (7 - idx) for idx, b in enumerate(byte_bits))
            if (32 <= code < 128) or code in (9, 10, 13):
                result.append(chr(code))
        i = j

    text = ''.join(result)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — WaveSig / GGWave  (ported from decodeGGWave, app.html 4110–4193)
# ═══════════════════════════════════════════════════════════════════════════════

def _probe_ggwave(samples: np.ndarray, sr: int) -> bool:
    """
    Spectral fingerprint: detect WaveSig from the signal itself, no metadata.

    WaveSig's HI marker tone (GGWAVE_MARKER_HI = 10800 Hz) is played during
    the 2-group preamble and falls well above all other AudioCipher modes:
        HZAlpha max  = 8869 Hz
        DTMF max     = 1633 Hz
        FSK max      = 1200 Hz

    We skip over any leading silence (e.g. the 400 ms pre-roll added by
    `audiocipher video`) to locate the true preamble, then probe ONLY those
    two groups where the HI marker plays exclusively — before any data tones
    appear.  In that window channel 5 data tones (9000–10500 Hz) are absent,
    giving a >100,000× signal-to-noise ratio vs the control band.

    Works from any source including Twitter-re-encoded video because 10800 Hz
    sits comfortably within AAC's passband and the 100 Hz bin spacing gives
    a generous ±50 Hz error margin.
    """
    pg = round(sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE)
    total_groups = len(samples) // pg
    if total_groups < 5:
        return False

    # Find the first active frame group (skip pre-roll silence or quiet padding).
    look_ahead = min(total_groups, 8)  # preamble can't be more than 8 groups in
    energies   = [
        float(np.sqrt(np.mean(samples[g * pg:(g + 1) * pg].astype(np.float64) ** 2)))
        for g in range(look_ahead)
    ]
    mean_e    = sum(energies) / len(energies) or 1.0
    threshold = mean_e * 0.2
    first_active = next((g for g, e in enumerate(energies) if e > threshold), -1)
    if first_active < 0:
        return False

    # Probe exactly those two preamble groups — no data tones yet.
    start = first_active * pg
    end   = start + 2 * pg
    if end > len(samples):
        return False
    probe = samples[start:end]

    # Energy at WaveSig HI marker (±1 bin around 10800 Hz).
    hi   = fft_peak(probe, sr, GGWAVE_MARKER_HI - GGWAVE_DF, GGWAVE_MARKER_HI + GGWAVE_DF)
    # Control: band above the marker — silent in all modes.
    ctrl = fft_peak(probe, sr, GGWAVE_MARKER_HI + GGWAVE_DF, GGWAVE_MARKER_HI + 3 * GGWAVE_DF)
    # HI marker must clearly outpower the noise floor (>10× is conservative).
    return hi['amp'] > 0.0 and hi['amp'] > ctrl['amp'] * 10.0


def _decode_ggwave(samples: np.ndarray, sr: int) -> str:
    # Single round() call to match the encoder's int(round(sr * frame_dur_ms/1000))
    # where frame_dur_ms = (GGWAVE_SAMPLES_PER_FRAME/GGWAVE_SAMPLE_RATE)*1000*GGWAVE_FRAMES_PER_TX.
    # Using round(sub_frame) * GGWAVE_FRAMES_PER_TX gives a 2-sample drift at 44100 Hz
    # (941*9=8469 vs the correct 8467) that causes total_groups to be one short,
    # cutting off the last data frame of longer messages.
    frame_dur_samples = round(
        sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE
    )

    # AAC/Opus re-encoding can truncate the last frame group by a few hundred
    # samples (codec frame-alignment at stream end).  If the tail is ≥50% of a
    # full group it almost certainly contains real signal, so pad with zeros to
    # include it — this recovers the last data/footer group that would otherwise
    # be lost, restoring the final decoded character.
    tail = len(samples) % frame_dur_samples
    if tail >= frame_dur_samples // 2:
        samples = np.concatenate([samples, np.zeros(frame_dur_samples - tail, dtype=samples.dtype)])

    total_groups = len(samples) // frame_dur_samples

    if total_groups < 5:
        return '(audio too short for WaveSig)'

    # RMS energy per frame group
    energies = []
    for g in range(total_groups):
        offset = g * frame_dur_samples
        seg    = samples[offset:offset + frame_dur_samples]
        energies.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))

    mean_e    = sum(energies) / len(energies)
    threshold = mean_e * 0.3

    first_active = next((g for g, e in enumerate(energies) if e > threshold), -1)
    last_active  = (
        len(energies) - 1
        - next((g for g, e in enumerate(reversed(energies)) if e > threshold), -1)
    )

    if first_active < 0 or last_active <= first_active + 4:
        return '(no WaveSig signal detected)'

    # ggwave structure: [2 preamble groups][1 silent gap][N data groups][1 silent gap][2 footer groups]
    # Skip preamble (2) + silent gap (1) at start; skip silent gap (1) + footer (2) at end.
    data_start = first_active + 3
    data_end   = last_active - 3  # inclusive

    if data_end < data_start:
        return '(no WaveSig signal detected)'

    # Collect raw nibbles from each data frame group
    raw_nibbles: list[int] = []
    for g in range(data_start, data_end + 1):
        offset = g * frame_dur_samples
        frame  = samples[offset:offset + frame_dur_samples]
        for ch in range(GGWAVE_TONES_PER_FRAME):
            f_lo = GGWAVE_F0 + ch * 16 * GGWAVE_DF
            f_hi = f_lo + 15 * GGWAVE_DF
            peak = fft_peak(frame, sr, f_lo, f_hi)
            nib  = max(0, min(15, round((peak['freq'] - f_lo) / GGWAVE_DF)))
            raw_nibbles.append(nib)

    # RS(12,8) ECC decoding — fall back to raw data nibbles if RS corrupts the result.
    # The FFT frequency detection is reliable enough that raw nibbles are often cleaner.
    decoded_nibbles: list[int] = []
    for i in range(0, len(raw_nibbles) - 11, 12):
        block    = raw_nibbles[i:i + 12]
        raw_data = block[:8]
        try:
            rs_data = GF16.decode(block)[:8]
            # Prefer RS only when it agrees with raw or raw has non-printable pairs
            raw_bytes = [(raw_data[j] << 4) | raw_data[j + 1] for j in range(0, 7, 2)]
            rs_bytes  = [(rs_data[j]  << 4) | rs_data[j + 1]  for j in range(0, 7, 2)]
            raw_printable = sum(1 for b in raw_bytes if 32 <= b < 128)
            rs_printable  = sum(1 for b in rs_bytes  if 32 <= b < 128)
            decoded_nibbles.extend(rs_data if rs_printable >= raw_printable else raw_data)
        except Exception:
            decoded_nibbles.extend(raw_data)

    # Nibble pairs → bytes → text
    bytes_out = [
        (decoded_nibbles[i] << 4) | decoded_nibbles[i + 1]
        for i in range(0, len(decoded_nibbles) - 1, 2)
    ]
    result = ''.join(
        chr(b) for b in bytes_out
        if (32 <= b < 128) or b in (9, 10, 13)
    )
    result = result.replace('\r\n', '\n').replace('\r', '\n')
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip() or '(nothing decoded — check mode and file)'


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODE — AcDense  (high-density: zlib + 9-tone RS(9,7) MFSK)
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_acdense(text: str, sr: int, volume: float) -> np.ndarray:
    """
    Encode any Unicode text (including emoji) as AcDense audio.

    Pipeline:
      text → UTF-8 bytes → zlib compress → 4-byte header → nibbles
        → RS(9,7) over GF(16) → 9-tone MFSK frames at 192ms per frame

    Throughput (typical English):
      Raw:       18.2 bytes/sec compressed (vs WaveSig 10.4 bytes/sec uncompressed)
      Effective: ~64 chars/sec after zlib (~6× WaveSig)

    Frame layout:
      400ms silence (pre-roll)
      2× frame_dur: marker tone at 15600 Hz (preamble)
      1× frame_dur: silence gap
      N× frame_dur: data frames (9 tones simultaneously)
      1× frame_dur: silence gap
      2× frame_dur: marker tone at 15600 Hz (postamble)
      400ms silence (post-roll for AAC tail safety)

    Header (4 bytes): [0xAC, 0xDE, len_hi, len_lo]
      len = len(zlib_compressed_bytes) as uint16 big-endian
    """
    import zlib

    frame_dur_samples = round(
        sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE
    )

    # ── Step 1: compress ────────────────────────────────────────────────────
    raw_bytes   = text.encode('utf-8')
    compressed  = zlib.compress(raw_bytes, level=9)
    comp_len    = len(compressed)

    # ── Step 2: build payload ───────────────────────────────────────────────
    # 4-byte header: magic 0xAC 0xDE + uint16 BE compressed length
    payload = bytearray()
    payload.extend(b'\xAC\xDE')
    payload.extend(struct.pack('>H', comp_len))
    payload.extend(compressed)

    # ── Step 3: nibbles ─────────────────────────────────────────────────────
    nibbles: list[int] = []
    for b in payload:
        nibbles.append((b >> 4) & 0xF)
        nibbles.append(b & 0xF)

    # ── Step 4: RS(9,7) encode — 7 data nibbles → 9 coded nibbles per frame ─
    coded: list[int] = []
    for i in range(0, len(nibbles), 7):
        block = list(nibbles[i:i + 7])
        while len(block) < 7:
            block.append(0)
        coded.extend(ACDENSE_RS.encode(block))

    # ── Step 5: synthesise audio ────────────────────────────────────────────
    segments: list[np.ndarray] = []

    # Pre-roll silence (400 ms)
    segments.append(np.zeros(round(sr * 0.4), dtype=np.float32))

    # Preamble: 2 frame-durations of marker tone at ACDENSE_MARKER Hz
    n_mark = frame_dur_samples * 2
    t_mark = np.arange(n_mark, dtype=np.float64) / sr
    marker_wave = (volume * np.sin(2.0 * np.pi * ACDENSE_MARKER * t_mark)).astype(np.float32)
    segments.append(marker_wave)

    # Silent gap (1 frame)
    segments.append(np.zeros(frame_dur_samples, dtype=np.float32))

    # Data frames: 9 coded nibbles → 9 simultaneous tones per frame
    n_frame = frame_dur_samples
    t_frame = np.arange(n_frame, dtype=np.float64) / sr
    fade_n  = min(int(round(sr * 0.010)), n_frame // 2)   # 10 ms fade
    env     = np.ones(n_frame, dtype=np.float64)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)
        env[:fade_n]        = ramp
        env[n_frame - fade_n:] = ramp[::-1]

    for i in range(0, len(coded), ACDENSE_N_TONES):
        group = list(coded[i:i + ACDENSE_N_TONES])
        while len(group) < ACDENSE_N_TONES:
            group.append(0)
        wave = np.zeros(n_frame, dtype=np.float64)
        for tone_idx, nib in enumerate(group):
            freq = ACDENSE_F0 + (nib + tone_idx * 16) * ACDENSE_DF
            wave += (volume / ACDENSE_N_TONES) * np.sin(2.0 * np.pi * freq * t_frame)
        segments.append((wave * env).astype(np.float32))

    # Silent gap (1 frame)
    segments.append(np.zeros(frame_dur_samples, dtype=np.float32))

    # Postamble: 2 frame-durations of marker tone
    segments.append(marker_wave)   # same waveform, same length

    # Post-roll silence (400 ms — mirrors video.py pre-roll, prevents AAC tail clip)
    segments.append(np.zeros(round(sr * 0.4), dtype=np.float32))

    audio = np.concatenate(segments)
    peak  = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio * (0.9 / peak)
    return audio.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — AcDense probe + decoder
# ═══════════════════════════════════════════════════════════════════════════════

def _probe_acdense(samples: np.ndarray, sr: int) -> bool:
    """
    Spectral fingerprint: detect AcDense by probing for marker at 15600 Hz.

    ACDENSE_MARKER (15600 Hz) is above ALL other AudioCipher modes:
        WaveSig  HI marker = 10800 Hz
        HZAlpha  maximum   =  8869 Hz
        DTMF     maximum   =  1633 Hz
        FSK      maximum   =  1200 Hz

    Skips leading silence (pre-roll) and probes the first two active frame
    groups where the marker plays exclusively.
    """
    pg = round(sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE)
    total_groups = len(samples) // pg
    if total_groups < 5:
        return False

    look_ahead = min(total_groups, 8)
    energies   = [
        float(np.sqrt(np.mean(samples[g * pg:(g + 1) * pg].astype(np.float64) ** 2)))
        for g in range(look_ahead)
    ]
    mean_e    = sum(energies) / len(energies) or 1.0
    threshold = mean_e * 0.2
    first_active = next((g for g, e in enumerate(energies) if e > threshold), -1)
    if first_active < 0:
        return False

    start = first_active * pg
    end   = start + 2 * pg
    if end > len(samples):
        return False
    probe = samples[start:end]

    # AcDense marker at ACDENSE_MARKER Hz; control is the band just above it
    hi   = fft_peak(probe, sr, ACDENSE_MARKER - ACDENSE_DF, ACDENSE_MARKER + ACDENSE_DF)
    ctrl = fft_peak(probe, sr, ACDENSE_MARKER + ACDENSE_DF, ACDENSE_MARKER + 3 * ACDENSE_DF)
    return hi['amp'] > 0.0 and hi['amp'] > ctrl['amp'] * 10.0


def _decode_acdense(samples: np.ndarray, sr: int) -> str:
    """
    Decode AcDense audio → UTF-8 text (full Unicode + emoji).

    Pipeline (inverse of _encode_acdense):
      FFT per frame → 9 tone positions → RS(9,7) decode → nibbles
        → bytes → zlib decompress → UTF-8 text
    """
    import zlib

    frame_dur_samples = round(
        sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE
    )

    # Tail padding — same fix as WaveSig: AAC trims stream end by ~380 samples
    tail = len(samples) % frame_dur_samples
    if tail >= frame_dur_samples // 2:
        samples = np.concatenate(
            [samples, np.zeros(frame_dur_samples - tail, dtype=samples.dtype)]
        )

    total_groups = len(samples) // frame_dur_samples
    if total_groups < 5:
        return '(audio too short for AcDense)'

    # RMS energy per group → find active signal region
    energies = []
    for g in range(total_groups):
        off = g * frame_dur_samples
        seg = samples[off:off + frame_dur_samples]
        energies.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))

    mean_e    = sum(energies) / len(energies)
    threshold = mean_e * 0.3

    first_active = next((g for g, e in enumerate(energies) if e > threshold), -1)
    last_active  = (
        len(energies) - 1
        - next((g for g, e in enumerate(reversed(energies)) if e > threshold), -1)
    )

    if first_active < 0 or last_active <= first_active + 4:
        return '(no AcDense signal detected)'

    # Frame layout: [2 preamble][1 gap][N data][1 gap][2 postamble]
    data_start = first_active + 3
    data_end   = last_active - 3    # inclusive

    if data_end < data_start:
        return '(no AcDense data found)'

    # ── Collect raw nibbles ─────────────────────────────────────────────────
    raw_nibbles: list[int] = []
    for g in range(data_start, data_end + 1):
        off   = g * frame_dur_samples
        frame = samples[off:off + frame_dur_samples]
        for tone_idx in range(ACDENSE_N_TONES):
            f_lo = ACDENSE_F0 + tone_idx * 16 * ACDENSE_DF
            f_hi = f_lo + 15 * ACDENSE_DF
            peak = fft_peak(frame, sr, f_lo, f_hi)
            nib  = max(0, min(15, round((peak['freq'] - f_lo) / ACDENSE_DF)))
            raw_nibbles.append(nib)

    # ── RS(9,7) decode ──────────────────────────────────────────────────────
    # Unlike WaveSig, AcDense carries binary (zlib) data — not ASCII.
    # Printability check is meaningless; just use RS correction and let
    # zlib CRC32 serve as the ultimate integrity check.
    decoded_nibbles: list[int] = []
    for i in range(0, len(raw_nibbles), 9):
        block = list(raw_nibbles[i:i + 9])
        if len(block) < 9:
            block += [0] * (9 - len(block))
        try:
            decoded_nibbles.extend(ACDENSE_RS.decode(block)[:7])
        except Exception:
            decoded_nibbles.extend(block[:7])   # raw fallback

    # ── Nibbles → bytes ─────────────────────────────────────────────────────
    bytes_out = bytearray()
    for i in range(0, len(decoded_nibbles) - 1, 2):
        bytes_out.append((decoded_nibbles[i] << 4) | decoded_nibbles[i + 1])

    if len(bytes_out) < 4:
        return '(AcDense: too few bytes decoded)'

    # ── Parse header ────────────────────────────────────────────────────────
    magic = bytes_out[:2]
    if magic != b'\xac\xde':
        return f'(AcDense: bad magic {magic.hex()} — wrong mode or decode error)'

    comp_len = struct.unpack('>H', bytes_out[2:4])[0]
    # Clamp in case of decode corruption
    comp_len = min(comp_len, max(0, len(bytes_out) - 4))
    payload  = bytes(bytes_out[4:4 + comp_len])

    if len(payload) < comp_len:
        return f'(AcDense: incomplete — got {len(payload)}/{comp_len} compressed bytes)'

    # ── zlib decompress → UTF-8 ──────────────────────────────────────────────
    try:
        return zlib.decompress(payload).decode('utf-8')
    except zlib.error as e:
        # If header was correct but zlib fails, likely a decode error — give
        # helpful context rather than a raw Python exception.
        return f'(AcDense zlib error: {e} — audio may be corrupted or wrong mode)'
    except UnicodeDecodeError as e:
        return f'(AcDense UTF-8 decode error: {e})'


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODE — ACHD  (HyperDense: zlib + 15-tone RS(15,11) MFSK)
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_achd(text: str, sr: int, volume: float) -> np.ndarray:
    """
    Encode any Unicode text as ACHD (AudioCipher HyperDense) audio.

    Identical pipeline to AcDense but with 15 simultaneous tones (vs 9)
    and RS(15,11) FEC correcting up to 2 symbol errors per frame (vs 1).

    Frequency layout:
      Tone n, position p: freq = 900 + (n×16 + p) × 75 Hz
      Tone 0:   900–2025 Hz
      Tone 14: 16725–17850 Hz
      Marker:  19200 Hz

    Header (4 bytes): [0xAC, 0x1D, len_hi, len_lo]
      where len = len(zlib_compressed_bytes) as uint16 big-endian

    Throughput: ~100 chars/sec for typical English (vs AcDense ~64 chars/sec)
    """
    import zlib

    frame_dur_samples = round(
        sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE
    )

    # ── Step 1: compress ────────────────────────────────────────────────────
    raw_bytes  = text.encode('utf-8')
    compressed = zlib.compress(raw_bytes, level=9)
    comp_len   = len(compressed)

    # ── Step 2: build payload ───────────────────────────────────────────────
    # 4-byte header: magic 0xAC 0x1D + uint16 BE compressed length
    payload = bytearray()
    payload.extend(ACHD_MAGIC)
    payload.extend(struct.pack('>H', comp_len))
    payload.extend(compressed)

    # ── Step 3: nibbles ─────────────────────────────────────────────────────
    nibbles: list[int] = []
    for b in payload:
        nibbles.append((b >> 4) & 0xF)
        nibbles.append(b & 0xF)

    # ── Step 4: RS(15,11) encode — 11 data nibbles → 15 coded nibbles/frame ─
    coded: list[int] = []
    for i in range(0, len(nibbles), 11):
        block = list(nibbles[i:i + 11])
        while len(block) < 11:
            block.append(0)
        coded.extend(ACHD_RS.encode(block))

    # ── Step 5: synthesise audio ────────────────────────────────────────────
    segments: list[np.ndarray] = []

    # Pre-roll silence (400 ms)
    segments.append(np.zeros(round(sr * 0.4), dtype=np.float32))

    # Preamble: 2 frame-durations of marker tone at ACHD_MARKER Hz
    n_mark  = frame_dur_samples * 2
    t_mark  = np.arange(n_mark, dtype=np.float64) / sr
    marker_wave = (volume * np.sin(2.0 * np.pi * ACHD_MARKER * t_mark)).astype(np.float32)
    segments.append(marker_wave)

    # Silent gap (1 frame)
    segments.append(np.zeros(frame_dur_samples, dtype=np.float32))

    # Data frames: 15 coded nibbles → 15 simultaneous tones per frame
    n_frame = frame_dur_samples
    t_frame = np.arange(n_frame, dtype=np.float64) / sr
    fade_n  = min(int(round(sr * 0.010)), n_frame // 2)   # 10 ms fade
    env     = np.ones(n_frame, dtype=np.float64)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)
        env[:fade_n]           = ramp
        env[n_frame - fade_n:] = ramp[::-1]

    for i in range(0, len(coded), ACHD_N_TONES):
        group = list(coded[i:i + ACHD_N_TONES])
        while len(group) < ACHD_N_TONES:
            group.append(0)
        wave = np.zeros(n_frame, dtype=np.float64)
        for tone_idx, nib in enumerate(group):
            freq = ACHD_F0 + (nib + tone_idx * 16) * ACHD_DF
            wave += (volume / ACHD_N_TONES) * np.sin(2.0 * np.pi * freq * t_frame)
        segments.append((wave * env).astype(np.float32))

    # Silent gap (1 frame)
    segments.append(np.zeros(frame_dur_samples, dtype=np.float32))

    # Postamble: 2 frame-durations of marker tone
    segments.append(marker_wave)

    # Post-roll silence (400 ms)
    segments.append(np.zeros(round(sr * 0.4), dtype=np.float32))

    audio = np.concatenate(segments)
    peak  = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio * (0.9 / peak)
    return audio.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DECODE — ACHD probe + decoder
# ═══════════════════════════════════════════════════════════════════════════════

def _probe_achd(samples: np.ndarray, sr: int) -> bool:
    """
    Spectral fingerprint: detect ACHD by probing for marker at 19200 Hz.

    ACHD_MARKER (19200 Hz) is above ALL other AudioCipher modes:
        AcDense  marker = 15600 Hz
        WaveSig  marker = 10800 Hz
        HZAlpha  max    =  8869 Hz

    Probes the first two active frame groups where the marker plays exclusively.
    """
    pg = round(sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE)
    total_groups = len(samples) // pg
    if total_groups < 5:
        return False

    look_ahead = min(total_groups, 8)
    energies   = [
        float(np.sqrt(np.mean(samples[g * pg:(g + 1) * pg].astype(np.float64) ** 2)))
        for g in range(look_ahead)
    ]
    mean_e    = sum(energies) / len(energies) or 1.0
    threshold = mean_e * 0.2
    first_active = next((g for g, e in enumerate(energies) if e > threshold), -1)
    if first_active < 0:
        return False

    start = first_active * pg
    end   = start + 2 * pg
    if end > len(samples):
        return False
    probe = samples[start:end]

    hi   = fft_peak(probe, sr, ACHD_MARKER - ACHD_DF, ACHD_MARKER + ACHD_DF)
    ctrl = fft_peak(probe, sr, ACHD_MARKER + ACHD_DF, ACHD_MARKER + 3 * ACHD_DF)
    return hi['amp'] > 0.0 and hi['amp'] > ctrl['amp'] * 10.0


def _decode_achd(samples: np.ndarray, sr: int) -> str:
    """
    Decode ACHD audio → UTF-8 text (full Unicode + emoji).

    Pipeline (inverse of _encode_achd):
      FFT per frame → 15 tone positions → RS(15,11) decode → nibbles
        → bytes → zlib decompress → UTF-8 text
    """
    import zlib

    frame_dur_samples = round(
        sr * GGWAVE_SAMPLES_PER_FRAME * GGWAVE_FRAMES_PER_TX / GGWAVE_SAMPLE_RATE
    )

    # Tail padding — AAC trims stream end by ~380 samples
    tail = len(samples) % frame_dur_samples
    if tail >= frame_dur_samples // 2:
        samples = np.concatenate(
            [samples, np.zeros(frame_dur_samples - tail, dtype=samples.dtype)]
        )

    total_groups = len(samples) // frame_dur_samples
    if total_groups < 5:
        return '(audio too short for ACHD)'

    # RMS energy per group → find active signal region
    energies = []
    for g in range(total_groups):
        off = g * frame_dur_samples
        seg = samples[off:off + frame_dur_samples]
        energies.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))

    mean_e    = sum(energies) / len(energies)
    threshold = mean_e * 0.3

    first_active = next((g for g, e in enumerate(energies) if e > threshold), -1)
    last_active  = (
        len(energies) - 1
        - next((g for g, e in enumerate(reversed(energies)) if e > threshold), -1)
    )

    if first_active < 0 or last_active <= first_active + 4:
        return '(no ACHD signal detected)'

    # Frame layout: [2 preamble][1 gap][N data][1 gap][2 postamble]
    data_start = first_active + 3
    data_end   = last_active - 3    # inclusive

    if data_end < data_start:
        return '(no ACHD data found)'

    # ── Collect raw nibbles ─────────────────────────────────────────────────
    raw_nibbles: list[int] = []
    for g in range(data_start, data_end + 1):
        off   = g * frame_dur_samples
        frame = samples[off:off + frame_dur_samples]
        for tone_idx in range(ACHD_N_TONES):
            f_lo = ACHD_F0 + tone_idx * 16 * ACHD_DF
            f_hi = f_lo + 15 * ACHD_DF
            peak = fft_peak(frame, sr, f_lo, f_hi)
            nib  = max(0, min(15, round((peak['freq'] - f_lo) / ACHD_DF)))
            raw_nibbles.append(nib)

    # ── RS(15,11) decode ─────────────────────────────────────────────────────
    decoded_nibbles: list[int] = []
    for i in range(0, len(raw_nibbles), 15):
        block = list(raw_nibbles[i:i + 15])
        if len(block) < 15:
            block += [0] * (15 - len(block))
        try:
            decoded_nibbles.extend(ACHD_RS.decode(block)[:11])
        except Exception:
            decoded_nibbles.extend(block[:11])   # raw fallback

    # ── Nibbles → bytes ─────────────────────────────────────────────────────
    bytes_out = bytearray()
    for i in range(0, len(decoded_nibbles) - 1, 2):
        bytes_out.append((decoded_nibbles[i] << 4) | decoded_nibbles[i + 1])

    if len(bytes_out) < 4:
        return '(ACHD: too few bytes decoded)'

    # ── Parse header ────────────────────────────────────────────────────────
    magic = bytes(bytes_out[:2])
    if magic != ACHD_MAGIC:
        return f'(ACHD: bad magic {magic.hex()} — wrong mode or decode error)'

    comp_len = struct.unpack('>H', bytes_out[2:4])[0]
    comp_len = min(comp_len, max(0, len(bytes_out) - 4))
    payload  = bytes(bytes_out[4:4 + comp_len])

    if len(payload) < comp_len:
        return f'(ACHD: incomplete — got {len(payload)}/{comp_len} compressed bytes)'

    # ── zlib decompress → UTF-8 ──────────────────────────────────────────────
    try:
        return zlib.decompress(payload).decode('utf-8')
    except zlib.error as e:
        return f'(ACHD zlib error: {e} — audio may be corrupted or wrong mode)'
    except UnicodeDecodeError as e:
        return f'(ACHD UTF-8 decode error: {e})'


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC DECODE API
# ═══════════════════════════════════════════════════════════════════════════════

def decode(audio_path: str, mode: str = 'auto',
           freq_map: dict | None = None, **kwargs) -> str:
    """
    Decode an AudioCipher WAV file to text.

    Args:
        audio_path: Path to WAV file
        mode:       'auto' | 'hzalpha' | 'morse' | 'dtmf' | 'fsk' | 'ggwave' | 'acdense' | 'achd' | 'custom'
                    'auto' reads embedded WAV metadata first; if absent, probes
                    spectral fingerprints in order: ACHD → AcDense → WaveSig → HZAlpha.
        freq_map:   Custom frequency map for mode='custom'
        **kwargs:   Override decode parameters (morse_wpm, fsk_baud, letter_gap_ms, etc.)

    Returns:
        Decoded text string
    """
    samples, sr = read_wav(audio_path)
    p = {**DEFAULTS, **kwargs}

    if mode == 'auto':
        meta = parse_wav_metadata(audio_path)
        if meta:
            mode = meta.get('mode', 'hzalpha')
            # Map JS camelCase keys → Python snake_case
            js_to_py = {
                'duration':   'duration_ms',
                'letterGap':  'letter_gap_ms',
                'wordGap':    'word_gap_ms',
                'fade':       'fade_ms',
                'morseFreq':  'morse_freq',
                'morseWpm':   'morse_wpm',
                'fsk_baud':   'fsk_baud',
                'waveform':   'waveform',
                'volume':     'volume',
            }
            for js_key, py_key in js_to_py.items():
                v = meta.get('params', {}).get(js_key)
                if v is not None:
                    p[py_key] = v
        else:
            # No embedded metadata — probe spectral fingerprints.
            # Priority: ACHD (19200 Hz) → AcDense (15600 Hz) → WaveSig (10800 Hz) → HZAlpha.
            # All markers survive AAC re-encoding on Twitter.
            if _probe_achd(samples, sr):
                mode = 'achd'
            elif _probe_acdense(samples, sr):
                mode = 'acdense'
            elif _probe_ggwave(samples, sr):
                mode = 'ggwave'
            else:
                mode = 'hzalpha'

    if mode in ('hzalpha', 'custom'):
        return _decode_hzalpha(samples, sr, p, freq_map=freq_map)
    elif mode == 'morse':
        return _decode_morse(samples, sr, p)
    elif mode == 'dtmf':
        return _decode_dtmf(samples, sr, p)
    elif mode == 'fsk':
        return _decode_fsk(samples, sr, p)
    elif mode == 'ggwave':
        return _decode_ggwave(samples, sr)
    elif mode == 'acdense':
        return _decode_acdense(samples, sr)
    elif mode == 'achd':
        return _decode_achd(samples, sr)
    else:
        return f'(unknown mode: {mode})'
