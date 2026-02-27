#!/usr/bin/env python3
"""
test_acdense.py — AcDense cipher round-trip test suite.

Tests:
  1. RS_GF16 encode/decode correctness (no audio, pure math)
  2. Short ASCII round-trip
  3. Emoji round-trip
  4. Long multi-paragraph round-trip
  5. Mixed Unicode (em-dash, curly quotes, etc.)
  6. Auto-detect from WAV (no mode specified)
  7. Throughput benchmarks vs WaveSig
"""
from __future__ import annotations
import sys, os, time, tempfile
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import soundfile as sf

from utils import RS_GF16, ACDENSE_RS, GF16
from cipher import encode, decode


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'

def check(label: str, got, expected):
    ok = got == expected
    icon = PASS if ok else FAIL
    print(f'  {icon}  {label}')
    if not ok:
        print(f'       got:      {repr(got)[:120]}')
        print(f'       expected: {repr(expected)[:120]}')
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 1. RS_GF16 math test
# ─────────────────────────────────────────────────────────────────────────────

def test_rs_math():
    print('\n━━ 1. RS_GF16(9,7) math round-trip ━━')
    rs = RS_GF16(9, 7)
    all_ok = True

    # Round-trip with no errors
    for trial in range(50):
        data = [int(x) for x in np.random.randint(0, 16, 7)]
        codeword = rs.encode(data)
        assert len(codeword) == 9
        recovered = rs.decode(codeword)
        assert recovered[:7] == data, f'clean round-trip failed: {data} → {codeword} → {recovered}'

    ok = check('50× random clean encode/decode', True, True)
    all_ok &= ok

    # Single error correction
    ok_count = 0
    for trial in range(50):
        data     = [int(x) for x in np.random.randint(0, 16, 7)]
        codeword = list(rs.encode(data))
        err_pos  = int(np.random.randint(0, 9))
        err_val  = (codeword[err_pos] + 1 + int(np.random.randint(1, 15))) % 16
        codeword[err_pos] = err_val
        recovered = rs.decode(codeword)
        if recovered[:7] == data:
            ok_count += 1
    ok = check(f'50× single-symbol error correction: {ok_count}/50 correct', ok_count, 50)
    all_ok &= ok

    # Verify RS(12,8) still works (backward compat)
    for trial in range(20):
        data     = [int(x) for x in np.random.randint(0, 16, 8)]
        codeword = GF16.encode(data)
        recovered = GF16.decode(codeword)
        assert recovered == data, f'GF16 RS(12,8) regression: {data}'
    ok = check('20× GF16 RS(12,8) backward compat', True, True)
    all_ok &= ok

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 2. Audio round-trip helper
# ─────────────────────────────────────────────────────────────────────────────

def audio_roundtrip(text: str, mode: str = 'acdense', decode_mode: str = 'auto') -> tuple[str, float, float]:
    """Encode text → WAV → decode. Returns (decoded_text, duration_secs, bytes_per_sec_raw)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        t0 = time.time()
        samples = encode(text, mode=mode, sr=44100)
        duration = len(samples) / 44100
        sf.write(path, samples, 44100)
        decoded  = decode(path, mode=decode_mode)
        elapsed  = time.time() - t0

        # Raw compressed size (approx)
        import zlib
        comp = len(zlib.compress(text.encode('utf-8'), level=9))
        raw_bps = comp / duration if duration > 0 else 0

        return decoded, duration, raw_bps
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Round-trip tests
# ─────────────────────────────────────────────────────────────────────────────

def test_roundtrips():
    print('\n━━ 2. Audio round-trip tests ━━')
    all_ok = True

    cases = [
        ('Short ASCII',
         'HELLO WORLD! This is a test of AcDense.'),

        ('Numbers + punctuation',
         'Year 2024: 42% increase in Q3 ($1,337.00).'),

        ('Newlines preserved',
         'Line one.\nLine two.\nLine three.\n\nNew paragraph.'),

        ('Emoji',
         'Hello 🌍🎵🔮💀 World! 🤖✨🎸🔥'),

        ('Mixed emoji + text',
         'NULL BOT 🤖 — audiocipher test 🎵\n'
         'Signal: 📡 | Status: ✅ | Score: 💯'),

        ('Unicode chars (em-dash, curly quotes)',
         'The signal — as measured by "the oracle" — was strong.'),

        ('Japanese text',
         'こんにちは世界 — Hello World in Japanese'),

        ('Multi-paragraph',
         'THE TRANSMISSION BEGINS NOW.\n\n'
         'Deep in the lattice of frequencies there are signals '
         'that most humans cannot perceive. These tones encode '
         'messages in plain sight — hidden within the noise.\n\n'
         'The cipher does not conceal. It reveals.'),
    ]

    for name, text in cases:
        decoded, dur, bps = audio_roundtrip(text)
        ok = check(
            f'{name} [{len(text)}c, {dur:.1f}s, {bps:.0f}B/s raw compressed]',
            decoded, text
        )
        all_ok &= ok

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 4. Auto-detect test (no mode hint)
# ─────────────────────────────────────────────────────────────────────────────

def test_auto_detect():
    print('\n━━ 3. Auto-detect probe test ━━')
    all_ok = True

    # AcDense probe should correctly identify AcDense WAV
    text = 'Auto-detect test: can the probe find me? 🔍'
    decoded, dur, _ = audio_roundtrip(text, mode='acdense', decode_mode='auto')
    ok = check(f'AcDense auto-detect from WAV ({dur:.1f}s)', decoded, text)
    all_ok &= ok

    # WaveSig probe should NOT falsely trigger on AcDense
    from cipher import _probe_acdense, _probe_ggwave
    from utils import read_wav
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        samples = encode('test', mode='acdense', sr=44100)
        sf.write(path, samples, 44100)
        s, sr = read_wav(path)
        ok1 = check('AcDense probe TRUE  on AcDense file', _probe_acdense(s, sr), True)
        ok2 = check('WaveSig probe FALSE on AcDense file', _probe_ggwave(s, sr),   False)
        all_ok &= ok1 & ok2
    finally:
        try: os.unlink(path)
        except: pass

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    try:
        samples = encode('test', mode='ggwave', sr=44100)
        sf.write(path, samples, 44100)
        s, sr = read_wav(path)
        ok1 = check('AcDense probe FALSE on WaveSig file', _probe_acdense(s, sr), False)
        ok2 = check('WaveSig probe TRUE  on WaveSig file', _probe_ggwave(s, sr),   True)
        all_ok &= ok1 & ok2
    finally:
        try: os.unlink(path)
        except: pass

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# 5. Throughput benchmark
# ─────────────────────────────────────────────────────────────────────────────

def test_benchmarks():
    print('\n━━ 4. Throughput benchmarks ━━')

    import zlib

    paragraph = (
        'In the architecture of sound, every frequency carries meaning. '
        'The cipher does not hide — it encodes. And in that encoding, '
        'a new language emerges: one that lives in the spectrogram, '
        'visible only to those who know where to look.'
    )
    long_text = '\n\n'.join([paragraph] * 4)  # ~4 paragraphs

    emoji_text = '🌍🎵🔮💀🤖✨🎸🔥📡✅💯🎯🛸🌊⚡🎭🔑🎬🌙🦾' * 5  # 100 emoji

    tests = [
        ('1 paragraph', paragraph),
        ('4 paragraphs', long_text),
        ('100 emoji', emoji_text),
    ]

    print(f'\n  {"Text":<20} {"Chars":>6} {"UTF8B":>6} {"zlib":>6} {"Audio":>7} {"Eff.chars/s":>12}')
    print(f'  {"-"*20} {"-"*6} {"-"*6} {"-"*6} {"-"*7} {"-"*12}')

    for name, text in tests:
        samples = encode(text, mode='acdense', sr=44100)
        dur = len(samples) / 44100
        utf8b = len(text.encode('utf-8'))
        zlibb = len(zlib.compress(text.encode('utf-8'), level=9))
        eff_cps = len(text) / dur
        print(f'  {name:<20} {len(text):>6} {utf8b:>6} {zlibb:>6} {dur:>6.1f}s {eff_cps:>11.1f}/s')

    print()
    # Compare WaveSig for same short text (ASCII)
    gg_samples = encode(paragraph, mode='ggwave', sr=44100)
    gg_dur = len(gg_samples) / 44100
    ac_samples = encode(paragraph, mode='acdense', sr=44100)
    ac_dur = len(ac_samples) / 44100
    print(f'  WaveSig: {paragraph[:40]!r}... → {gg_dur:.1f}s')
    print(f'  AcDense: same text                     → {ac_dur:.1f}s  ({gg_dur/ac_dur:.1f}× faster)')
    print()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('═' * 60)
    print(' AcDense cipher test suite')
    print('═' * 60)

    results = []
    results.append(test_rs_math())
    results.append(test_roundtrips())
    results.append(test_auto_detect())
    results.append(test_benchmarks())

    print('\n' + '═' * 60)
    if all(results):
        print(f' {PASS}  All tests passed!')
    else:
        n_fail = results.count(False)
        print(f' {FAIL}  {n_fail} test group(s) failed.')
    print('═' * 60)
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
