# test_image2audio.py
"""Tests for smart_optimize_params() and image_to_audio_with_preview()."""
from __future__ import annotations
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image
import numpy as np

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
_failures = []

def check(label, cond):
    if cond:
        print(f'  {PASS} {label}')
    else:
        print(f'  {FAIL} {label}')
        _failures.append(label)


def test_smart_optimize_params():
    from spectrogram import smart_optimize_params
    print('smart_optimize_params()')

    # 800×400 image → num_cols=800, num_bins=400, duration=800*0.016=12.8
    p = smart_optimize_params(800, 400)
    check('num_cols = min(800, 1024) = 800', p['num_cols'] == 800)
    check('num_bins = min(400, 512) = 400', p['num_bins'] == 400)
    check('duration = 12.8', abs(p['duration'] - 12.8) < 0.01)
    check('fmin = 200', p['fmin'] == 200.0)
    check('fmax = 16000', p['fmax'] == 16000.0)

    # Wide image: 2000×100 → num_cols capped at 1024, duration = 1024*0.016 = 16.384
    p2 = smart_optimize_params(2000, 100)
    check('wide: num_cols capped at 1024', p2['num_cols'] == 1024)
    check('wide: duration = 16.4', abs(p2['duration'] - 16.4) < 0.01)

    # Tall image: 50×2000 → num_bins capped at 512
    p3 = smart_optimize_params(50, 2000)
    check('tall: num_bins capped at 512', p3['num_bins'] == 512)

    # Tiny image: 10×10 → duration clamped to 2.0 (10*0.016=0.16 < 2)
    p4 = smart_optimize_params(10, 10)
    check('tiny: duration clamped to 2.0', p4['duration'] == 2.0)
    check('duration always >= 2', p4['duration'] >= 2.0)
    check('duration always <= 30', p4['duration'] <= 30.0)


if __name__ == '__main__':
    test_smart_optimize_params()
    if _failures:
        print(f'\n{len(_failures)} failure(s): {_failures}')
        sys.exit(1)
    else:
        print('\nAll tests passed.')
