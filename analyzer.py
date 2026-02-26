"""
analyzer.py — Full hidden-content detection pipeline for AudioCipher.

Pipeline (6 steps):
  1. Generate high-res spectrogram image from audio (librosa STFT)
  2. QR / barcode detection (pyzbar) — tries raw, inverted, thresholded variants
  3. OCR text extraction (pytesseract) — full image + per-contour regions
  4. Contour / shape detection (OpenCV) — find structured regions
  5. Cipher tone auto-detection — match HZAlpha / Morse / DTMF / FSK
  6. Anomaly / entropy detection — narrow sustained tones, periodic patterns

Each finding is saved as a cropped PNG (noise-stripped, padded).
A consolidated results.json is written to output_dir/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    HZALPHA_MAP, HZ_SHIFT_FREQ,
    DTMF_KEY_MAP,
    FSK_F0, FSK_F1, FSK_BAUD,
    fft_peak,
    read_wav,
)
from spectrogram import audio_to_spectrogram_fast


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(
    audio_path: str,
    output_dir: str = './findings/',
    detect_qr:      bool = True,
    detect_ocr:     bool = True,
    detect_cipher:  bool = True,
    detect_anomaly: bool = True,
    crop:           bool = True,
) -> list[dict[str, Any]]:
    """
    Analyze audio for hidden content.

    Returns:
        List of finding dicts, each containing:
          type, confidence, freq_range_hz, time_range_s,
          bounding_box (x,y,w,h in spectrogram pixels),
          decoded_value (if applicable),
          crop_path (if crop=True),
          notes
    """
    output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    samples, sr = read_wav(audio_path)
    duration = len(samples) / sr

    # ── Step 1: High-res spectrogram ─────────────────────────────────────────
    # We generate two versions:
    #   spec_img  — coloured (for saving / display)
    #   grey_img  — grayscale (for CV / OCR / QR)
    spec_w, spec_h = 2048, 1024
    fft_size = 8192

    spec_img = audio_to_spectrogram_fast(
        audio_path, fft_size=fft_size, fmin=0, fmax=None,
        colormap='green', width=spec_w, height=spec_h, gain_db=6.0,
    )

    # Grayscale version (average of RGB channels)
    import PIL.Image  # type: ignore
    grey_img = spec_img.convert('L')
    grey_arr = np.array(grey_img, dtype=np.uint8)  # [h, w]

    findings: list[dict[str, Any]] = []
    crop_idx = [0]

    def _save_crop(region_arr: np.ndarray, tag: str) -> str:
        """Strip noise floor, apply morphological close, pad, save PNG."""
        # Noise floor threshold (< 8 % of max → black)
        thr = max(20, int(region_arr.max() * 0.08))
        clean = region_arr.copy()
        clean[clean < thr] = 0

        try:
            import cv2  # type: ignore
            kernel = np.ones((3, 3), np.uint8)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        except ImportError:
            pass

        PAD = 12
        h, w = clean.shape
        padded = np.zeros((h + PAD * 2, w + PAD * 2), dtype=np.uint8)
        padded[PAD:PAD + h, PAD:PAD + w] = clean

        crop_idx[0] += 1
        path = os.path.join(output_dir, f'crop_{crop_idx[0]:03d}_{tag}.png')
        PIL.Image.fromarray(padded, mode='L').save(path)
        return path

    # ── Step 2: QR / Barcode detection ──────────────────────────────────────
    if detect_qr:
        qr_findings = _detect_qr(grey_arr, spec_img, duration, sr,
                                  crop, _save_crop)
        findings.extend(qr_findings)

    # ── Step 3: OCR ──────────────────────────────────────────────────────────
    if detect_ocr:
        ocr_findings = _detect_ocr(grey_arr, spec_img, duration, sr,
                                   crop, _save_crop)
        findings.extend(ocr_findings)

    # ── Step 4: Contour / shape detection ────────────────────────────────────
    contour_regions = _detect_contours(grey_arr, spec_w, spec_h,
                                        duration, sr, crop, _save_crop)
    findings.extend(contour_regions)

    # ── Step 5: Cipher tone auto-detection ───────────────────────────────────
    if detect_cipher:
        cipher_findings = _detect_cipher_tones(samples, sr, duration)
        findings.extend(cipher_findings)

    # ── Step 6: Anomaly / entropy ─────────────────────────────────────────────
    if detect_anomaly:
        anomaly_findings = _detect_anomalies(samples, sr, duration, fft_size)
        findings.extend(anomaly_findings)

    # ── Write JSON ────────────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(findings, f, indent=2)

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — QR / Barcode detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_qr(
    grey_arr: np.ndarray,
    spec_img,
    duration: float,
    sr: int,
    do_crop: bool,
    save_crop,
) -> list[dict]:
    findings = []
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode  # type: ignore
        import PIL.Image  # type: ignore
        import cv2  # type: ignore
    except ImportError:
        return findings

    spec_h, spec_w = grey_arr.shape

    # Variants to try: raw, inverted, thresholded, high-contrast
    variants: list[tuple[str, np.ndarray]] = [
        ('raw',        grey_arr),
        ('inverted',   255 - grey_arr),
        ('thresh',     cv2.adaptiveThreshold(
                           grey_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY, 31, 5)),
        ('thresh_inv', cv2.adaptiveThreshold(
                           255 - grey_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY, 31, 5)),
    ]

    # Also try each 1/4 frequency sub-band
    for i in range(4):
        y0 = i * spec_h // 4
        y1 = (i + 1) * spec_h // 4
        band = grey_arr[y0:y1, :]
        variants.append((f'band{i}', band))
        variants.append((f'band{i}_inv', 255 - band))

    seen_data: set[str] = set()
    for variant_name, img_arr in variants:
        pil_img = PIL.Image.fromarray(img_arr)
        results = pyzbar_decode(pil_img)
        for obj in results:
            try:
                data_str = obj.data.decode('utf-8', errors='replace')
            except Exception:
                data_str = repr(obj.data)
            if data_str in seen_data:
                continue
            seen_data.add(data_str)

            rect = obj.rect
            # Map from the sub-band's coordinate to full-image coordinates
            # (variants from sub-bands have offset y0)
            y_off = 0
            if variant_name.startswith('band') and not variant_name.endswith('_inv'):
                band_idx = int(re.search(r'\d+', variant_name).group())
                y_off = band_idx * spec_h // 4
            elif variant_name.startswith('band') and variant_name.endswith('_inv'):
                band_idx = int(re.search(r'\d+', variant_name).group())
                y_off = band_idx * spec_h // 4

            x, y, w, h = rect.left, rect.top + y_off, rect.width, rect.height
            freq_lo = _y_to_freq(y + h, spec_h, sr)
            freq_hi = _y_to_freq(y,     spec_h, sr)
            t_lo    = _x_to_time(x,     spec_w, duration)
            t_hi    = _x_to_time(x + w, spec_w, duration)

            finding: dict = {
                'type':          'qr_code',
                'confidence':    1.0,
                'bounding_box':  {'x': x, 'y': y, 'w': w, 'h': h},
                'freq_range_hz': [round(freq_lo, 1), round(freq_hi, 1)],
                'time_range_s':  [round(t_lo, 3), round(t_hi, 3)],
                'decoded_value': data_str,
                'notes':         f'pyzbar detected {obj.type} in {variant_name} variant',
            }

            if do_crop:
                pad = 20
                cy0 = max(0, y - pad)
                cy1 = min(spec_h, y + h + pad)
                cx0 = max(0, x - pad)
                cx1 = min(spec_w, x + w + pad)
                region = grey_arr[cy0:cy1, cx0:cx1]
                finding['crop_path'] = save_crop(region, 'qr')

            findings.append(finding)

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — OCR text extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_ocr(
    grey_arr: np.ndarray,
    spec_img,
    duration: float,
    sr: int,
    do_crop: bool,
    save_crop,
) -> list[dict]:
    findings = []
    try:
        import pytesseract  # type: ignore
        import cv2  # type: ignore
        import PIL.Image  # type: ignore
    except ImportError:
        return findings

    spec_h, spec_w = grey_arr.shape

    # Enhance contrast before OCR
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grey_arr)

    # Try OCR on full spectrogram (various psm modes)
    seen_texts: set[str] = set()
    # Use image_to_data with per-word confidence filtering (conf > 70)
    # instead of image_to_string — dramatically reduces hallucinations on
    # spectrogram images where text rarely covers more than a few pixels.
    configs = [
        '--oem 3 --psm 6',   # uniform block of text
        '--oem 3 --psm 11',  # sparse text
    ]
    for cfg in configs:
        try:
            data_out = pytesseract.image_to_data(
                PIL.Image.fromarray(enhanced), config=cfg,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        n_items = len(data_out.get('text', []))
        for idx in range(n_items):
            conf = int(data_out['conf'][idx])
            word = str(data_out['text'][idx]).strip()
            if conf < 70 or len(word) < 3:
                continue
            if not re.search(r'[A-Za-z0-9]{3,}', word):
                continue
            if word in seen_texts:
                continue
            seen_texts.add(word)
            findings.append({
                'type':          'text',
                'confidence':    round(conf / 100.0, 2),
                'bounding_box':  {'x': 0, 'y': 0, 'w': spec_w, 'h': spec_h},
                'freq_range_hz': [0.0, float(sr // 2)],
                'time_range_s':  [0.0, round(duration, 3)],
                'decoded_value': word,
                'notes':         f'pytesseract full-image OCR (conf={conf})',
            })

    # Also try OCR on each detected contour region (see step 4 helper)
    contours_bb = _find_contour_bboxes(grey_arr)
    for (cx, cy, cw, ch) in contours_bb:
        if cw < 20 or ch < 10:
            continue
        region = enhanced[cy:cy + ch, cx:cx + cw]
        # Scale up small regions for better OCR
        scale = max(1, 200 // max(cw, ch))
        if scale > 1:
            region = cv2.resize(region, (cw * scale, ch * scale),
                                interpolation=cv2.INTER_CUBIC)
        try:
            # Per-word confidence filtering prevents hallucinated text
            data_out = pytesseract.image_to_data(
                PIL.Image.fromarray(region), config='--oem 3 --psm 8',
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        n_items = len(data_out.get('text', []))
        for idx in range(n_items):
            conf = int(data_out['conf'][idx])
            word = str(data_out['text'][idx]).strip()
            if conf < 70 or len(word) < 3:
                continue
            if not re.search(r'[A-Za-z0-9]{3,}', word):
                continue
            if word in seen_texts:
                continue
            seen_texts.add(word)
            freq_lo = _y_to_freq(cy + ch, spec_h, sr)
            freq_hi = _y_to_freq(cy,       spec_h, sr)
            t_lo    = _x_to_time(cx,        spec_w, duration)
            t_hi    = _x_to_time(cx + cw,   spec_w, duration)
            finding = {
                'type':          'text',
                'confidence':    round(conf / 100.0, 2),
                'bounding_box':  {'x': cx, 'y': cy, 'w': cw, 'h': ch},
                'freq_range_hz': [round(freq_lo, 1), round(freq_hi, 1)],
                'time_range_s':  [round(t_lo, 3), round(t_hi, 3)],
                'decoded_value': word,
                'notes':         f'pytesseract contour-region OCR (conf={conf})',
            }
            if do_crop:
                finding['crop_path'] = save_crop(
                    grey_arr[cy:cy + ch, cx:cx + cw], 'text'
                )
            findings.append(finding)

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Contour / shape detection (OpenCV)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_contour_bboxes(grey_arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) for significant contours."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return []
    thresh = cv2.adaptiveThreshold(
        grey_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10,
    )
    kernel  = np.ones((5, 5), np.uint8)
    closed  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes  = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 2000:        # raised from 500 — eliminates tiny noise contours
            continue
        ar = w / max(h, 1)
        if ar < 0.1 or ar > 10.0:
            continue
        bboxes.append((x, y, w, h))
    return bboxes


def _detect_contours(
    grey_arr: np.ndarray,
    spec_w: int,
    spec_h: int,
    duration: float,
    sr: int,
    do_crop: bool,
    save_crop,
) -> list[dict]:
    findings = []
    bboxes = _find_contour_bboxes(grey_arr)
    for (x, y, w, h) in bboxes:
        region = grey_arr[y:y + h, x:x + w]
        # Score "internal structure": std dev of pixel values
        std  = float(np.std(region))
        mean = float(np.mean(region))
        # Low-detail / uniform regions are likely noise.
        # Threshold raised to 0.35 (was 0.15) to cut uniform spectrogram bands.
        structure_score = min(1.0, std / max(mean, 1.0))
        if structure_score < 0.35:
            continue

        freq_lo = _y_to_freq(y + h, spec_h, sr)
        freq_hi = _y_to_freq(y,     spec_h, sr)
        t_lo    = _x_to_time(x,     spec_w, duration)
        t_hi    = _x_to_time(x + w, spec_w, duration)

        finding: dict = {
            'type':          'image',
            'confidence':    round(structure_score, 3),
            'bounding_box':  {'x': x, 'y': y, 'w': w, 'h': h},
            'freq_range_hz': [round(freq_lo, 1), round(freq_hi, 1)],
            'time_range_s':  [round(t_lo, 3), round(t_hi, 3)],
            'decoded_value': None,
            'notes':         f'structured region (std/mean={structure_score:.2f})',
        }
        if do_crop:
            finding['crop_path'] = save_crop(region, 'img')
        findings.append(finding)

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Cipher tone auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_cipher_tones(
    samples: np.ndarray,
    sr: int,
    duration: float,
) -> list[dict]:
    """
    Sliding window FFT across audio.
    Checks for HZAlpha, Morse, DTMF, and FSK tone presence.
    Returns findings for modes where confidence > threshold.
    """
    findings = []

    # Build HZAlpha frequency list (with shift marker)
    hz_freqs = [v for v in HZALPHA_MAP.values() if v > 0] + [HZ_SHIFT_FREQ]
    sorted_hz = sorted(hz_freqs)
    min_gap   = min(sorted_hz[i+1] - sorted_hz[i] for i in range(len(sorted_hz) - 1))
    hz_tol    = min_gap * 0.45

    DTMF_ROWS = [697, 770, 852, 941]
    DTMF_COLS = [1209, 1336, 1477, 1633]

    win_ms      = 200        # ms per analysis window
    hop_ms      = 100        # ms hop
    win_samples = max(4096, int(sr * win_ms / 1000))
    hop_samples = max(2048, int(sr * hop_ms / 1000))

    hz_hits   = 0
    dtmf_hits = 0
    fsk_hits  = 0
    total_windows = 0

    for i in range(0, len(samples) - win_samples, hop_samples):
        frame = samples[i:i + win_samples]
        rms   = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms < 0.003:
            continue

        total_windows += 1
        peak = fft_peak(frame, sr, min(hz_freqs) * 0.9, min(HZ_SHIFT_FREQ * 1.1, sr / 2 - 100))
        f    = peak['freq']

        # HZAlpha: does peak match any mapped frequency?
        for hf in hz_freqs:
            if abs(f - hf) < hz_tol:
                hz_hits += 1
                break

        # DTMF: does a row freq + col freq appear simultaneously?
        row_match = any(
            abs(fft_peak(frame, sr, rf - 60, rf + 60)['freq'] - rf) < 60
            for rf in DTMF_ROWS
        )
        col_match = any(
            abs(fft_peak(frame, sr, cf - 80, cf + 80)['freq'] - cf) < 80
            for cf in DTMF_COLS
        )
        if row_match and col_match:
            dtmf_hits += 1

        # FSK: energy at F0 or F1
        p0 = fft_peak(frame, sr, FSK_F0 - 80, FSK_F0 + 80)
        p1 = fft_peak(frame, sr, FSK_F1 - 80, FSK_F1 + 80)
        dominant_amp = max(p0['amp'], p1['amp'])
        if dominant_amp > peak['amp'] * 0.6:
            fsk_hits += 1

    if total_windows == 0:
        return findings

    hz_conf   = hz_hits   / total_windows
    dtmf_conf = dtmf_hits / total_windows
    fsk_conf  = fsk_hits  / total_windows

    # Report cipher detections above 65 % match rate.
    # 40 % was too permissive — random music hits HZAlpha/DTMF freqs by coincidence.
    THRESHOLD = 0.65

    if hz_conf >= THRESHOLD:
        try:
            from cipher import decode as cipher_decode  # type: ignore
            import tempfile, soundfile as sf  # type: ignore
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, samples, sr)
                decoded = cipher_decode(tmp.name, mode='hzalpha')
            os.unlink(tmp.name)
        except Exception as e:
            decoded = f'(decode error: {e})'
        findings.append({
            'type':          'cipher_hzalpha',
            'confidence':    round(hz_conf, 3),
            'bounding_box':  None,
            'freq_range_hz': [round(min(hz_freqs), 1), round(HZ_SHIFT_FREQ, 1)],
            'time_range_s':  [0.0, round(duration, 3)],
            'decoded_value': decoded,
            'notes':         f'{hz_hits}/{total_windows} windows matched HZAlpha frequencies',
        })

    if dtmf_conf >= THRESHOLD:
        try:
            from cipher import decode as cipher_decode  # type: ignore
            import tempfile, soundfile as sf  # type: ignore
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, samples, sr)
                decoded = cipher_decode(tmp.name, mode='dtmf')
            os.unlink(tmp.name)
        except Exception as e:
            decoded = f'(decode error: {e})'
        findings.append({
            'type':          'cipher_dtmf',
            'confidence':    round(dtmf_conf, 3),
            'bounding_box':  None,
            'freq_range_hz': [697.0, 1633.0],
            'time_range_s':  [0.0, round(duration, 3)],
            'decoded_value': decoded,
            'notes':         f'{dtmf_hits}/{total_windows} windows matched DTMF row+col pairs',
        })

    if fsk_conf >= THRESHOLD and hz_conf < THRESHOLD:
        try:
            from cipher import decode as cipher_decode  # type: ignore
            import tempfile, soundfile as sf  # type: ignore
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, samples, sr)
                decoded = cipher_decode(tmp.name, mode='fsk')
            os.unlink(tmp.name)
        except Exception as e:
            decoded = f'(decode error: {e})'
        findings.append({
            'type':          'cipher_fsk',
            'confidence':    round(fsk_conf, 3),
            'bounding_box':  None,
            'freq_range_hz': [float(FSK_F0), float(FSK_F1)],
            'time_range_s':  [0.0, round(duration, 3)],
            'decoded_value': decoded,
            'notes':         f'{fsk_hits}/{total_windows} windows dominated by FSK F0/F1',
        })

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Anomaly / entropy detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_anomalies(
    samples: np.ndarray,
    sr: int,
    duration: float,
    fft_size: int = 8192,
) -> list[dict]:
    """
    Divide the spectrogram into 8 frequency bands.
    Flag bands exhibiting:
      - Narrow sustained tones (low bandwidth, high consistent energy)
      - Repeating periodic patterns (autocorrelation peak)
      - Sharp start/end edges relative to neighbours
    """
    findings = []
    n = len(samples)
    if n < fft_size * 4:
        return findings

    NUM_BANDS = 8
    nyquist   = sr / 2.0
    band_hz   = nyquist / NUM_BANDS

    # Compute RMS energy profile per band over time (hop = fft_size // 4)
    hop = fft_size // 4
    n_frames = (n - fft_size) // hop

    band_rms: list[list[float]] = [[] for _ in range(NUM_BANDS)]

    for fi in range(n_frames):
        frame = samples[fi * hop: fi * hop + fft_size].astype(np.float64)
        spec  = np.abs(np.fft.rfft(frame * np.hanning(fft_size)))
        bin_hz = sr / fft_size

        for bi in range(NUM_BANDS):
            f_lo = bi * band_hz
            f_hi = f_lo + band_hz
            lo_bin = int(f_lo / bin_hz)
            hi_bin = min(int(f_hi / bin_hz), len(spec) - 1)
            rms = float(np.sqrt(np.mean(spec[lo_bin:hi_bin] ** 2))) if hi_bin > lo_bin else 0.0
            band_rms[bi].append(rms)

    for bi in range(NUM_BANDS):
        rms_arr = np.array(band_rms[bi], dtype=np.float64)
        if rms_arr.max() < 1e-9:
            continue
        rms_norm = rms_arr / rms_arr.max()

        mean_e = float(rms_norm.mean())
        std_e  = float(rms_norm.std())

        notes: list[str] = []
        confidence = 0.0

        # Narrow sustained tone: high mean energy AND low variance
        if mean_e > 0.25 and std_e < 0.2:
            confidence = max(confidence, min(1.0, mean_e * 2.0 * (1.0 - std_e)))
            notes.append(f'sustained tone (mean={mean_e:.2f}, std={std_e:.2f})')

        # Repeating pattern: autocorrelation of the RMS profile
        if len(rms_norm) > 20:
            ac = np.correlate(rms_norm - mean_e, rms_norm - mean_e, mode='full')
            ac = ac[len(ac) // 2:]   # keep positive lags
            # Normalise
            if ac[0] > 0:
                ac = ac / ac[0]
            # Look for a secondary peak (besides lag 0) > 0.85.
            # 0.6 was too low — virtually all music and speech has periodicity
            # that strong, leading to constant false positives.
            if len(ac) > 4:
                secondary_peak = float(np.max(ac[2:len(ac) // 2]))
                if secondary_peak > 0.85:
                    confidence = max(confidence, secondary_peak)
                    notes.append(f'repeating pattern (autocorr peak={secondary_peak:.2f})')

        # Sharp edge: sudden onset or end (ratio of first/last quartile to middle)
        q   = max(1, len(rms_norm) // 4)
        mid = float(rms_norm[q:-q].mean()) if len(rms_norm) > 2 * q else mean_e
        if mid > 0.05:
            start_ratio = float(rms_norm[:q].mean()) / mid
            end_ratio   = float(rms_norm[-q:].mean()) / mid
            edge_score  = max(abs(start_ratio - 1.0), abs(end_ratio - 1.0))
            if edge_score > 0.5:
                confidence = max(confidence, min(0.9, edge_score))
                notes.append(f'sharp edge (ratio={edge_score:.2f})')

        if confidence < 0.4 or not notes:
            continue

        f_lo = bi * band_hz
        f_hi = f_lo + band_hz

        # Find time range where band is active (> 10 % of peak)
        active = np.where(rms_norm > 0.1)[0]
        if len(active) > 0:
            t_lo = float(active[0]) * hop / sr
            t_hi = float(active[-1]) * hop / sr
        else:
            t_lo, t_hi = 0.0, duration

        findings.append({
            'type':          'anomaly',
            'confidence':    round(confidence, 3),
            'bounding_box':  None,
            'freq_range_hz': [round(f_lo, 1), round(f_hi, 1)],
            'time_range_s':  [round(t_lo, 3), round(t_hi, 3)],
            'decoded_value': None,
            'notes':         '; '.join(notes),
        })

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _y_to_freq(y: int, spec_h: int, sr: int) -> float:
    """Convert spectrogram pixel y (row 0 = top = high freq) to Hz."""
    nyquist = sr / 2.0
    frac    = 1.0 - (y / max(spec_h - 1, 1))   # row 0 → 1.0 (high), last → 0.0 (low)
    return frac * nyquist


def _x_to_time(x: int, spec_w: int, duration: float) -> float:
    """Convert spectrogram pixel x to seconds."""
    return x / max(spec_w - 1, 1) * duration
