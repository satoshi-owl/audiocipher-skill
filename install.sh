#!/usr/bin/env bash
# install.sh — Install all AudioCipher skill dependencies
# Usage: bash install.sh
set -e

echo "── AudioCipher Skill Installer ──────────────────────────────"

# ── Python packages ───────────────────────────────────────────────────────────
echo "→ Installing Python dependencies..."
pip3 install -r requirements.txt
echo "  Python packages OK"

# ── ffmpeg ────────────────────────────────────────────────────────────────────
echo "→ Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "  ffmpeg OK  ($(ffmpeg -version 2>&1 | head -1))"
else
    echo "  ffmpeg not found — installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y ffmpeg
    else
        echo "  ⚠ Auto-install not supported on $OSTYPE. Install from https://ffmpeg.org"
    fi
fi

# ── Tesseract (OCR engine) ────────────────────────────────────────────────────
echo "→ Checking tesseract (OCR)..."
if command -v tesseract &> /dev/null; then
    echo "  tesseract OK  ($(tesseract --version 2>&1 | head -1))"
else
    echo "  tesseract not found — installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install tesseract
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y tesseract-ocr
    else
        echo "  ⚠ Auto-install not supported on $OSTYPE. Install tesseract manually."
    fi
fi

# ── zbar (QR / barcode library required by pyzbar) ───────────────────────────
echo "→ Checking zbar..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    if brew list zbar &>/dev/null; then
        echo "  zbar OK"
    else
        echo "  Installing zbar via Homebrew..."
        brew install zbar
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if dpkg -l libzbar0 &>/dev/null 2>&1; then
        echo "  libzbar0 OK"
    else
        echo "  Installing libzbar0..."
        sudo apt-get install -y libzbar0
    fi
fi

echo ""
echo "✓ All done!"
echo ""
echo "Usage:"
echo "  python3 audiocipher.py encode 'HELLO' --output cipher.wav"
echo "  python3 audiocipher.py decode cipher.wav"
echo "  python3 audiocipher.py image2audio logo.png --output hidden.wav"
echo "  python3 audiocipher.py analyze mystery.wav --output-dir ./findings/"
echo "  python3 audiocipher.py video cipher.wav --output cipher.mp4"
echo ""
echo "Run 'python3 audiocipher.py --help' for full options."
