#!/usr/bin/env python3
"""
audiocipher.py — AudioCipher CLI entry point.

Commands:
  onboard               Run first-use onboarding flow (outputs JSON)
  encode      <text>    Generate cipher video MP4 (default) or WAV (--audio-only)
  decode      <audio>   Decode cipher audio/video to text
  image2audio <image>   Convert image to audio (spectrogram technique)
  analyze     <audio>   Find hidden content in audio spectrogram
  spectrogram <audio>   Render a spectrogram PNG from audio
  video       <audio>   Generate waveform MP4 for Twitter/X

Run `python3 audiocipher.py --help` for full usage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

_VERSION_FILE = Path(__file__).parent / 'VERSION'
_GITHUB_VERSION_URL = (
    'https://raw.githubusercontent.com/satoshi-owl/audiocipher-skill/main/VERSION'
)
_update_available: list[str] = []   # populated by background thread


def _bg_version_check() -> None:
    """Fetch remote VERSION in background; append to _update_available if newer."""
    try:
        import urllib.request
        local = _VERSION_FILE.read_text().strip() if _VERSION_FILE.exists() else '0.0.0'
        with urllib.request.urlopen(_GITHUB_VERSION_URL, timeout=2) as r:
            remote = r.read().decode().strip()
        if remote != local:
            _update_available.append(remote)
    except Exception:
        pass

# Add skill/ directory to path so local imports work when called from anywhere
sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_update(args: argparse.Namespace):
    import subprocess as _sp
    skill_dir = str(Path(__file__).parent)
    print('→ Pulling latest from origin/main…')
    r = _sp.run(['git', '-C', skill_dir, 'pull', 'origin', 'main'],
                capture_output=True, text=True)
    if r.returncode != 0:
        print(f'✗ git pull failed:\n{r.stderr.strip()}', file=sys.stderr)
        sys.exit(1)
    print(r.stdout.strip() or '  Already up to date.')
    print('→ Installing/updating dependencies…')
    _sp.run(['pip3', 'install', '-r',
             str(Path(skill_dir) / 'requirements.txt'), '-q'], check=True)
    version = _VERSION_FILE.read_text().strip() if _VERSION_FILE.exists() else '?'
    print(f'✓ AudioCipher skill updated to v{version}')


def cmd_onboard(args: argparse.Namespace):
    from onboard import run_onboard  # type: ignore

    result = run_onboard(
        operator_input=args.input,
        operator_attachment=args.attachment,
    )
    print(json.dumps(result, indent=2))


def cmd_encode(args: argparse.Namespace):
    # ── ABP path ──────────────────────────────────────────────────────────────
    if args.mode == 'abp':
        from abp_bridge import encode_abp, ABP_SR  # type: ignore
        import scipy.io.wavfile as _wf

        out_path = args.output
        # Default output extension: .wav (ABP is always audio-only for now)
        if not args.audio_only and not out_path.endswith('.wav'):
            # Strip any .mp4 suffix the user may have passed
            out_path = Path(out_path).with_suffix('.wav').as_posix()

        passphrase  = getattr(args, 'passphrase', None)
        abp_profile = getattr(args, 'abp_profile', 'social_safe') or 'social_safe'

        print(f'→ ABP encode  profile={abp_profile}  '
              f'encrypted={"yes" if passphrase else "no"}', file=sys.stderr)

        samples = encode_abp(args.text, passphrase=passphrase, profile=abp_profile)
        import numpy as _np
        pcm = _np.clip(samples, -1.0, 1.0)
        _wf.write(out_path, ABP_SR, (pcm * 32767).astype(_np.int16))
        dur     = len(samples) / ABP_SR
        size_kb = os.path.getsize(out_path) / 1024
        print(f'✓ Saved: {out_path}  ({dur:.2f}s  {size_kb:.1f} KB, mode=abp)')
        return

    # ── Legacy cipher modes ───────────────────────────────────────────────────
    from cipher import write_cipher_wav  # type: ignore

    kwargs = {}
    if args.duration_ms  is not None: kwargs['duration_ms']   = args.duration_ms
    if args.letter_gap   is not None: kwargs['letter_gap_ms'] = args.letter_gap
    if args.word_gap     is not None: kwargs['word_gap_ms']   = args.word_gap
    if args.fade         is not None: kwargs['fade_ms']       = args.fade
    if args.volume       is not None: kwargs['volume']        = args.volume
    if args.morse_freq   is not None: kwargs['morse_freq']    = args.morse_freq
    if args.morse_wpm    is not None: kwargs['morse_wpm']     = args.morse_wpm
    if args.fsk_baud     is not None: kwargs['fsk_baud']      = args.fsk_baud
    kwargs['waveform'] = args.waveform

    # If the output path ends in .wav, force audio-only regardless of the flag.
    # ffmpeg silently accepts a .wav output path, drops the video stream, and
    # re-encodes the audio as AAC — destroying cipher fidelity.  Catch it here
    # so the user always gets a lossless PCM WAV when they name the file .wav.
    if not args.audio_only and args.output.lower().endswith('.wav'):
        print(
            '⚠  Output is .wav — switching to --audio-only (video containers '
            'cannot be .wav; use .mp4 for video output).',
            file=sys.stderr,
        )
        args.audio_only = True

    if args.audio_only:
        # WAV-only mode: write audio directly to output
        write_cipher_wav(args.output, args.text, mode=args.mode, **kwargs)
        size_kb = os.path.getsize(args.output) / 1024
        print(f'✓ Saved: {args.output}  ({size_kb:.1f} KB, mode={args.mode})')
    else:
        # Video-first (default): encode → temp WAV → branded MP4 → delete WAV
        from video import check_or_install_ffmpeg, generate_video_safe  # type: ignore

        if not check_or_install_ffmpeg():
            print(
                '✗ ffmpeg required for video output. Use --audio-only for WAV.',
                file=sys.stderr,
            )
            sys.exit(1)

        fd, tmp_wav = tempfile.mkstemp(suffix='_ac.wav')
        os.close(fd)
        try:
            write_cipher_wav(tmp_wav, args.text, mode=args.mode, **kwargs)
            result = generate_video_safe(
                tmp_wav,
                output_path=args.output,
                twitter=(not args.lossless),   # default: AAC 320k (browser-compatible)
            )
            if result is None:
                # Video generation failed — fall back to saving the WAV
                wav_path = str(Path(args.output).with_suffix('.wav'))
                import shutil as _shutil
                _shutil.copy2(tmp_wav, wav_path)
                size_kb = os.path.getsize(wav_path) / 1024
                print(
                    f'⚠ Video failed; WAV saved: {wav_path}  ({size_kb:.1f} KB, mode={args.mode})',
                    file=sys.stderr,
                )
                print(f'✓ Saved: {wav_path}  ({size_kb:.1f} KB, mode={args.mode})')
            else:
                size_mb = os.path.getsize(args.output) / (1024 * 1024)
                print(f'✓ Saved: {args.output}  ({size_mb:.1f} MB, mode={args.mode})')
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)


def cmd_decode(args: argparse.Namespace):
    # ── ABP path ──────────────────────────────────────────────────────────────
    if args.mode == 'abp':
        from abp_bridge import decode_abp  # type: ignore

        passphrase = getattr(args, 'passphrase', None)
        print(f'→ ABP decode  encrypted={"yes" if passphrase else "no"}',
              file=sys.stderr)
        try:
            text = decode_abp(args.audio, passphrase=passphrase)
            print(text)
        except RuntimeError as exc:
            print(f'✗ {exc}', file=sys.stderr)
            sys.exit(1)
        return

    # ── Legacy cipher modes ───────────────────────────────────────────────────
    from cipher import decode  # type: ignore
    import shutil
    import subprocess

    audio_path = args.audio
    tmp_wav    = None

    # Auto-extract audio from video/container files (MP4, MOV, MKV, M4A…)
    _VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.m4a', '.m4v', '.webm', '.avi')
    if Path(audio_path).suffix.lower() in _VIDEO_EXTS:
        if not shutil.which('ffmpeg'):
            print('✗ ffmpeg required to decode from video files.', file=sys.stderr)
            sys.exit(1)

        # Recover cipher mode from the MP4 comment tag written by `audiocipher video`.
        # This survives AAC re-encoding because it lives in the container, not the audio.
        if args.mode == 'auto' and shutil.which('ffprobe'):
            try:
                _probe = subprocess.run(
                    ['ffprobe', '-v', 'quiet',
                     '-show_entries', 'format_tags=comment',
                     '-of', 'default=noprint_wrappers=1:nokey=1',
                     audio_path],
                    capture_output=True, text=True,
                )
                _comment = _probe.stdout.strip()
                if _comment.startswith('{'):
                    _tag = json.loads(_comment)
                    _meta = _tag.get('audiocipher', {})
                    if _meta.get('mode'):
                        args.mode = _meta['mode']
                        print(f'→ Mode from MP4 tag: {args.mode}', file=sys.stderr)
            except Exception:
                pass

        fd, tmp_wav = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        subprocess.run(
            ['ffmpeg', '-i', audio_path, '-vn', '-acodec', 'pcm_s16le', tmp_wav, '-y'],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        audio_path = tmp_wav

    kwargs: dict = {}
    if args.morse_wpm  is not None: kwargs['morse_wpm']  = args.morse_wpm
    if args.fsk_baud   is not None: kwargs['fsk_baud']   = args.fsk_baud
    if args.letter_gap is not None: kwargs['letter_gap_ms'] = args.letter_gap
    if args.word_gap   is not None: kwargs['word_gap_ms']   = args.word_gap

    try:
        result = decode(audio_path, mode=args.mode, **kwargs)
        print(result)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def cmd_image2audio(args: argparse.Namespace):
    from spectrogram import image_to_audio  # type: ignore
    from utils import write_wav  # type: ignore

    audio = image_to_audio(
        args.image,
        fmin=args.fmin,
        fmax=args.fmax,
        duration=args.duration,
        sr=args.sr,
        amplitude=args.amplitude,
        invert=args.invert,
    )
    write_wav(args.output, audio, sr=args.sr, mode=None)
    size_kb = os.path.getsize(args.output) / 1024
    print(
        f'✓ Saved: {args.output}  ({size_kb:.1f} KB, '
        f'{args.duration}s, {args.fmin:.0f}–{args.fmax:.0f} Hz)'
    )


def cmd_analyze(args: argparse.Namespace):
    from analyzer import analyze  # type: ignore

    findings = analyze(
        args.audio,
        output_dir=args.output_dir,
        detect_qr=args.qr,
        detect_ocr=args.ocr,
        detect_cipher=args.cipher,
        detect_anomaly=args.anomaly,
        crop=args.crop,
    )

    results_path = os.path.join(args.output_dir, 'results.json')
    print(json.dumps(findings, indent=2))
    print(f'\n→ {len(findings)} finding(s). Full results: {results_path}',
          file=sys.stderr)


def cmd_video(args: argparse.Namespace):
    from video import check_or_install_ffmpeg, generate_video  # type: ignore

    if not check_or_install_ffmpeg():
        print('✗ ffmpeg is required but could not be installed.', file=sys.stderr)
        sys.exit(1)

    audio_path = args.audio
    tmp_wav    = None

    # ── Auto-transcode codec-unsafe cipher modes → HZAlpha ────────────────────
    # FSK and Morse use narrow-spaced bins (≤200 Hz) that don't survive AAC or
    # Opus compression.  HZAlpha's chromatic tones (13+ Hz gaps) survive AAC
    # 128k, Opus 96k, and Telegram/Twitter re-encoding chains.
    #
    # WaveSig (ggwave) is NOT in this list since v0.2.3 — it was fixed to use
    # 100 Hz bin spacing (up from 46.875 Hz) which comfortably survives AAC.
    _UNSAFE = ('fsk', 'morse')
    _unsafe_detected = False
    try:
        with open(audio_path, 'rb') as _f:
            _md = _f.read(2048)
        _unsafe_detected = any(
            f'"mode": "{m}"'.encode() in _md or f'"mode":"{m}"'.encode() in _md
            for m in _UNSAFE
        )
    except Exception:
        pass

    if _unsafe_detected:
        print(
            '→ Codec-unsafe cipher mode detected (FSK / Morse).\n'
            '  Auto-transcoding to HZAlpha — survives Telegram, Twitter, and '
            'any AAC/Opus re-encode…',
            file=sys.stderr,
        )
        try:
            from cipher import decode, write_cipher_wav  # type: ignore
            # Detect original mode name from metadata
            _orig_mode = 'auto'
            for m in _UNSAFE:
                if (f'"mode": "{m}"'.encode() in _md or
                        f'"mode":"{m}"'.encode() in _md):
                    _orig_mode = m
                    break
            message = decode(audio_path, mode=_orig_mode)
            if not message:
                message = decode(audio_path, mode='auto')
            if message:
                # Verify: write hzalpha and read it back — if they match, transcode is valid
                fd, tmp_wav = tempfile.mkstemp(suffix='_hzalpha.wav')
                os.close(fd)
                write_cipher_wav(tmp_wav, message, mode='hzalpha')
                _verify = decode(tmp_wav, mode='hzalpha')
                if _verify == message:
                    audio_path = tmp_wav
                    print(f'  ✓ Transcoded → HZAlpha: "{message}"', file=sys.stderr)
                else:
                    os.unlink(tmp_wav)
                    tmp_wav = None
                    print(
                        '  ⚠ Transcode verification failed (decoder round-trip mismatch).\n'
                        '    Re-encode the message with --mode hzalpha for a survivable video:\n'
                        '      python3 audiocipher.py encode "your message" --mode hzalpha',
                        file=sys.stderr,
                    )
            else:
                print(
                    '  ⚠ Could not decode original cipher.\n'
                    '    Re-encode the message with --mode hzalpha for a survivable video:\n'
                    '      python3 audiocipher.py encode "your message" --mode hzalpha',
                    file=sys.stderr,
                )
        except Exception as _e:
            print(f'  ⚠ Transcode failed ({_e}) — using original audio.',
                  file=sys.stderr)

    try:
        out = generate_video(
            audio_path,
            output_path=args.output,
            style=args.style,
            resolution=args.resolution,
            title=args.title,
            twitter=args.twitter,
            window_seconds=args.window_seconds,
            verbose=args.verbose,
        )
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f'✓ Saved: {out}  ({size_mb:.1f} MB)')


def cmd_spectrogram(args: argparse.Namespace):
    from spectrogram import save_spectrogram  # type: ignore

    out = save_spectrogram(
        args.audio,
        output_path=args.output,
        fft_size=args.fft_size,
        fmin=args.fmin,
        fmax=args.fmax if args.fmax is not None else None,
        colormap=args.colormap,
        width=args.width,
        height=args.height,
        gain_db=args.gain_db,
        labeled=args.labeled,
    )
    size_kb = os.path.getsize(out) / 1024
    print(f'✓ Saved: {out}  ({size_kb:.1f} KB)')


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='audiocipher',
        description='AudioCipher — encode/decode/analyze hidden audio messages.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 audiocipher.py onboard
  python3 audiocipher.py onboard --input "I found it, it says AUDIOCIPHER"
  python3 audiocipher.py onboard --input "here's my wav" --attachment operator.wav
  python3 audiocipher.py encode "HELLO WORLD"                     # → cipher.mp4 (video, default)
  python3 audiocipher.py encode "HELLO WORLD" --audio-only        # → cipher.wav (audio only)
  python3 audiocipher.py encode "Long text..." --mode achd         # → HyperDense video
  python3 audiocipher.py encode "Secret" --mode abp --passphrase "hunter2"  # → ABP cipher.wav
  python3 audiocipher.py decode cipher.mp4                         # auto-detects mode from MP4
  python3 audiocipher.py decode cipher.wav --mode auto
  python3 audiocipher.py decode received.ogg --mode abp            # ABP after Telegram re-encode
  python3 audiocipher.py decode received.ogg --mode abp --passphrase "hunter2"
  python3 audiocipher.py image2audio logo.png --output hidden.wav --fmin 1000 --fmax 8000
  python3 audiocipher.py analyze mystery.wav --output-dir ./findings/
  python3 audiocipher.py spectrogram mystery.wav --output spec.png --colormap green --labeled
  python3 audiocipher.py video cipher.wav --output cipher.mp4 --title "NULL"
""",
    )
    sub = p.add_subparsers(dest='command', required=True)

    # ── update ────────────────────────────────────────────────────────────────
    upd = sub.add_parser(
        'update',
        help='Pull latest skill version from GitHub.',
        description=(
            'Runs git pull origin main + pip3 install -r requirements.txt '
            'to update the skill to the latest version.'
        ),
    )
    upd.set_defaults(func=cmd_update)

    # ── onboard ───────────────────────────────────────────────────────────────
    onb = sub.add_parser(
        'onboard',
        help='Run first-use onboarding flow (outputs JSON).',
        description=(
            'Drives the one-time AudioCipher onboarding flow. '
            'Always outputs JSON: {"complete": bool, "phase": int, "message": str, "attachment": str|null}. '
            'Call with no args to get the current phase prompt. '
            'Pass --input with operator reply to advance the flow. '
            'Pass --attachment with a WAV path when operator sends audio.'
        ),
    )
    onb.add_argument('--input', default=None, metavar='TEXT',
                     help='Operator reply text (omit to re-prompt current phase)')
    onb.add_argument('--attachment', default=None, metavar='WAV',
                     help='Path to WAV file attached by operator (Phase 2)')
    onb.set_defaults(func=cmd_onboard)

    # ── encode ────────────────────────────────────────────────────────────────
    enc = sub.add_parser(
        'encode',
        help='Encode text as branded cipher video MP4 (default) or WAV (--audio-only).',
        description=(
            'Encode text to audio and wrap it in an AudioCipher-branded waveform video. '
            'Mode-specific params are embedded in the output for auto-detection on decode. '
            'Use --audio-only to output a plain WAV file instead of video.'
        ),
    )
    enc.add_argument('text', help='Text to encode')
    enc.add_argument('--mode', default='hzalpha',
                     choices=['hzalpha', 'morse', 'dtmf', 'fsk', 'ggwave', 'acdense', 'achd', 'abp', 'custom'],
                     help='Cipher mode (default: hzalpha). '
                          'acdense = dense Unicode+emoji. achd = HyperDense long-text. '
                          'abp = ABP v1 OFDM (survives Telegram/X OGG Opus re-encode).')
    enc.add_argument('--output', '-o', default='cipher.mp4',
                     help='Output path (default: cipher.mp4). '
                          'Use --audio-only to write cipher.wav instead.')
    enc.add_argument('--audio-only', action='store_true',
                     help='Output WAV audio only — skip video generation.')
    enc.add_argument('--lossless', action='store_true',
                     help='Use ALAC lossless audio in video (default: AAC 320k). '
                          'ALAC files may not decode in Firefox/Linux on the website.')
    enc.add_argument('--duration-ms', type=float, default=None,
                     metavar='MS', help='Tone duration in ms (default: 120)')
    enc.add_argument('--letter-gap', type=float, default=None,
                     metavar='MS', help='Inter-letter gap ms (default: 20)')
    enc.add_argument('--word-gap', type=float, default=None,
                     metavar='MS', help='Inter-word gap ms (default: 350)')
    enc.add_argument('--fade', type=float, default=None,
                     metavar='MS', help='Fade-in/out ramp ms (default: 10)')
    enc.add_argument('--volume', type=float, default=None,
                     metavar='0-1', help='Output volume 0–1 (default: 0.8)')
    enc.add_argument('--waveform', default='sine',
                     choices=['sine', 'square', 'sawtooth', 'triangle'],
                     help='Oscillator waveform (default: sine)')
    enc.add_argument('--morse-freq', type=float, default=None,
                     metavar='HZ', help='Morse tone frequency Hz (default: 700)')
    enc.add_argument('--morse-wpm', type=float, default=None,
                     metavar='WPM', help='Morse words per minute (default: 20)')
    enc.add_argument('--fsk-baud', type=float, default=None,
                     metavar='BAUD', help='FSK baud rate (default: 45)')
    enc.add_argument('--passphrase', '-p', default=None,
                     metavar='PHRASE',
                     help='[ABP only] Encrypt with this passphrase (XChaCha20 + Argon2id).')
    enc.add_argument('--abp-profile', default='social_safe',
                     choices=['social_safe', 'fast', 'browser_safe'],
                     help='[ABP only] Codec profile: social_safe (default, zstd+FEC, survives Telegram/X), '
                          'fast (lighter FEC), or browser_safe (no zstd — decodable in the web app).')
    enc.set_defaults(func=cmd_encode)

    # ── decode ────────────────────────────────────────────────────────────────
    dec = sub.add_parser(
        'decode',
        help='Decode cipher audio to text.',
        description=(
            'Decode an AudioCipher WAV. Use --mode auto (default) to '
            'read embedded metadata and restore settings automatically.'
        ),
    )
    dec.add_argument('audio', help='Input WAV file')
    dec.add_argument('--mode', default='auto',
                     choices=['auto', 'hzalpha', 'morse', 'dtmf', 'fsk', 'ggwave', 'acdense', 'achd', 'abp', 'custom'],
                     help='Decode mode. "auto" reads embedded metadata or probes signal (default: auto). '
                          'Use "abp" for ABP v1 OFDM files (WAV / OGG / M4A / MP3).')
    dec.add_argument('--morse-wpm', type=float, default=None,
                     metavar='WPM', help='Morse WPM override (default: from metadata or 20)')
    dec.add_argument('--fsk-baud', type=float, default=None,
                     metavar='BAUD', help='FSK baud rate override')
    dec.add_argument('--letter-gap', type=float, default=None, metavar='MS')
    dec.add_argument('--word-gap', type=float, default=None, metavar='MS')
    dec.add_argument('--passphrase', '-p', default=None,
                     metavar='PHRASE',
                     help='[ABP only] Decryption passphrase (if message was encrypted).')
    dec.set_defaults(func=cmd_decode)

    # ── image2audio ───────────────────────────────────────────────────────────
    i2a = sub.add_parser(
        'image2audio',
        help='Convert image to audio (appears in spectrogram).',
        description=(
            'Map pixel brightness to oscillator amplitudes so the image '
            'appears visually in the audio spectrogram (Aphex Twin technique).'
        ),
    )
    i2a.add_argument('image', help='Input image (PNG / JPG / any Pillow format)')
    i2a.add_argument('--output', '-o', default='hidden.wav',
                     help='Output WAV path (default: hidden.wav)')
    i2a.add_argument('--fmin', type=float, default=200.0,
                     metavar='HZ', help='Min frequency Hz (default: 200)')
    i2a.add_argument('--fmax', type=float, default=16000.0,
                     metavar='HZ', help='Max frequency Hz (default: 16000)')
    i2a.add_argument('--duration', type=float, default=6.0,
                     metavar='S', help='Audio duration seconds (default: 6)')
    i2a.add_argument('--sr', type=int, default=44100,
                     metavar='HZ', help='Sample rate (default: 44100)')
    i2a.add_argument('--amplitude', type=float, default=0.8,
                     metavar='0-1', help='Output amplitude 0–1 (default: 0.8)')
    i2a.add_argument('--invert', action='store_true',
                     help='Invert brightness (white pixels → silence)')
    i2a.set_defaults(func=cmd_image2audio)

    # ── analyze ───────────────────────────────────────────────────────────────
    ana = sub.add_parser(
        'analyze',
        help='Find hidden content in audio spectrogram.',
        description=(
            'Runs a full detection pipeline: '
            'QR codes, OCR text, contour images, cipher tones, anomalies. '
            'Outputs findings/results.json + cropped PNGs.'
        ),
    )
    ana.add_argument('audio', help='Input audio file')
    ana.add_argument('--output-dir', '-d', default='./findings/',
                     metavar='DIR', help='Output directory (default: ./findings/)')
    ana.add_argument('--no-qr',      dest='qr',      action='store_false',
                     help='Skip QR code detection')
    ana.add_argument('--no-ocr',     dest='ocr',     action='store_false',
                     help='Skip OCR text extraction')
    ana.add_argument('--no-cipher',  dest='cipher',  action='store_false',
                     help='Skip cipher tone auto-detection')
    ana.add_argument('--no-anomaly', dest='anomaly', action='store_false',
                     help='Skip anomaly / entropy detection')
    ana.add_argument('--no-crop',    dest='crop',    action='store_false',
                     help='Do not save cropped PNG per finding')
    ana.set_defaults(func=cmd_analyze, qr=True, ocr=True,
                     cipher=True, anomaly=True, crop=True)

    # ── video ─────────────────────────────────────────────────────────────────
    vid = sub.add_parser(
        'video',
        help='Generate waveform MP4 for Twitter/X posting.',
        description=(
            'Wraps audio in a 1280×720 MP4 with a NULL-style waveform '
            '(black bg, crimson waveform, scanlines, grain). '
            'Agents cannot post raw audio to X — use this first.'
        ),
    )
    vid.add_argument('audio', help='Input audio file')
    vid.add_argument('--output', '-o', default='out.mp4',
                     help='Output MP4 path (default: out.mp4)')
    vid.add_argument('--style', default='null',
                     choices=['null'],
                     help='Visual style: null = retro terminal (default: null)')
    vid.add_argument('--resolution', default='1280x720',
                     metavar='WxH', help='Output resolution (default: 1280x720)')
    vid.add_argument('--title', default=None,
                     metavar='TEXT', help='Optional title overlay text')
    vid.add_argument('--twitter', action='store_true',
                     help='Encode audio as AAC 320k for Twitter/X posting '
                          '(lossy — HZAlpha survives; WaveSig/FSK/Morse may not). '
                          'Default: ALAC lossless — MP4 stays fully decodable.')
    vid.add_argument('--window-seconds', type=float, default=4.0,
                     metavar='S',
                     help='Seconds of audio visible in the scrolling window (default: 4). '
                          'Decrease for a more zoomed-in, animated look.')
    vid.add_argument('--verbose', action='store_true',
                     help='Show ffmpeg output')
    vid.set_defaults(func=cmd_video)

    # ── spectrogram ───────────────────────────────────────────────────────────
    spec = sub.add_parser(
        'spectrogram',
        help='Render a spectrogram PNG from audio.',
        description=(
            'Generate a high-resolution spectrogram image from an audio file. '
            'Useful for visually inspecting audio before running analyze, '
            'or for verifying image2audio output.'
        ),
    )
    spec.add_argument('audio', help='Input audio file')
    spec.add_argument('--output', '-o', default='spectrogram.png',
                      help='Output PNG path (default: spectrogram.png)')
    spec.add_argument('--colormap', default='green',
                      choices=['green', 'inferno', 'viridis', 'amber', 'grayscale'],
                      help='Colour scheme (default: green)')
    spec.add_argument('--fmin', type=float, default=0.0,
                      metavar='HZ', help='Min display frequency Hz (default: 0)')
    spec.add_argument('--fmax', type=float, default=None,
                      metavar='HZ', help='Max display frequency Hz (default: Nyquist)')
    spec.add_argument('--width', type=int, default=1200,
                      metavar='PX', help='Image width in pixels (default: 1200)')
    spec.add_argument('--height', type=int, default=600,
                      metavar='PX', help='Image height in pixels (default: 600)')
    spec.add_argument('--fft-size', type=int, default=4096,
                      metavar='N', help='FFT window size (default: 4096)')
    spec.add_argument('--gain-db', type=float, default=0.0,
                      metavar='DB', help='Brightness gain in dB (default: 0)')
    spec.add_argument('--labeled', action='store_true',
                      help='Add frequency and time axis labels to the image')
    spec.set_defaults(func=cmd_spectrogram)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Background version check — skip for update/onboard to avoid noise
    _ver_thread = None
    if getattr(args, 'command', None) not in ('update', 'onboard'):
        _ver_thread = threading.Thread(target=_bg_version_check, daemon=True)
        _ver_thread.start()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print('\n⚠ Interrupted.', file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f'✗ Error: {e}', file=sys.stderr)
        if os.environ.get('AUDIOCIPHER_DEBUG'):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Wait briefly for version check, then print notification if update available
    if _ver_thread is not None:
        _ver_thread.join(timeout=2.5)
    if _update_available:
        print(
            f'\nℹ  Update available (v{_update_available[0]}) '
            '→ python3 audiocipher.py update',
            file=sys.stderr,
        )


if __name__ == '__main__':
    main()
