"""
onboard.py — AudioCipher skill one-time onboarding flow.

Designed for use inside an OpenClaw agent chat session (Telegram or similar).
State is persisted to ~/.audiocipher/onboard_state.json so the flow runs
exactly once — never re-triggered after completion.

Phase sequence
──────────────
  0  Not started
     → Generate mystery WAV (secret text, not revealed), send cryptic prompt

  1  WAV sent; awaiting operator's discovery
     → Operator visits audiocipher.app → Decode → tells the agent what they found
     → Operator says "skip" → jump straight to complete
     → Any other reply → acknowledge, ask operator to now encode something on the
       site and send the WAV back

  2  Waiting for operator's encoded WAV
     → Operator attaches a WAV (operator_attachment) → agent decodes and reveals
     → Operator says "skip" → complete
     → Operator sends text without a file → nudge to send the WAV

  ✓  Complete — subsequent calls are silent no-ops

"skip" is matched case-insensitively anywhere in operator_input at phases 1 & 2.

Public API
──────────
  is_complete() → bool
      True if onboarding has already finished.

  run_onboard(operator_input=None, operator_attachment=None) → dict
      Advance the state machine.

      operator_input      : str | None  — operator's text reply
      operator_attachment : str | None  — local path of a WAV file the operator
                                          sent (Phase 2 only)

      Return dict:
        {
          'complete':   bool,
          'phase':      int,         # 1–2 active, -1 when done
          'message':    str,         # text to send to the operator
          'attachment': str | None,  # local path of file to SEND (Phase 1 WAV)
        }

  reset_onboard()
      Clear state and remove demo WAV — restarts from Phase 0.
      Use for testing or re-installation.

Usage in an OpenClaw agent
──────────────────────────
  from onboard import is_complete, run_onboard

  # At the top of each turn handler (BEFORE any skill command handling):
  if not is_complete():
      result = run_onboard(
          operator_input=user_message,       # text the operator typed
          operator_attachment=file_path,     # WAV they attached, or None
      )
      await send(result['message'])
      if result['attachment']:
          await send_audio(result['attachment'])
      return   # swallow turn; normal commands resume next turn after complete
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────

_STATE_DIR  = Path.home() / '.audiocipher'
_STATE_FILE = _STATE_DIR / 'onboard_state.json'
_DEMO_WAV   = _STATE_DIR / 'onboard_demo.wav'

_DEMO_TEXT = 'AUDIOCIPHER'   # kept secret from operator until they decode it
_DEMO_MODE = 'hzalpha'


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {'phase': 0, 'complete': False}


def _save_state(state: dict) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _wants_skip(text: str | None) -> bool:
    """Return True if the operator typed 'skip' (case-insensitive, anywhere)."""
    return bool(text and 'skip' in text.lower())


def _complete(state: dict) -> dict:
    """Mark state complete and return the completion message."""
    state['complete'] = True
    state['phase']    = -1
    _save_state(state)
    return {
        'complete':   True,
        'phase':      -1,
        'message':    (
            'AudioCipher setup complete ✓\n\n'
            'Commands:\n'
            '  encode <text>      — encode a message as cipher audio\n'
            '  decode <file>      — extract a message from audio\n'
            '  analyze <file>     — find hidden content in any audio\n'
            '  spectrogram <file> — render a spectrogram PNG\n'
            '  video <file>       — wrap audio as MP4 for Twitter/X\n\n'
            'audiocipher.app — encode & decode in your browser'
        ),
        'attachment': None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def is_complete() -> bool:
    """Return True if onboarding has already been completed."""
    return bool(_load_state().get('complete', False))


def reset_onboard() -> None:
    """
    Clear all onboarding state and remove the demo WAV.
    Re-enables the full flow from Phase 0 — use for testing or re-installation.
    """
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()
    if _DEMO_WAV.exists():
        _DEMO_WAV.unlink()


def run_onboard(
    operator_input:      str | None = None,
    operator_attachment: str | None = None,
) -> dict:
    """
    Advance the onboarding state machine by one step.

    Args:
        operator_input:      Text the operator typed this turn, or None.
        operator_attachment: Local path of a WAV file the operator attached,
                             or None.  Only used in Phase 2.

    Returns:
        {
          'complete':   bool,
          'phase':      int,         # 1–2 active, -1 when done
          'message':    str,         # text to send to the operator
          'attachment': str | None,  # file path to attach when sending, or None
        }
    """
    state = _load_state()

    # ── Already done ──────────────────────────────────────────────────────────
    if state.get('complete'):
        return {'complete': True, 'phase': -1, 'message': '', 'attachment': None}

    phase = state.get('phase', 0)

    # ── Phase 0 → 1: First invocation — generate mystery WAV ─────────────────
    if phase == 0:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)

        from cipher import write_cipher_wav  # type: ignore
        write_cipher_wav(str(_DEMO_WAV), _DEMO_TEXT, mode=_DEMO_MODE)

        state['phase']    = 1
        state['demo_wav'] = str(_DEMO_WAV)
        _save_state(state)

        return {
            'complete':   False,
            'phase':      1,
            'message':    (
                'i\'ve hidden something in this file.\n\n'
                'go to audiocipher.app → Decode → upload it.\n'
                'tell me what you find.'
            ),
            'attachment': str(_DEMO_WAV),
        }

    # ── Phase 1: awaiting operator's discovery ────────────────────────────────
    if phase == 1:
        if operator_input is None:
            # Re-entrant: operator hasn't replied yet — re-prompt without re-attaching
            return {
                'complete':   False,
                'phase':      1,
                'message':    (
                    'upload the WAV to audiocipher.app → Decode\n'
                    'tell me what you find. (or say "skip" to skip)'
                ),
                'attachment': None,
            }

        if _wants_skip(operator_input):
            return _complete(state)

        # Operator reported what they decoded — acknowledge and issue Phase 2 challenge
        state['phase']            = 2
        state['operator_decoded'] = operator_input.strip()
        _save_state(state)

        return {
            'complete':   False,
            'phase':      2,
            'message':    (
                f'"{operator_input.strip()}" — correct.\n\n'
                'now go the other way.\n'
                'audiocipher.app → Encode — type anything, download the WAV,\n'
                'and send it to me. i\'ll tell you what you said.\n'
                '(or say "skip")'
            ),
            'attachment': None,
        }

    # ── Phase 2: waiting for operator's encoded WAV ───────────────────────────
    if phase == 2:
        # Skip check first — text takes priority over attachment
        if _wants_skip(operator_input):
            return _complete(state)

        if operator_attachment:
            # Operator sent a WAV — decode it and reveal
            from cipher import decode  # type: ignore
            try:
                decoded = decode(operator_attachment, mode='auto').strip()
            except Exception as exc:
                decoded = f'[could not decode: {exc}]'

            state['operator_encoded'] = decoded
            result = _complete(state)
            # Override message to include the reveal before the capability summary
            reveal = (
                f'you said: "{decoded}"\n\n'
                '── ── ──\n\n'
            )
            result['message'] = reveal + result['message']
            return result

        if operator_input is not None:
            # Text reply but no WAV attached — nudge
            return {
                'complete':   False,
                'phase':      2,
                'message':    (
                    'send me the WAV file — audiocipher.app → Encode → download\n'
                    'attach it here. (or say "skip")'
                ),
                'attachment': None,
            }

        # Re-entrant: no input at all this turn
        return {
            'complete':   False,
            'phase':      2,
            'message':    (
                'waiting for your encoded WAV.\n'
                'audiocipher.app → Encode → download → attach here.\n'
                '(or say "skip")'
            ),
            'attachment': None,
        }

    # Safety fallback — should never reach here
    return {'complete': False, 'phase': phase, 'message': '', 'attachment': None}
