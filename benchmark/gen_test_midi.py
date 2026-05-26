"""Generate a test MIDI file (Twinkle Twinkle) for benchmarking the planner.

Primary path: use ``robopianist.music.library.twinkle_twinkle_rousseau`` to
produce the same Twinkle Twinkle file the RoboPianist debug environment
expects, then save it to ``/tmp/twinkle.mid``.

Fallback path: if RoboPianist's library is not importable (or ``.save``
fails for any reason), build a small Twinkle Twinkle melody around middle C
directly with ``pretty_midi`` and save that instead. The fallback is enough
for the planner's smoke pipeline because the debug environment exposes a
matching MIDI internally; the on-disk file is used only as a placeholder
``--midi-path`` argument.
"""

from __future__ import annotations

import os
import sys
import traceback

OUTPUT_PATH = "/tmp/twinkle.mid"


def _save_via_robopianist(path: str) -> bool:
    """Try the official RoboPianist generator. Returns True on success."""
    try:
        from robopianist.music.library import twinkle_twinkle_rousseau
    except Exception:
        print("[gen_test_midi] robopianist import failed:", file=sys.stderr)
        traceback.print_exc()
        return False

    try:
        song = twinkle_twinkle_rousseau()
        song.save(path)
    except Exception:
        print("[gen_test_midi] twinkle_twinkle_rousseau().save() failed:",
              file=sys.stderr)
        traceback.print_exc()
        return False
    return True


def _save_via_pretty_midi(path: str) -> bool:
    """Fallback: emit a minimal Twinkle melody around middle C."""
    try:
        import pretty_midi
    except Exception:
        print("[gen_test_midi] pretty_midi import failed too:", file=sys.stderr)
        traceback.print_exc()
        return False

    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    # Twinkle Twinkle Little Star, first phrase (12 notes), around middle C (60).
    # Pitches: C C G G A A G, F F E E D D C
    midi_pitches = [60, 60, 67, 67, 69, 69, 67,
                    65, 65, 64, 64, 62, 62, 60]
    durations = [0.5] * 6 + [1.0] + [0.5] * 6 + [1.0]
    start = 0.0
    for pitch, dur in zip(midi_pitches, durations):
        note = pretty_midi.Note(velocity=90, pitch=pitch,
                                start=start, end=start + dur)
        piano.notes.append(note)
        start += dur

    pm.instruments.append(piano)
    pm.write(path)
    return True


def main() -> int:
    abs_path = os.path.abspath(OUTPUT_PATH)
    used_fallback = False
    if not _save_via_robopianist(abs_path):
        print("[gen_test_midi] falling back to pretty_midi", file=sys.stderr)
        used_fallback = True
        if not _save_via_pretty_midi(abs_path):
            print("[gen_test_midi] both generators failed", file=sys.stderr)
            return 1

    try:
        size = os.path.getsize(abs_path)
    except OSError as exc:
        print(f"[gen_test_midi] could not stat {abs_path}: {exc}",
              file=sys.stderr)
        return 1

    print(f"midi_path={abs_path}")
    print(f"midi_size_bytes={size}")
    print(f"used_fallback={used_fallback}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
