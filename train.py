"""
VOICE TRAINING TOOL — Smart Home Assistant
===========================================
Records your real voice, transcribes with the same Whisper model
used at runtime, and saves the EXACT transcription as a learned alias
or trigger word so the assistant recognises you perfectly.

WHY training matters:
  Whisper hears your Darija voice and outputs, for example, "ديبي افتح"
  instead of "تيفي افتح". If "ديبي" is not in the trigger list, nothing
  works. This tool records what Whisper actually produces and saves it.

Usage:
    python train.py
    python train.py --model small
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

from tv_voice_assistant_enhanced import (
    TV_COMMANDS, AC_COMMANDS, calibrate_mic,
)

LEARNED_FILE = Path(__file__).parent / "learned_aliases.json"

SEP = "─" * 65


# ─────────────────────────────────────────────
#  JSON HELPERS
# ─────────────────────────────────────────────

def load_learned() -> dict:
    if LEARNED_FILE.exists():
        try:
            return json.loads(LEARNED_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"TV": {}, "AC": {}, "_triggers": {"TV": [], "AC": []}}


def save_learned(data: dict):
    LEARNED_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _add_trigger(device: str, word: str, learned: dict) -> bool:
    """Return True if the word was newly added."""
    trigs = learned.setdefault("_triggers", {}).setdefault(device, [])
    if word.strip().lower() not in [t.lower() for t in trigs]:
        trigs.append(word.strip())
        return True
    return False


def _add_alias(device: str, cmd_key: str, alias: str, learned: dict) -> bool:
    """Return True if the alias was newly added."""
    bucket = learned.setdefault(device, {}).setdefault(cmd_key, [])
    if alias.strip().lower() not in [a.lower() for a in bucket]:
        bucket.append(alias.strip())
        return True
    return False


# ─────────────────────────────────────────────
#  RECORD ONE UTTERANCE
# ─────────────────────────────────────────────

def record_one(recognizer: sr.Recognizer, whisper: WhisperModel) -> str | None:
    with sr.Microphone() as source:
        print("  >> Speak now (6 s max) ...")
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            print("  [!] Timeout — nothing heard.\n")
            return None
    print("  Transcribing...")
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    segs, _ = whisper.transcribe(
        samples, language="ar", beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    text = " ".join(s.text for s in segs).strip()
    return text if text else None


# ─────────────────────────────────────────────
#  1. TRIGGER WORD TRAINING  ← START HERE
# ─────────────────────────────────────────────

def train_trigger_words(recognizer: sr.Recognizer, whisper: WhisperModel):
    """
    Record how Whisper hears YOUR voice saying 'تيفي' or 'كليم'.
    Every accepted sample is saved as a learned trigger word.
    """
    print(f"\n{'='*65}")
    print("  STEP 1 — TRAIN TRIGGER WORDS  (do this first!)")
    print(f"{'='*65}")
    print()
    print("  The system needs to learn how YOUR voice says the trigger word.")
    print("  Say 'تيفي' or 'TV' several times → system saves what it hears.")
    print("  Next run: 'ديبي افتح' will be recognised as TV + open.")
    print()

    while True:
        print(f"  {SEP}")
        print("  Which trigger to train?")
        print("  1. TV trigger  (تيفي / TV / تلفزيون)")
        print("  2. AC trigger  (كليم / clim / مكيف)")
        print("  3. Show all learned triggers")
        print("  0. Back")
        c = input("\n  Choice: ").strip()

        if c == "0":
            break

        if c not in ("1", "2"):
            if c == "3":
                _print_triggers()
            continue

        device = "TV" if c == "1" else "AC"
        hint = "تيفي / TV / تلفزيون" if device == "TV" else "كليم / clim / مكيف"
        print(f"\n  Training {device} trigger.")
        print(f"  Say your trigger word ({hint}) — 5 samples.")
        print(f"  Tip: say ONLY the trigger word, nothing after it.\n")

        learned = load_learned()
        saved = 0

        for i in range(5):
            print(f"  Sample {i+1}/5 — say just the trigger word:")
            text = record_one(recognizer, whisper)
            if not text:
                continue

            # Keep only first word (in case Whisper adds noise)
            first = text.strip().split()[0]
            full  = text.strip()

            print(f'  Whisper heard : "{full}"')
            if first != full:
                print(f'  First word    : "{first}"')
            print()

            print("  What to save?")
            print(f'  [1] First word only : "{first}"')
            print(f'  [2] Full text       : "{full}"')
            print( "  [n] Skip this sample")
            ans = input("  Choice [1/2/n]: ").strip().lower()

            if ans == "2":
                save_val = full
            elif ans == "n":
                print("  Skipped.\n")
                continue
            else:
                save_val = first

            learned = load_learned()
            if _add_trigger(device, save_val, learned):
                save_learned(learned)
                print(f'  Saved "{save_val}" as {device} trigger.\n')
                saved += 1
            else:
                print(f'  "{save_val}" already saved.\n')

        print(f"  Done. {saved} new trigger word(s) added for {device}.")
        _print_triggers()


def _print_triggers():
    learned = load_learned()
    trigs = learned.get("_triggers", {})
    print()
    for dev in ("TV", "AC"):
        t_list = trigs.get(dev, [])
        print(f"  {dev} triggers: {t_list if t_list else '(none)'}")
    print()


# ─────────────────────────────────────────────
#  2. QUICK TRAIN — speak freely → pick command
# ─────────────────────────────────────────────

def quick_train(recognizer: sr.Recognizer, whisper: WhisperModel):
    """
    Speak any phrase → Whisper transcribes → you pick the command.
    If Whisper didn't detect the trigger word from your speech,
    the tool asks whether to save the first word as a new trigger.
    """
    print(f"\n{'='*65}")
    print("  QUICK TRAIN — Speak freely, pick the command")
    print(f"{'='*65}")
    print("  Type 'q' to quit.\n")

    while True:
        print(f"  {SEP}")
        raw_input = input(
            "  Press Enter to record, or type text directly (q=quit): "
        ).strip()

        if raw_input.lower() == "q":
            break

        if raw_input == "":
            text = record_one(recognizer, whisper)
            if text is None:
                continue
            print(f'  Whisper heard: "{text}"')
        else:
            text = raw_input
            print(f'  Using text: "{text}"')

        print()

        # ── Try auto-detect trigger ──────────────────────────────────────
        words = text.strip().split()
        detected_device = None
        command_part = text.strip()

        from tv_voice_assistant_enhanced import _TRIGGERS
        learned_now = load_learned()
        all_triggers = {
            dev: list(_TRIGGERS[dev]) + learned_now.get("_triggers", {}).get(dev, [])
            for dev in ("TV", "AC")
        }

        for dev, variants in all_triggers.items():
            for v in variants:
                if text.lower().startswith(v.lower() + " ") or text.lower() == v.lower():
                    command_part = text[len(v):].strip()
                    detected_device = dev
                    break
            if detected_device:
                break

        trigger_word = words[0] if words else ""

        if detected_device:
            print(f"  Trigger detected: {detected_device}")
            print(f"  Command part   : \"{command_part}\"")
        else:
            print(f"  No trigger detected in \"{trigger_word}\".")
            print(f"  Which device?  1=TV  2=AC  0=discard")
            d = input("  Choice: ").strip()
            if d == "1":
                detected_device = "TV"
            elif d == "2":
                detected_device = "AC"
            else:
                print("  Discarded.\n")
                continue

            # Offer to save the first word as a trigger
            print()
            print(f'  Save "{trigger_word}" as a {detected_device} trigger word?')
            print("  (This will fix future detection when you say this word.)")
            save_t = input("  [y/n]: ").strip().lower()
            if save_t in ("y", ""):
                learned_now = load_learned()
                if _add_trigger(detected_device, trigger_word, learned_now):
                    save_learned(learned_now)
                    print(f'  Saved "{trigger_word}" as {detected_device} trigger.\n')
                else:
                    print(f'  Already saved.\n')
            command_part = " ".join(words[1:]).strip() if len(words) > 1 else text.strip()
            print(f'  Command part: "{command_part}"')

        if not command_part:
            print("  No command part. Discarded.\n")
            continue

        # ── Pick command ─────────────────────────────────────────────────
        commands = TV_COMMANDS if detected_device == "TV" else AC_COMMANDS
        cmd_list = list(commands.keys())

        print()
        print(f"  Map \"{command_part}\" → which {detected_device} command?")
        print(f"  {SEP}")
        for i, (k, v) in enumerate(commands.items(), 1):
            print(f"  {i:>3}. {k:<22}  {v['description']}")
        print("    0. Discard")
        print()
        raw_num = input("  Command number: ").strip()

        if not raw_num.isdigit() or int(raw_num) == 0:
            print("  Discarded.\n")
            continue

        idx = int(raw_num) - 1
        if not (0 <= idx < len(cmd_list)):
            print("  Invalid number.\n")
            continue

        cmd_key = cmd_list[idx]
        learned_now = load_learned()
        if _add_alias(detected_device, cmd_key, command_part, learned_now):
            save_learned(learned_now)
            print(f'  Saved: "{command_part}" → {detected_device}:{cmd_key}\n')
        else:
            print(f'  Already learned.\n')


# ─────────────────────────────────────────────
#  3. BROWSE-AND-RECORD PER COMMAND
# ─────────────────────────────────────────────

def train_device_menu(device: str, commands: dict,
                      recognizer: sr.Recognizer, whisper: WhisperModel):
    cmd_list = list(commands.keys())

    while True:
        learned = load_learned()
        device_data = learned.get(device, {})
        print(f"\n{'='*65}")
        print(f"  TRAIN {device} COMMANDS")
        print(f"{'='*65}")
        for i, (k, v) in enumerate(commands.items(), 1):
            n = len(device_data.get(k, []))
            star = " *" if n > 0 else ""
            print(f"  {i:>3}. [{v['code']}] {k:<22} {v['description']:<30} ({n}){star}")
        print()
        print("  Enter number, 'a' = train all, 0 = back")
        raw = input("  Choice: ").strip().lower()

        if raw == "0":
            break
        elif raw == "a":
            n_str = input("  Samples per command (default 3): ").strip()
            n = int(n_str) if n_str.isdigit() and int(n_str) > 0 else 3
            for cmd_key in cmd_list:
                print(f"\n  Command: {cmd_key} — {commands[cmd_key]['description']}")
                ans = input("  Train? [y/n/q]: ").strip().lower()
                if ans == "q":
                    break
                if ans == "n":
                    continue
                _record_for_command(device, cmd_key, commands, recognizer, whisper, n)
        elif raw.isdigit():
            idx = int(raw) - 1
            if not (0 <= idx < len(cmd_list)):
                print("  Invalid.")
                continue
            cmd_key = cmd_list[idx]
            n_str = input("  Samples (default 3): ").strip()
            n = int(n_str) if n_str.isdigit() and int(n_str) > 0 else 3
            _record_for_command(device, cmd_key, commands, recognizer, whisper, n)


def _record_for_command(device, cmd_key, commands,
                        recognizer, whisper, n_samples):
    learned = load_learned()
    existing = learned.get(device, {}).get(cmd_key, [])
    cmd_data = commands[cmd_key]

    print(f"\n  [{cmd_data['code']}] {cmd_key} — {cmd_data['description']}")
    print(f"  Recorded so far: {len(existing)}")
    if existing:
        for a in existing[-3:]:
            print(f"    \"{a}\"")
    print()

    for i in range(n_samples):
        print(f"  Sample {i+1}/{n_samples}")
        text = record_one(recognizer, whisper)
        if not text:
            continue

        # Strip trigger word if present
        clean = text.strip()
        from tv_voice_assistant_enhanced import _TRIGGERS
        for variants in _TRIGGERS.values():
            for v in variants:
                if clean.lower().startswith(v.lower() + " "):
                    clean = clean[len(v):].strip()
                    break

        print(f'  Whisper heard : "{text}"')
        if clean != text.strip():
            print(f'  Command part  : "{clean}"')
        print()

        while True:
            print("  [y] Save  [n] Discard  [r] Re-record  [q] Quit")
            ans = input("  Choice: ").strip().lower()
            if ans in ("y", ""):
                learned = load_learned()
                if _add_alias(device, cmd_key, clean, learned):
                    save_learned(learned)
                    print(f'  Saved "{clean}" → {device}:{cmd_key}\n')
                else:
                    print("  Already saved.\n")
                break
            elif ans == "n":
                print("  Discarded.\n")
                break
            elif ans == "r":
                text = record_one(recognizer, whisper)
                if text:
                    clean = text.strip()
                    print(f'  Whisper heard: "{text}"')
            elif ans == "q":
                return
            else:
                print("  Please enter y / n / r / q")


# ─────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────

def show_stats():
    learned = load_learned()
    print(f"\n{'='*65}")
    print("  LEARNED DATA")
    print(f"{'='*65}")

    print("\n  TRIGGER WORDS:")
    print(f"  {SEP}")
    trigs = learned.get("_triggers", {})
    any_trig = False
    for dev in ("TV", "AC"):
        t_list = trigs.get(dev, [])
        if t_list:
            print(f"  {dev}:")
            for t in t_list:
                print(f"    \"{t}\"")
            any_trig = True
    if not any_trig:
        print("  (none yet — run option 1 first!)")

    print(f"\n  COMMAND ALIASES:")
    print(f"  {SEP}")
    total = 0
    for device in ("TV", "AC"):
        cmds = learned.get(device, {})
        if not cmds:
            continue
        print(f"\n  {device}:")
        for cmd, aliases in sorted(cmds.items()):
            print(f"    {cmd:<22} {len(aliases)} sample(s)")
            for a in aliases:
                print(f"      \"{a}\"")
            total += len(aliases)
    if total == 0:
        print("  (none yet)")
    print(f"\n  Total: {total} command alias(es)\n")


def clear_all():
    print("  Clear ALL learned data? [y/n]: ", end="")
    if input().strip().lower() == "y":
        save_learned({"TV": {}, "AC": {}, "_triggers": {"TV": [], "AC": []}})
        print("  Cleared.\n")
    else:
        print("  Cancelled.\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voice Training Tool")
    parser.add_argument("--model", default="tiny",
                        choices=["tiny", "small", "medium"])
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  VOICE TRAINING TOOL — Smart Home Assistant")
    print("=" * 65)
    print(f"  Whisper model : {args.model}")
    print(f"  Saved to      : {LEARNED_FILE}")
    print()
    print("  RECOMMENDED ORDER:")
    print("  1. First train TRIGGER WORDS (so 'ديبي'/'كلم' are recognized)")
    print("  2. Then train COMMANDS (so your Darija words are matched)")
    print()

    print("[STT] Loading Whisper...")
    whisper = WhisperModel(args.model, device="cpu", compute_type="int8")
    silence = np.zeros(16000, dtype=np.float32)
    list(whisper.transcribe(silence, language="ar", beam_size=1)[0])
    print("[STT] Ready!\n")

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6
    calibrate_mic(recognizer)

    while True:
        print(f"\n  {'='*55}")
        print("  MAIN MENU")
        print(f"  {'─'*55}")
        print("  1. Train TRIGGER WORDS  ← DO THIS FIRST")
        print("       (teach system your 'تيفي' / 'كليم' voice)")
        print("  2. Quick train commands")
        print("       (speak freely → pick the command)")
        print("  3. Train TV commands  (browse list)")
        print("  4. Train AC commands  (browse list)")
        print("  5. Show all learned data")
        print("  6. Clear all learned data")
        print("  7. Exit")
        print()
        c = input("  Choice: ").strip()

        if c == "1":
            train_trigger_words(recognizer, whisper)
        elif c == "2":
            quick_train(recognizer, whisper)
        elif c == "3":
            train_device_menu("TV", TV_COMMANDS, recognizer, whisper)
        elif c == "4":
            train_device_menu("AC", AC_COMMANDS, recognizer, whisper)
        elif c == "5":
            show_stats()
        elif c == "6":
            clear_all()
        elif c == "7":
            print("  Bye!\n")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
