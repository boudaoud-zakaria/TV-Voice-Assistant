"""
SMART HOME VOICE ASSISTANT - TESTING & DEBUGGING UTILITIES
===========================================================
Offline tests for TV + AC command matching. No microphone needed.
Usage:
    python test_and_debug_utilities.py
"""

import time
import json
from tv_voice_assistant_enhanced import (
    DeviceCommandMatcher, detect_device,
    TV_COMMANDS, AC_COMMANDS, PerformanceTracker,
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _build_matchers(quiet=True):
    tv = DeviceCommandMatcher("TV", TV_COMMANDS)
    ac = DeviceCommandMatcher("AC", AC_COMMANDS)
    return tv, ac


def _row(ok, desc, inp, expected, got, conf, ms):
    mark = "OK " if ok else "FAIL"
    print(f"[{mark}] {desc:<35} | '{inp}'")
    print(f"       expected={expected!r:15}  got={got!r:15}  conf={conf:.2f}  {ms:.1f}ms")


# ─────────────────────────────────────────────
#  1. COMMAND MATCHING
# ─────────────────────────────────────────────

def test_command_matching():
    print("=" * 65)
    print("  COMMAND MATCHING TEST  (TV + AC)")
    print("=" * 65)
    tv, ac = _build_matchers()

    cases = [
        # ── TV Power ──────────────────────────
        (tv, "open",          "open",         "TV exact EN"),
        (tv, "turn on",       "open",         "TV substring EN"),
        (tv, "allumer",       "open",         "TV exact FR"),
        (tv, "شعل",           "open",         "TV exact AR"),
        (tv, "طفي",           "close",        "TV exact AR off"),
        (tv, "éteindre",      "close",        "TV exact FR off"),
        # ── TV Volume ─────────────────────────
        (tv, "volume up",     "volume up",    "TV exact EN"),
        (tv, "زيد الصوت",     "volume up",    "TV exact Darija"),
        (tv, "نقص الصوت",     "volume down",  "TV exact Darija"),
        (tv, "mute",          "mute",         "TV exact EN"),
        (tv, "كتم الصوت",     "mute",         "TV exact AR"),
        # ── TV new commands ───────────────────
        (tv, "zoom",          "zoom",         "TV zoom EN"),
        (tv, "كبر الصورة",    "zoom",         "TV zoom AR"),
        (tv, "sous-titres",   "subtitles",    "TV sub FR"),
        (tv, "ترجمة",         "subtitles",    "TV sub AR"),
        (tv, "info",          "info",         "TV info EN"),
        (tv, "معلومات",       "info",         "TV info AR"),
        (tv, "list",          "list",         "TV list EN"),
        (tv, "liste",         "list",         "TV list FR"),
        (tv, "live",          "live",         "TV live EN"),
        (tv, "مباشر",         "live",         "TV live AR"),
        (tv, "hdmi 1",        "hdmi1",        "TV hdmi1 EN"),
        (tv, "hdmi واحد",     "hdmi1",        "TV hdmi1 AR"),
        (tv, "2",             "digit_2",      "TV digit 2"),
        (tv, "جوج",           "digit_2",      "TV digit Darija جوج"),
        (tv, "تلاتة",         "digit_3",      "TV digit Darija تلاتة"),
        # ── AC Power ──────────────────────────
        (ac, "شعل الكليم",    "open",         "AC on Darija"),
        (ac, "allumer le clim","open",        "AC on FR"),
        (ac, "طفي الكليم",    "close",        "AC off Darija"),
        # ── AC Modes ──────────────────────────
        (ac, "cool",          "cool",         "AC cool EN"),
        (ac, "برد",           "cool",         "AC cool AR"),
        (ac, "froid",         "cool",         "AC cool FR"),
        (ac, "heat",          "heat",         "AC heat EN"),
        (ac, "سخن",           "heat",         "AC heat AR"),
        (ac, "dry",           "dry",          "AC dry EN"),
        (ac, "جاف",           "dry",          "AC dry AR"),
        # ── AC Fan speeds ─────────────────────
        (ac, "fan low",       "fan_low",      "AC fan_low EN"),
        (ac, "ريح هادية",     "fan_low",      "AC fan_low AR"),
        (ac, "fan medium",    "fan_med",      "AC fan_med EN"),
        (ac, "ريح متوسطة",    "fan_med",      "AC fan_med AR"),
        (ac, "fan high",      "fan_high",     "AC fan_high EN"),
        (ac, "ريح قوية",      "fan_high",     "AC fan_high AR"),
        (ac, "fan auto",      "fan_auto",     "AC fan_auto EN"),
        (ac, "ريح أوتو",      "fan_auto",     "AC fan_auto AR"),
        # ── AC Temperatures ───────────────────
        (ac, "16",            "temp_16",      "AC temp 16"),
        (ac, "ستاش درجة",     "temp_16",      "AC temp 16 Darija"),
        (ac, "عشرين",         "temp_20",      "AC temp 20 Darija"),
        (ac, "24",            "temp_24",      "AC temp 24"),
        (ac, "تلاتين",        "temp_30",      "AC temp 30 Darija"),
        # ── AC Swing ──────────────────────────
        (ac, "swing",         "swing",        "AC swing EN"),
        (ac, "دوّر الريش",    "swing",        "AC swing AR"),
        # ── Edge cases ────────────────────────
        (tv, "",              None,           "empty input"),
        (tv, "xyzqwerty",     None,           "nonsense"),
    ]

    passed = 0
    for matcher, inp, expected, desc in cases:
        t0 = time.time()
        r = matcher.match(inp)
        ms = (time.time() - t0) * 1000
        ok = r["command"] == expected
        if ok:
            passed += 1
        _row(ok, desc, inp, expected, r["command"], r["confidence"], ms)

    total = len(cases)
    print("-" * 65)
    print(f"RESULT: {passed}/{total} passed  ({passed/total*100:.1f}%)\n")


# ─────────────────────────────────────────────
#  2. DEVICE TRIGGER DETECTION
# ─────────────────────────────────────────────

def test_trigger_detection():
    print("=" * 65)
    print("  DEVICE TRIGGER DETECTION TEST")
    print("=" * 65)

    cases = [
        # (utterance, expected_device, expected_remainder_not_none)
        ("tv open",               "TV",  True),
        ("TV volume up",          "TV",  True),
        ("تيفي شعل",              "TV",  True),
        ("التلفزيون زيد الصوت",   "TV",  True),
        ("كليم شعل",              "AC",  True),
        ("clim cool",             "AC",  True),
        ("AC fan low",            "AC",  True),
        ("المكيف برد",            "AC",  True),
        ("tv",                    "TV",  False),  # trigger only, no remainder
        ("open",                  None,  False),  # no trigger
        ("netflix",               None,  False),
    ]

    passed = 0
    for utterance, exp_device, exp_has_remainder in cases:
        device, remainder = detect_device(utterance)
        ok_device = device == exp_device
        ok_rem = (remainder is not None) == exp_has_remainder
        ok = ok_device and ok_rem
        if ok:
            passed += 1
        mark = "OK " if ok else "FAIL"
        print(f"[{mark}] '{utterance}'")
        print(f"       device={device!r} (exp {exp_device!r})  "
              f"remainder={remainder!r} (has_rem={remainder is not None})\n")

    total = len(cases)
    print("-" * 65)
    print(f"RESULT: {passed}/{total} passed  ({passed/total*100:.1f}%)\n")


# ─────────────────────────────────────────────
#  3. DATABASE STATISTICS
# ─────────────────────────────────────────────

def print_database_stats():
    print("=" * 65)
    print("  COMMAND DATABASE STATISTICS")
    print("=" * 65)

    for label, db in [("TV", TV_COMMANDS), ("AC", AC_COMMANDS)]:
        cats = {}
        total_aliases = 0
        for cmd, data in db.items():
            cat = data.get("category", "Other")
            n = len(data["aliases"])
            total_aliases += n
            cats.setdefault(cat, {"cmds": 0, "aliases": 0})
            cats[cat]["cmds"] += 1
            cats[cat]["aliases"] += n

        print(f"\n{label} ({len(db)} commands, {total_aliases} aliases)")
        print("-" * 50)
        for cat, s in sorted(cats.items()):
            print(f"  {cat:<18} {s['cmds']:>2} cmds  {s['aliases']:>4} aliases")
        print(f"  {'TOTAL':<18} {len(db):>2} cmds  {total_aliases:>4} aliases")

    print()


# ─────────────────────────────────────────────
#  4. PERFORMANCE BENCHMARK
# ─────────────────────────────────────────────

def benchmark_matching():
    print("=" * 65)
    print("  PERFORMANCE BENCHMARK  (1 000 iterations each)")
    print("=" * 65)
    tv, ac = _build_matchers()

    scenarios = [
        (tv, "open",               "TV exact match"),
        (tv, "power on",           "TV substring match"),
        (tv, "volum up",           "TV fuzzy match"),
        (tv, "make it louder",     "TV semantic match"),
        (ac, "fan low",            "AC exact match"),
        (ac, "ستاش درجة",          "AC Darija exact"),
    ]

    for matcher, inp, label in scenarios:
        times = []
        for _ in range(1000):
            t0 = time.time()
            matcher.match(inp)
            times.append((time.time() - t0) * 1000)
        avg = sum(times) / len(times)
        print(f"  {label:<30} avg={avg:.2f}ms  min={min(times):.2f}ms  max={max(times):.2f}ms")

    print()


# ─────────────────────────────────────────────
#  5. COMPLETE COMMAND REFERENCE
# ─────────────────────────────────────────────

def print_command_reference():
    print("=" * 65)
    print("  COMPLETE COMMAND REFERENCE")
    print("=" * 65)
    for label, db in [("TV", TV_COMMANDS), ("AC", AC_COMMANDS)]:
        print(f"\n{'─'*65}\n  {label} COMMANDS\n{'─'*65}")
        prev_cat = None
        for cmd, data in sorted(db.items(), key=lambda x: x[1]["code"]):
            cat = data["category"]
            if cat != prev_cat:
                print(f"\n  [{cat}]")
                prev_cat = cat
            n = len(data["aliases"])
            print(f"    {data['code']}  {cmd:<18} {data['description']:<30} ({n} aliases)")
    print()


# ─────────────────────────────────────────────
#  6. JSON EXPORT
# ─────────────────────────────────────────────

def export_commands_json(filename="smart_home_commands.json"):
    payload = {
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "TV": {
            "total_commands": len(TV_COMMANDS),
            "commands": {
                k: {
                    "code": v["code"],
                    "category": v["category"],
                    "description": v["description"],
                    "alias_count": len(v["aliases"]),
                    "aliases": v["aliases"],
                }
                for k, v in TV_COMMANDS.items()
            },
        },
        "AC": {
            "total_commands": len(AC_COMMANDS),
            "commands": {
                k: {
                    "code": v["code"],
                    "category": v["category"],
                    "description": v["description"],
                    "alias_count": len(v["aliases"]),
                    "aliases": v["aliases"],
                }
                for k, v in AC_COMMANDS.items()
            },
        },
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Exported to {filename}  "
          f"(TV={len(TV_COMMANDS)} cmds, AC={len(AC_COMMANDS)} cmds)\n")


# ─────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  SMART HOME VOICE ASSISTANT — TEST UTILITIES")
    print("=" * 65)
    print("  1. Command Matching Test  (TV + AC, all languages)")
    print("  2. Device Trigger Detection Test")
    print("  3. Database Statistics")
    print("  4. Performance Benchmark")
    print("  5. Complete Command Reference")
    print("  6. Export Commands to JSON")
    print("  7. Run All\n")

    choice = input("Select (1-7): ").strip()

    if choice == "1":
        test_command_matching()
    elif choice == "2":
        test_trigger_detection()
    elif choice == "3":
        print_database_stats()
    elif choice == "4":
        benchmark_matching()
    elif choice == "5":
        print_command_reference()
    elif choice == "6":
        export_commands_json()
    elif choice == "7":
        test_command_matching()
        test_trigger_detection()
        print_database_stats()
        benchmark_matching()
        print_command_reference()
        export_commands_json()
    else:
        print("Invalid choice.")
