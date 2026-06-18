"""
MULTI-DEVICE VOICE ASSISTANT - TESTING & DEBUGGING UTILITIES
=============================================================
Quick testing tools to verify system functionality (TV + Air Conditioner)
"""

import time
import json
from tv_voice_assistant_enhanced import MultiDeviceAssistant, DEVICES, PerformanceTracker

# ─────────────────────────────────────────────
#  COMMAND MATCHER TESTER
# ─────────────────────────────────────────────

def test_command_matching():
    """Test command matching without microphone (both devices)"""
    print("═" * 60)
    print("   🧪 COMMAND MATCHING TEST")
    print("═" * 60 + "\n")

    assistant = MultiDeviceAssistant(DEVICES, show_progress=False)

    # Test cases: (device_key, input, expected_command, description)
    test_cases = [
        # ── TV ──────────────────────────────────────────
        ("tv", "open", "open", "TV English exact"),
        ("tv", "turn on", "open", "TV English substring"),
        ("tv", "volum up", "volume up", "TV English fuzzy typo"),
        ("tv", "make it louder", "volume up", "TV English semantic"),
        ("tv", "watch netflix", "netflix", "TV semantic app"),
        ("tv", "افتح", "open", "TV Arabic exact"),
        ("tv", "ارفع الصوت", "volume up", "TV Arabic Darija"),
        ("tv", "نيتفليكس", "netflix", "TV Arabic app"),

        # ── Air Conditioner ─────────────────────────────
        ("ac", "power on", "power on", "AC English exact"),
        ("ac", "turn off", "power off", "AC English alias"),
        ("ac", "cooler", "temp down", "AC English temp"),
        ("ac", "cool", "cool mode", "AC mode"),
        ("ac", "ولّع الكليما", "power on", "AC Arabic Darija on"),
        ("ac", "طفّي", "power off", "AC Arabic Darija off"),
        ("ac", "تبريد", "cool mode", "AC Arabic cool"),
        ("ac", "زيد الحرارة", "temp up", "AC Arabic temp up"),

        # ── Edge cases ──────────────────────────────────
        ("tv", "", None, "Empty input"),
        ("ac", "xyzabc", None, "Nonsense input"),
    ]

    results = []

    for device_key, test_input, expected, description in test_cases:
        matcher = assistant.matchers[device_key]
        start = time.time()
        result = matcher.match(test_input)
        elapsed = time.time() - start

        matched = result["command"] == expected
        status = "✅" if matched else "❌"

        print(f"{status} {description:<28} | [{DEVICES[device_key]['name']}] \"{test_input}\"")
        print(f"   Expected: {expected}, Got: {result['command']}, "
              f"Output: {result['full_code']}, "
              f"Conf: {result['confidence']:.2f}, Time: {elapsed*1000:.2f}ms")
        print()

        results.append({
            "description": description,
            "device": device_key,
            "input": test_input,
            "expected": expected,
            "got": result["command"],
            "full_code": result["full_code"],
            "matched": matched,
            "confidence": result["confidence"],
            "time_ms": elapsed * 1000
        })

    total = len(results)
    passed = sum(1 for r in results if r["matched"])

    print("─" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("═" * 60 + "\n")

    return results

# ─────────────────────────────────────────────
#  DEVICE DETECTION TESTER
# ─────────────────────────────────────────────

def test_device_detection():
    """Test device-name detection (TV / Air Conditioner)"""
    print("═" * 60)
    print("   🎯 DEVICE DETECTION TEST")
    print("═" * 60 + "\n")

    assistant = MultiDeviceAssistant(DEVICES, show_progress=False)

    # (input, expected_device_key, expected_remainder_present, description)
    test_cases = [
        ("tv open", "tv", True, "TV + command"),
        ("Tv volume up", "tv", True, "Lowercase tv"),
        ("tv", "tv", False, "TV only"),
        ("clem power on", "ac", True, "AC 'clem' + command"),
        ("clim cool", "ac", True, "AC 'clim' variant"),
        ("clem", "ac", False, "AC only"),
        ("المكيف تبريد", "ac", True, "Arabic AC name"),
        ("التلفزيون افتح", "tv", True, "Arabic TV name"),
        ("hello world", None, False, "No device"),
        ("open", None, False, "Command without device"),
    ]

    results = []

    for test_input, expected_key, has_remainder, description in test_cases:
        device_key, remainder = assistant.detect_device(test_input)

        key_ok = device_key == expected_key
        rem_ok = (remainder is not None) == has_remainder
        ok = key_ok and rem_ok
        status = "✅" if ok else "❌"

        name = DEVICES[device_key]["name"] if device_key else "—"
        print(f"{status} {description:<28} | Input: \"{test_input}\"")
        print(f"   Device: {name}, Remainder: \"{remainder}\"")
        print()

        results.append({
            "description": description,
            "input": test_input,
            "device": device_key,
            "expected": expected_key,
            "remainder": remainder,
            "ok": ok,
        })

    total = len(results)
    passed = sum(1 for r in results if r["ok"])

    print("─" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("═" * 60 + "\n")

    return results

# ─────────────────────────────────────────────
#  DATABASE STATISTICS
# ─────────────────────────────────────────────

def print_database_stats():
    """Print command database statistics for every device"""
    print("═" * 60)
    print("   📊 COMMAND DATABASE STATISTICS")
    print("═" * 60 + "\n")

    grand_commands = 0
    grand_aliases = 0

    for dev_key, dev in DEVICES.items():
        commands = dev["commands"]
        categories = {}
        total_aliases = 0

        for cmd_name, cmd_data in commands.items():
            category = cmd_data.get("category", "Unknown")
            alias_count = len(cmd_data["aliases"])
            total_aliases += alias_count
            categories.setdefault(category, {"count": 0, "aliases": 0})
            categories[category]["count"] += 1
            categories[category]["aliases"] += alias_count

        print(f"🎛️  {dev['name']}  (output prefix '{dev['prefix']}')")
        print("─" * 60)
        for category, stats in sorted(categories.items()):
            print(f"  {category:<15} : {stats['count']:>2} commands, {stats['aliases']:>3} aliases")
        print("─" * 60)
        print(f"  TOTAL           : {len(commands):>2} commands, {total_aliases:>3} aliases\n")

        grand_commands += len(commands)
        grand_aliases += total_aliases

    print("═" * 60)
    print(f"  ALL DEVICES     : {grand_commands} commands, {grand_aliases} aliases")
    print("═" * 60 + "\n")

# ─────────────────────────────────────────────
#  PERFORMANCE BENCHMARK
# ─────────────────────────────────────────────

def benchmark_matching():
    """Benchmark command matching speed"""
    print("═" * 60)
    print("   ⚡ PERFORMANCE BENCHMARK")
    print("═" * 60 + "\n")

    print("Initializing assistant...")
    start_total = time.time()
    assistant = MultiDeviceAssistant(DEVICES, show_progress=False)
    init_time = time.time() - start_total
    print(f"✓ Initialization time: {init_time:.2f}s\n")

    # (device_key, input, method)
    test_inputs = [
        ("tv", "open", "TV exact match"),
        ("tv", "power on", "TV substring match"),
        ("tv", "volum up", "TV fuzzy match"),
        ("tv", "make it louder", "TV semantic match"),
        ("ac", "power on", "AC exact match"),
        ("ac", "cool", "AC substring match"),
        ("ac", "make it cooler", "AC semantic match"),
    ]

    print("Matching Speed Tests (200 iterations each):")
    print("─" * 60)

    for device_key, test_input, method in test_inputs:
        matcher = assistant.matchers[device_key]
        times = []
        for _ in range(200):
            start = time.time()
            matcher.match(test_input)
            times.append((time.time() - start) * 1000)

        print(f"\n{method}: \"{test_input}\"")
        print(f"  Average: {sum(times)/len(times):.2f}ms")
        print(f"  Min:     {min(times):.2f}ms")
        print(f"  Max:     {max(times):.2f}ms")

    print("\n" + "═" * 60 + "\n")

# ─────────────────────────────────────────────
#  COMMAND CODE REFERENCE
# ─────────────────────────────────────────────

def print_command_reference():
    """Print all commands with their full output codes (prefix + code)"""
    print("═" * 60)
    print("   📋 COMPLETE COMMAND REFERENCE")
    print("═" * 60 + "\n")

    for dev_key, dev in DEVICES.items():
        print(f"🎛️  {dev['name']}  (output prefix '{dev['prefix']}')")
        print("=" * 60)

        current_category = None
        for cmd_name, cmd_data in sorted(dev["commands"].items(),
                                         key=lambda x: x[1]["code"]):
            category = cmd_data["category"]
            if category != current_category:
                print(f"\n{category.upper()}")
                print("─" * 60)
                current_category = category

            full_code = dev["prefix"] + cmd_data["code"]
            desc = cmd_data["description"]
            aliases_count = len(cmd_data["aliases"])
            print(f"  {full_code} | {cmd_name:<15} | {desc:<28} | {aliases_count} aliases")

        print("\n" + "═" * 60 + "\n")

# ─────────────────────────────────────────────
#  JSON EXPORT
# ─────────────────────────────────────────────

def export_commands_json(filename="device_commands.json"):
    """Export all device commands to JSON"""
    print(f"Exporting commands to {filename}...")

    export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "devices": {}
    }

    for dev_key, dev in DEVICES.items():
        device_entry = {
            "name": dev["name"],
            "prefix": dev["prefix"],
            "triggers": dev["triggers"],
            "total_commands": len(dev["commands"]),
            "commands": {}
        }
        for cmd_name, cmd_data in dev["commands"].items():
            device_entry["commands"][cmd_name] = {
                "code": cmd_data["code"],
                "full_code": dev["prefix"] + cmd_data["code"],
                "category": cmd_data["category"],
                "description": cmd_data["description"],
                "aliases": cmd_data["aliases"],
                "alias_count": len(cmd_data["aliases"])
            }
        export_data["devices"][dev_key] = device_entry

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Exported to {filename}\n")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   MULTI-DEVICE VOICE ASSISTANT - TESTING & DEBUGGING      ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    print("Available Tests:")
    print("  1. Command Matching Test")
    print("  2. Device Detection Test")
    print("  3. Database Statistics")
    print("  4. Performance Benchmark")
    print("  5. Command Reference")
    print("  6. Export Commands to JSON")
    print("  7. Run All Tests")
    print("\n")

    choice = input("Select test (1-7): ").strip()

    if choice == "1":
        test_command_matching()
    elif choice == "2":
        test_device_detection()
    elif choice == "3":
        print_database_stats()
    elif choice == "4":
        benchmark_matching()
    elif choice == "5":
        print_command_reference()
    elif choice == "6":
        export_commands_json()
    elif choice == "7":
        print("Running all tests...\n")
        test_command_matching()
        test_device_detection()
        print_database_stats()
        benchmark_matching()
        print_command_reference()
        export_commands_json()
    else:
        print("Invalid choice!")
