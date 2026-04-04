"""
TV VOICE ASSISTANT - TESTING & DEBUGGING UTILITIES
===================================================
Quick testing tools to verify system functionality
"""

import time
import json
from tv_voice_assistant_enhanced import TVCommandMatcher, TV_COMMANDS, PerformanceTracker

# ─────────────────────────────────────────────
#  COMMAND MATCHER TESTER
# ─────────────────────────────────────────────

def test_command_matching():
    """Test command matching without microphone"""
    print("═" * 60)
    print("   🧪 COMMAND MATCHING TEST")
    print("═" * 60 + "\n")
    
    matcher = TVCommandMatcher(TV_COMMANDS, show_progress=False)
    
    # Test cases: (input, expected_command, description)
    test_cases = [
        # English - Exact matches
        ("open", "open", "English exact match"),
        ("turn on", "open", "English substring"),
        ("power on", "open", "English alias"),
        
        # English - Fuzzy matches
        ("volum up", "volume up", "English fuzzy typo"),
        ("channl up", "channel up", "English fuzzy typo"),
        ("playe", "play", "English fuzzy typo"),
        
        # English - Semantic matches
        ("make it louder", "volume up", "English semantic"),
        ("next program", "channel up", "English semantic"),
        ("watch netflix", "netflix", "English semantic"),
        
        # French
        ("allumer", "open", "French exact"),
        ("augmenter le volume", "volume up", "French exact"),
        ("netflix", "netflix", "French cross-language"),
        
        # Arabic Dariha
        ("افتح", "open", "Arabic exact"),
        ("ارفع الصوت", "volume up", "Arabic exact"),
        ("نيتفليكس", "netflix", "Arabic cross-language"),
        ("شغّل", "open", "Arabic Dariha variant"),
        
        # Edge cases
        ("tv open", "open", "With TV trigger (will extract)"),
        ("", None, "Empty input"),
        ("xyzabc", None, "Nonsense input"),
    ]
    
    results = []
    
    for test_input, expected, description in test_cases:
        start = time.time()
        result = matcher.match(test_input)
        elapsed = time.time() - start
        
        matched = result["command"] == expected
        status = "✅" if matched else "❌"
        
        print(f"{status} {description:<30} | Input: \"{test_input}\"")
        print(f"   Expected: {expected}, Got: {result['command']}, "
              f"Confidence: {result['confidence']:.2f}, Time: {elapsed*1000:.2f}ms")
        print()
        
        results.append({
            "description": description,
            "input": test_input,
            "expected": expected,
            "got": result["command"],
            "matched": matched,
            "confidence": result["confidence"],
            "time_ms": elapsed * 1000
        })
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["matched"])
    
    print("─" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("═" * 60 + "\n")
    
    return results

# ─────────────────────────────────────────────
#  TRIGGER WORD DETECTION TESTER
# ─────────────────────────────────────────────

def test_trigger_detection():
    """Test 'TV' trigger word detection"""
    print("═" * 60)
    print("   🎯 TRIGGER WORD DETECTION TEST")
    print("═" * 60 + "\n")
    
    matcher = TVCommandMatcher(TV_COMMANDS, show_progress=False)
    
    test_cases = [
        ("TV open", True, "open", "Perfect trigger"),
        ("Tv volume up", True, "volume up", "Lowercase tv"),
        ("T V mute", True, None, "Spaced T V"),
        ("tiv netflix", True, None, "Phonetic tiv"),
        ("TV", True, None, "TV only"),
        ("open", False, None, "No trigger"),
        ("Can you tv open", False, None, "TV not at start"),
        ("تي في افتح", True, None, "Arabic تي في"),
    ]
    
    results = []
    
    for test_input, should_trigger, expected_command, description in test_cases:
        detected, remainder = matcher.detect_trigger(test_input)
        
        status = "✅" if detected == should_trigger else "❌"
        
        print(f"{status} {description:<30} | Input: \"{test_input}\"")
        print(f"   Triggered: {detected}, Remainder: \"{remainder}\"")
        print()
        
        results.append({
            "description": description,
            "input": test_input,
            "triggered": detected,
            "expected_trigger": should_trigger,
            "remainder": remainder
        })
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["triggered"] == r["expected_trigger"])
    
    print("─" * 60)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("═" * 60 + "\n")
    
    return results

# ─────────────────────────────────────────────
#  DATABASE STATISTICS
# ─────────────────────────────────────────────

def print_database_stats():
    """Print command database statistics"""
    print("═" * 60)
    print("   📊 COMMAND DATABASE STATISTICS")
    print("═" * 60 + "\n")
    
    categories = {}
    total_aliases = 0
    
    for cmd_name, cmd_data in TV_COMMANDS.items():
        category = cmd_data.get("category", "Unknown")
        alias_count = len(cmd_data["aliases"])
        total_aliases += alias_count
        
        if category not in categories:
            categories[category] = {"count": 0, "aliases": 0}
        
        categories[category]["count"] += 1
        categories[category]["aliases"] += alias_count
    
    print("Category Breakdown:")
    print("─" * 60)
    for category, stats in sorted(categories.items()):
        print(f"  {category:<15} : {stats['count']:>2} commands, {stats['aliases']:>3} aliases")
    
    print("─" * 60)
    print(f"  TOTAL           : {len(TV_COMMANDS):>2} commands, {total_aliases:>3} aliases")
    print("=" * 60 + "\n")
    
    print("Language Distribution:")
    print("─" * 60)
    
    langs = {"English": 0, "French": 0, "Arabic": 0}
    
    for cmd_name, cmd_data in TV_COMMANDS.items():
        aliases = cmd_data["aliases"]
        for alias in aliases:
            # Simple heuristic
            if any(ord(c) > 1200 for c in alias):  # Arabic unicode range
                langs["Arabic"] += 1
            elif alias in ["allumer", "ouvrir", "augmenter", "diminuer", "chaîne", "accueil",
                          "paramètres", "réglages", "éteindre", "fermer"]:
                langs["French"] += 1
            else:
                langs["English"] += 1
    
    for lang, count in langs.items():
        print(f"  {lang:<15} : {count:>3} aliases ({count/total_aliases*100:.1f}%)")
    
    print("═" * 60 + "\n")

# ─────────────────────────────────────────────
#  PERFORMANCE BENCHMARK
# ─────────────────────────────────────────────

def benchmark_matching():
    """Benchmark command matching speed"""
    print("═" * 60)
    print("   ⚡ PERFORMANCE BENCHMARK")
    print("═" * 60 + "\n")
    
    print("Initializing matcher...")
    start_total = time.time()
    
    matcher = TVCommandMatcher(TV_COMMANDS, show_progress=False)
    
    init_time = time.time() - start_total
    print(f"✓ Initialization time: {init_time:.2f}s\n")
    
    # Test different matching scenarios
    test_inputs = [
        ("open", "Exact match"),
        ("power on", "Substring match"),
        ("volum up", "Fuzzy match"),
        ("make it louder", "Semantic match"),
    ]
    
    print("Matching Speed Tests (1000 iterations):")
    print("─" * 60)
    
    for test_input, method in test_inputs:
        times = []
        
        for _ in range(1000):
            start = time.time()
            result = matcher.match(test_input)
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n{method}: \"{test_input}\"")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min:     {min_time:.2f}ms")
        print(f"  Max:     {max_time:.2f}ms")
    
    print("\n" + "═" * 60 + "\n")

# ─────────────────────────────────────────────
#  COMMAND CODE REFERENCE
# ─────────────────────────────────────────────

def print_command_reference():
    """Print all commands with codes"""
    print("═" * 60)
    print("   📋 COMPLETE COMMAND REFERENCE")
    print("═" * 60 + "\n")
    
    current_category = None
    
    for cmd_name, cmd_data in sorted(TV_COMMANDS.items(), 
                                     key=lambda x: x[1]["code"]):
        category = cmd_data["category"]
        
        if category != current_category:
            if current_category is not None:
                print()
            print(f"\n{category.upper()}")
            print("─" * 60)
            current_category = category
        
        code = cmd_data["code"]
        desc = cmd_data["description"]
        aliases_count = len(cmd_data["aliases"])
        
        print(f"  {code} | {cmd_name:<15} | {desc:<25} | {aliases_count} aliases")
    
    print("\n" + "═" * 60 + "\n")

# ─────────────────────────────────────────────
#  JSON EXPORT
# ─────────────────────────────────────────────

def export_commands_json(filename="tv_commands.json"):
    """Export all commands to JSON"""
    print(f"Exporting commands to {filename}...")
    
    export_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_commands": len(TV_COMMANDS),
        "commands": {}
    }
    
    for cmd_name, cmd_data in TV_COMMANDS.items():
        export_data["commands"][cmd_name] = {
            "code": cmd_data["code"],
            "category": cmd_data["category"],
            "description": cmd_data["description"],
            "aliases": cmd_data["aliases"],
            "alias_count": len(cmd_data["aliases"])
        }
    
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
    print("║   TV VOICE ASSISTANT - TESTING & DEBUGGING UTILITIES      ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    print("Available Tests:")
    print("  1. Command Matching Test")
    print("  2. Trigger Word Detection Test")
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
        print("Running all tests...\n")
        test_command_matching()
        test_trigger_detection()
        print_database_stats()
        benchmark_matching()
        print_command_reference()
        export_commands_json()
    else:
        print("Invalid choice!")
