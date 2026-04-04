"""
TV Voice Assistant - Live Microphone Mode
==========================================
Listens from mic, understands your command, returns the TV code.

Run:
    python tv_voice_mic.py
    python tv_voice_mic.py --lang fr-FR
    python tv_voice_mic.py --lang ar-DZ
"""

import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
import argparse
import sys

# ─────────────────────────────────────────────
#  TV COMMANDS
# ─────────────────────────────────────────────
TV_COMMANDS = {
    "open": {
        "code": "0001",
        "aliases": [
            "open", "turn on", "power on", "start", "on",
            "switch on", "wake up", "boot", "launch", "activate",
            "allumer", "ouvrir", "افتح", "شغّل", "تشغيل",
        ],
        "description": "Turn ON the TV"
    },
    "close": {
        "code": "0002",
        "aliases": [
            "close", "cloths", "clouse", "clothes", "cloze",
            "turn off", "power off", "shutdown", "off",
            "switch off", "stop", "quit", "exit", "sleep",
            "éteindre", "fermer", "أغلق", "إيقاف", "أوقف",
        ],
        "description": "Turn OFF the TV"
    },
    "settings": {
        "code": "0003",
        "aliases": [
            "settings", "setting", "options", "preferences",
            "configure", "configuration", "setup", "menu",
            "paramètres", "réglages", "الإعدادات", "ضبط", "إعدادات",
        ],
        "description": "Open Settings menu"
    },
    "mute": {
        "code": "0004",
        "aliases": [
            "mute", "muet", "silent", "silence", "quiet",
            "no sound", "no audio", "shh", "muting",
            "sourdine", "couper le son", "كتم", "صامت", "أسكت",
        ],
        "description": "Mute / Unmute"
    },
    "volume up": {
        "code": "0005",
        "aliases": [
            "volume up", "add volume", "increase volume", "louder",
            "turn up", "raise volume", "more sound", "higher volume",
            "sound up", "amplify", "boost volume",
            "augmenter le volume", "plus fort",
            "ارفع الصوت", "صوت أعلى", "زيادة الصوت",
        ],
        "description": "Increase Volume"
    },
    "volume down": {
        "code": "0006",
        "aliases": [
            "volume down", "minus volume", "decrease volume", "quieter",
            "turn down", "lower volume", "less sound", "reduce volume",
            "sound down", "softer",
            "diminuer le volume", "moins fort",
            "اخفض الصوت", "صوت أقل", "تخفيض الصوت",
        ],
        "description": "Decrease Volume"
    },
    "channel up": {
        "code": "0007",
        "aliases": [
            "channel up", "next channel", "channel forward", "next",
            "chaîne suivante", "القناة التالية", "القناة فوق",
        ],
        "description": "Next Channel"
    },
    "channel down": {
        "code": "0008",
        "aliases": [
            "channel down", "previous channel", "channel back", "back",
            "chaîne précédente", "القناة السابقة", "القناة تحت",
        ],
        "description": "Previous Channel"
    },
    "home": {
        "code": "0009",
        "aliases": [
            "home", "home screen", "main menu", "dashboard", "go home",
            "accueil", "écran principal", "الرئيسية", "الشاشة الرئيسية",
        ],
        "description": "Go to Home Screen"
    },
    "input": {
        "code": "0010",
        "aliases": [
            "input", "source", "hdmi", "change input", "change source",
            "entrée", "source d'entrée", "المصدر", "الإدخال",
        ],
        "description": "Switch Input Source"
    },
    "play": {
        "code": "0011",
        "aliases": [
            "play", "resume", "continue", "start playing",
            "jouer", "reprendre", "تشغيل", "استئناف",
        ],
        "description": "Play / Resume"
    },
    "pause": {
        "code": "0012",
        "aliases": [
            "pause", "freeze", "hold", "stop playing", "wait",
            "pauser", "mettre en pause", "إيقاف مؤقت", "توقف",
        ],
        "description": "Pause"
    },
}

# ─────────────────────────────────────────────
#  BUILD ALIAS MAP
# ─────────────────────────────────────────────
def build_alias_map(commands):
    alias_map = {}
    for cmd_name, cmd_data in commands.items():
        for alias in cmd_data["aliases"]:
            alias_map[alias.lower()] = cmd_name
    return alias_map

# ─────────────────────────────────────────────
#  NLP MATCHER
# ─────────────────────────────────────────────
class TVCommandMatcher:
    def __init__(self, commands):
        self.commands = commands
        self.alias_map = build_alias_map(commands)
        self.all_aliases = list(self.alias_map.keys())

        print("[NLP] Loading semantic model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("[NLP] Encoding aliases...")
        self.alias_embeddings = self.model.encode(
            self.all_aliases, convert_to_tensor=True, show_progress_bar=False
        )
        print("[NLP] ✓ Ready!\n")

    def match(self, spoken_text, fuzzy_threshold=70, semantic_threshold=0.55):
        text = spoken_text.lower().strip()
        result = {"input": spoken_text, "command": None, "code": None,
                  "method": None, "confidence": 0.0, "description": None}

        # 1. Exact / substring
        if text in self.alias_map:
            cmd = self.alias_map[text]
            result.update({"command": cmd, "code": self.commands[cmd]["code"],
                           "method": "exact", "confidence": 1.0,
                           "description": self.commands[cmd]["description"]})
            return result

        for alias, cmd in self.alias_map.items():
            if alias in text or text in alias:
                result.update({"command": cmd, "code": self.commands[cmd]["code"],
                               "method": "substring", "confidence": 0.95,
                               "description": self.commands[cmd]["description"]})
                return result

        # 2. Fuzzy
        best_fuzzy, score, _ = process.extractOne(
            text, self.all_aliases, scorer=fuzz.WRatio)
        if score >= fuzzy_threshold:
            cmd = self.alias_map[best_fuzzy]
            result.update({"command": cmd, "code": self.commands[cmd]["code"],
                           "method": f"fuzzy ({score:.0f}%)", "confidence": score / 100,
                           "description": self.commands[cmd]["description"]})
            return result

        # 3. Semantic NLP
        query_emb = self.model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.alias_embeddings)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        if best_score >= semantic_threshold:
            cmd = self.alias_map[self.all_aliases[best_idx]]
            result.update({"command": cmd, "code": self.commands[cmd]["code"],
                           "method": f"semantic ({best_score:.2f})",
                           "confidence": best_score,
                           "description": self.commands[cmd]["description"]})
            return result

        result["method"] = "no match"
        return result

# ─────────────────────────────────────────────
#  PRINT HELPERS
# ─────────────────────────────────────────────
def print_header():
    print("\n" + "═"*50)
    print("   📺  TV VOICE ASSISTANT  🎙️")
    print("═"*50)
    print("  Commands  │  Code")
    print("────────────┼───────")
    for name, data in TV_COMMANDS.items():
        print(f"  {name:<10} │  {data['code']}")
    print("═"*50)
    print("  Say a command — Ctrl+C to quit\n")

def print_result(result):
    if result["command"]:
        print("\n┌─────────────────────────────────┐")
        print(f"│  🎯  Heard   : {result['input']:<17} │")
        print(f"│  ✅  Command : {result['command']:<17} │")
        print(f"│  📟  Code    : {result['code']:<17} │")
        print(f"│  🔍  Method  : {result['method']:<17} │")
        print(f"│  📝  Action  : {result['description']:<17} │")
        print("└─────────────────────────────────┘\n")
    else:
        print(f"\n  ❌  Could not match: \"{result['input']}\"")
        print("     Try rephrasing or speak more clearly.\n")

# ─────────────────────────────────────────────
#  MICROPHONE LISTENER
# ─────────────────────────────────────────────
def listen_and_recognize(recognizer, language):
    """Open mic, listen, return transcribed text or None."""
    with sr.Microphone() as source:
        print("🎙️  Listening... (speak now)")
        # Calibrate noise for 0.5 seconds
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            print("⏱️  Timeout — no speech detected\n")
            return None

    print("⚙️  Processing...")
    try:
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        print("❓  Could not understand — please try again\n")
        return None
    except sr.RequestError as e:
        print(f"🌐  Network error: {e}")
        print("    Make sure you have internet access for Google STT\n")
        return None

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def run(language="en-US"):
    print_header()

    matcher = TVCommandMatcher(TV_COMMANDS)
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    print(f"[Info] Language : {language}")
    print(f"[Info] Mode     : continuous (press Ctrl+C to stop)\n")

    while True:
        try:
            spoken = listen_and_recognize(recognizer, language)

            if spoken is None:
                continue  # go back to listening

            print(f'🗣️  You said: "{spoken}"')
            result = matcher.match(spoken)
            print_result(result)

            # ── Here you can add your actual TV command sender ──
            # if result["code"]:
            #     send_to_tv(result["code"])

        except KeyboardInterrupt:
            print("\n\n[Assistant] Stopped. Goodbye! 👋\n")
            sys.exit(0)

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TV Voice Assistant - Mic Mode")
    parser.add_argument(
        "--lang", default="en-US",
        help="Speech language: en-US | fr-FR | ar-DZ | ar-SA (default: en-US)"
    )
    args = parser.parse_args()
    run(language=args.lang)