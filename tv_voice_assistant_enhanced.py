"""
TV VOICE ASSISTANT - ENHANCED VERSION
=====================================
Real-time voice recording with performance monitoring
"TV" trigger word detection + command processing
Multi-language support (English, French, Arabic Algerian Dialect)

Features:
  • Voice activation ("TV" trigger)
  • Performance metrics (recording time, processing time)
  • Semantic NLP matching
  • Arabic Dariha (Algerian dialect) support
  • Extensive TV command database
  • Real-time monitoring

Usage:
    python tv_voice_assistant_enhanced.py --lang en-US
    python tv_voice_assistant_enhanced.py --lang ar-DZ
    python tv_voice_assistant_enhanced.py --lang fr-FR
    python tv_voice_assistant_enhanced.py --performance
"""

import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
import json

# ─────────────────────────────────────────────
#  PERFORMANCE TRACKER
# ─────────────────────────────────────────────
class PerformanceTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.session_start = time.time()
        self.command_count = 0
        self.matched_count = 0
        
    def record_audio_time(self, duration):
        self.metrics["audio_recording_time"].append(duration)
    
    def record_processing_time(self, duration):
        self.metrics["processing_time"].append(duration)
    
    def record_stt_time(self, duration):
        self.metrics["speech_to_text_time"].append(duration)
    
    def record_matching_time(self, duration):
        self.metrics["matching_time"].append(duration)
    
    def command_attempted(self, matched=True):
        self.command_count += 1
        if matched:
            self.matched_count += 1
    
    def get_stats(self):
        total_time = time.time() - self.session_start
        stats = {
            "session_duration": total_time,
            "total_commands": self.command_count,
            "matched_commands": self.matched_count,
            "accuracy": (self.matched_count / max(1, self.command_count)) * 100,
            "avg_recording_time": self._avg("audio_recording_time"),
            "avg_stt_time": self._avg("speech_to_text_time"),
            "avg_matching_time": self._avg("matching_time"),
            "avg_total_time": self._avg("processing_time"),
        }
        return stats
    
    def _avg(self, key):
        values = self.metrics[key]
        return sum(values) / len(values) if values else 0

# ─────────────────────────────────────────────
#  TV COMMANDS DATABASE
# ─────────────────────────────────────────────
TV_COMMANDS = {
    # Power Controls
    "open": {
        "code": "0001",
        "aliases": [
            "open", "turn on", "power on", "start", "on", "switch on", "wake up",
            "boot", "launch", "activate", "begin", "fire it up", "wake", "go",
            "allumer", "ouvrir", "démarrer", "mettre en marche", "brancher",
            "افتح", "شغّل", "تشغيل", "دوّر", "ولّع", "اضغط", "شمّع",
            "فتح الجهاز", "شغيل التليفزيون",
        ],
        "category": "Power",
        "description": "Turn ON the TV"
    },
    "close": {
        "code": "0002",
        "aliases": [
            "close", "turn off", "power off", "shutdown", "off", "switch off",
            "stop", "quit", "exit", "sleep", "standby", "turn down", "mute all",
            "clothes", "cloths", "clouse", "cloze",
            "éteindre", "fermer", "arrêter", "désactiver", "sommeil", "veille",
            "أغلق", "إيقاف", "أوقف", "طفّي", "اطفي", "قفّل", "نام",
            "إطفاء التليفزيون", "قفل الجهاز",
        ],
        "category": "Power",
        "description": "Turn OFF the TV"
    },
    
    # Volume Controls
    "mute": {
        "code": "0004",
        "aliases": [
            "mute", "silent", "silence", "quiet", "no sound", "no audio",
            "shh", "muting", "silence please", "be quiet", "sound off",
            "mute", "mute", "mute",
            "sourdine", "couper le son", "silencieux", "sans son", "tranquille",
            "كتم", "صامت", "أسكت", "اسكت الصوت", "سكوت", "قطع الصوت",
        ],
        "category": "Volume",
        "description": "Mute / Unmute"
    },
    "volume up": {
        "code": "0005",
        "aliases": [
            "volume up", "add volume", "increase volume", "louder", "turn up",
            "raise volume", "more sound", "higher volume", "sound up", "amplify",
            "boost volume", "stronger", "louder please", "crank it up",
            "volume up", "volume up", "volume up", "volume up",
            "augmenter le volume", "plus fort", "augmenter", "monter le son",
            "ارفع الصوت", "صوت أعلى", "زيادة الصوت", "أرفع", "أعلى شوي",
        ],
        "category": "Volume",
        "description": "Increase Volume"
    },
    "volume down": {
        "code": "0006",
        "aliases": [
            "volume down", "minus volume", "decrease volume", "quieter", "turn down",
            "lower volume", "less sound", "reduce volume", "sound down", "softer",
            "diminuer", "less noise", "quiet it down",
            "diminuer le volume", "moins fort", "baisser le son", "réduire",
            "اخفض الصوت", "صوت أقل", "تخفيض الصوت", "اخفت", "خفيف",
        ],
        "category": "Volume",
        "description": "Decrease Volume"
    },
    
    # Channel Controls
    "channel up": {
        "code": "0007",
        "aliases": [
            "channel up", "next channel", "channel forward", "next", "advance",
            "channel plus", "skip forward", "go next",
            "channel up", "channel up", "channel up",
            "chaîne suivante", "chaîne suivante", "forward", "avancer",
            "القناة التالية", "القناة فوق", "الزر التالي", "زين",
        ],
        "category": "Channel",
        "description": "Next Channel"
    },
    "channel down": {
        "code": "0008",
        "aliases": [
            "channel down", "previous channel", "channel back", "back", "reverse",
            "channel minus", "skip back", "go back", "prior",
            "chaîne précédente", "chaîne précédente", "backward", "reculer",
            "القناة السابقة", "القناة تحت", "الزر السابق", "راجع",
        ],
        "category": "Channel",
        "description": "Previous Channel"
    },
    
    # Navigation
    "home": {
        "code": "0009",
        "aliases": [
            "home", "home screen", "main menu", "dashboard", "go home",
            "menu", "main", "start page", "homepage",
            "accueil", "écran principal", "menu principal", "page d'accueil",
            "الرئيسية", "الشاشة الرئيسية", "القائمة الرئيسية", "بيت",
        ],
        "category": "Navigation",
        "description": "Go to Home Screen"
    },
    "input": {
        "code": "0010",
        "aliases": [
            "input", "source", "hdmi", "change input", "change source",
            "switch input", "external", "input source",
            "entrée", "source d'entrée", "changer d'entrée", "externe",
            "المصدر", "الإدخال", "الموجة", "الكابل",
        ],
        "category": "Navigation",
        "description": "Switch Input Source"
    },
    "settings": {
        "code": "0003",
        "aliases": [
            "settings", "setting", "options", "preferences", "configure",
            "configuration", "setup", "menu", "adjust", "customize",
            "settings", "settings", "settings",
            "paramètres", "réglages", "options", "préférences", "configurer",
            "الإعدادات", "ضبط", "إعدادات", "التوازنات", "التفضيلات",
        ],
        "category": "Navigation",
        "description": "Open Settings Menu"
    },
    
    # Playback Controls
    "play": {
        "code": "0011",
        "aliases": [
            "play", "resume", "continue", "start playing", "press play",
            "begin", "go", "let's go", "play now",
            "jouer", "reprendre", "continuer", "démarrer", "lancer",
            "تشغيل", "استئناف", "ابدأ", "اضغط التشغيل", "شغل الفيديو",
        ],
        "category": "Playback",
        "description": "Play / Resume"
    },
    "pause": {
        "code": "0012",
        "aliases": [
            "pause", "freeze", "hold", "stop playing", "wait", "stop",
            "pause it", "hold on", "suspend",
            "pauser", "mettre en pause", "arrêter", "suspendre", "attendre",
            "إيقاف مؤقت", "توقف", "انتظر", "قف", "جمّد",
        ],
        "category": "Playback",
        "description": "Pause"
    },
    "stop": {
        "code": "0013",
        "aliases": [
            "stop", "cease", "end", "finish", "cut", "exit playback",
            "arrêter", "cesser", "terminer", "finir", "quitter",
            "إيقاف", "توقف نهائي", "انهي", "خرج", "نهاية",
        ],
        "category": "Playback",
        "description": "Stop Playback"
    },
    "rewind": {
        "code": "0014",
        "aliases": [
            "rewind", "go back", "backward", "back up", "previous",
            "rembobiner", "aller arrière", "retourner", "précédent",
            "ارجع", "للخلف", "رجّع", "اللي فات",
        ],
        "category": "Playback",
        "description": "Rewind"
    },
    "fast forward": {
        "code": "0015",
        "aliases": [
            "fast forward", "skip forward", "skip", "ahead", "forward",
            "avance rapide", "avancer", "sauter", "avance", "suivant",
            "ابعد للأمام", "تخطي", "قدّم", "روح لقدام",
        ],
        "category": "Playback",
        "description": "Fast Forward"
    },
    
    # App Controls
    "netflix": {
        "code": "0020",
        "aliases": [
            "netflix", "netflix app", "open netflix", "netflix please",
            "netflix show", "watch netflix",
            "netflix", "ouvrir netflix", "app netflix", "netflix s'il vous plaît",
            "نيتفليكس", "افتح نيتفليكس", "تطبيق نيتفليكس",
        ],
        "category": "App",
        "description": "Open Netflix"
    },
    "youtube": {
        "code": "0021",
        "aliases": [
            "youtube", "youtube app", "open youtube", "youtube please",
            "youtube video", "watch youtube", "you tube",
            "youtube", "ouvrir youtube", "app youtube", "youtube s'il vous plaît",
            "يوتيوب", "افتح يوتيوب", "تطبيق يوتيوب",
        ],
        "category": "App",
        "description": "Open YouTube"
    },
    "spotify": {
        "code": "0022",
        "aliases": [
            "spotify", "spotify app", "open spotify", "spotify please",
            "listen spotify", "music spotify",
            "spotify", "ouvrir spotify", "app spotify", "spotify s'il vous plaît",
            "سبوتيفاي", "افتح سبوتيفاي", "تطبيق الموسيقى",
        ],
        "category": "App",
        "description": "Open Spotify"
    },
    
    # Additional Controls
    "recording": {
        "code": "0030",
        "aliases": [
            "record", "recording", "record this", "start recording", "rec",
            "enregistrer", "enregistrement", "start rec", "recording on",
            "سجّل", "تسجيل", "ابدأ التسجيل", "بدّا التسجيل",
        ],
        "category": "Recording",
        "description": "Start Recording"
    },
    "screenshot": {
        "code": "0031",
        "aliases": [
            "screenshot", "screen capture", "capture", "snap", "take screenshot",
            "capture d'écran", "prendre screenshot", "snap shot",
            "صورة الشاشة", "التقط صورة", "خذ صورة",
        ],
        "category": "Capture",
        "description": "Take Screenshot"
    },
    "guide": {
        "code": "0032",
        "aliases": [
            "guide", "tv guide", "program guide", "what's on", "schedule",
            "guide tv", "guide des programmes", "horaire", "ce qui passe",
            "الدليل", "دليل البرامج", "البرامج", "الجدول",
        ],
        "category": "Navigation",
        "description": "Open TV Guide"
    },
    "search": {
        "code": "0033",
        "aliases": [
            "search", "find", "look for", "search for", "hunt",
            "chercher", "rechercher", "trouver", "find",
            "ابحث", "البحث", "لقّي", "شوف",
        ],
        "category": "Navigation",
        "description": "Open Search"
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
#  TV COMMAND MATCHER
# ─────────────────────────────────────────────
class TVCommandMatcher:
    def __init__(self, commands, show_progress=False):
        self.commands = commands
        self.alias_map = build_alias_map(commands)
        self.all_aliases = list(self.alias_map.keys())
        self.show_progress = show_progress
        
        print("[NLP] Loading semantic model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        print("[NLP] Encoding aliases...")
        self.alias_embeddings = self.model.encode(
            self.all_aliases, 
            convert_to_tensor=True, 
            show_progress_bar=show_progress
        )
        print("[NLP] ✓ Model ready!\n")

    def match(self, spoken_text, fuzzy_threshold=70, semantic_threshold=0.55):
        match_start = time.time()
        text = spoken_text.lower().strip()
        
        result = {
            "input": spoken_text,
            "command": None,
            "code": None,
            "method": None,
            "confidence": 0.0,
            "description": None,
            "category": None,
            "processing_time": 0
        }

        # 1. Exact match
        if text in self.alias_map:
            cmd = self.alias_map[text]
            cmd_data = self.commands[cmd]
            result.update({
                "command": cmd,
                "code": cmd_data["code"],
                "method": "exact",
                "confidence": 1.0,
                "description": cmd_data["description"],
                "category": cmd_data["category"]
            })
            result["processing_time"] = time.time() - match_start
            return result

        # 2. Substring match
        for alias, cmd in self.alias_map.items():
            if alias in text or text in alias:
                cmd_data = self.commands[cmd]
                result.update({
                    "command": cmd,
                    "code": cmd_data["code"],
                    "method": "substring",
                    "confidence": 0.95,
                    "description": cmd_data["description"],
                    "category": cmd_data["category"]
                })
                result["processing_time"] = time.time() - match_start
                return result

        # 3. Fuzzy matching
        best_fuzzy, score, _ = process.extractOne(
            text, self.all_aliases, scorer=fuzz.WRatio
        )
        if score >= fuzzy_threshold:
            cmd = self.alias_map[best_fuzzy]
            cmd_data = self.commands[cmd]
            result.update({
                "command": cmd,
                "code": cmd_data["code"],
                "method": f"fuzzy ({score:.0f}%)",
                "confidence": score / 100,
                "description": cmd_data["description"],
                "category": cmd_data["category"]
            })
            result["processing_time"] = time.time() - match_start
            return result

        # 4. Semantic NLP
        query_emb = self.model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.alias_embeddings)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        if best_score >= semantic_threshold:
            cmd = self.alias_map[self.all_aliases[best_idx]]
            cmd_data = self.commands[cmd]
            result.update({
                "command": cmd,
                "code": cmd_data["code"],
                "method": f"semantic ({best_score:.2f})",
                "confidence": best_score,
                "description": cmd_data["description"],
                "category": cmd_data["category"]
            })
            result["processing_time"] = time.time() - match_start
            return result

        result["method"] = "no match"
        result["processing_time"] = time.time() - match_start
        return result

    def detect_trigger(self, spoken_text):
        """Check if text starts with 'TV' trigger word"""
        text = spoken_text.lower().strip()
        # Check for TV trigger in various forms
        tv_triggers = ["tv", "tee vee", "t v", "tiv", "tv "]
        
        for trigger in tv_triggers:
            if text.startswith(trigger):
                # Extract command part after TV
                remainder = text[len(trigger):].strip()
                return True, remainder if remainder else None
        
        # Also check via fuzzy matching for Arabic, French variants
        if len(text) > 2:
            first_word = text.split()[0]
            tv_variants = ["tv", "tiv", "تي في", "تيفي", "تف"]
            for variant in tv_variants:
                if fuzz.ratio(first_word, variant) > 85:
                    remainder = " ".join(text.split()[1:])
                    return True, remainder if remainder else None
        
        return False, None

# ─────────────────────────────────────────────
#  MICROPHONE LISTENER
# ─────────────────────────────────────────────
def listen_and_recognize(recognizer, language, performance_tracker=None):
    """Listen from microphone and return transcribed text"""
    with sr.Microphone() as source:
        print("🎙️  Listening... (speak now)")
        
        recording_start = time.time()
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            print("⏱️  Timeout — no speech detected\n")
            return None
        
        recording_time = time.time() - recording_start
        if performance_tracker:
            performance_tracker.record_audio_time(recording_time)

        print("⚙️  Processing...")
        stt_start = time.time()
        
        try:
            text = recognizer.recognize_google(audio, language=language)
            stt_time = time.time() - stt_start
            if performance_tracker:
                performance_tracker.record_stt_time(stt_time)
            return text
        except sr.UnknownValueError:
            print("❓  Could not understand — please try again\n")
            return None
        except sr.RequestError as e:
            print(f"🌐  Network error: {e}")
            print("    Make sure you have internet access for Google STT\n")
            return None

# ─────────────────────────────────────────────
#  DISPLAY FUNCTIONS
# ─────────────────────────────────────────────
def print_header():
    print("\n" + "═"*60)
    print("   📺  TV VOICE ASSISTANT (ENHANCED)  🎙️")
    print("═"*60)
    print("  Say: 'TV' followed by a command")
    print("  Examples: 'TV open', 'TV volume up', 'TV netflix'")
    print("─"*60)
    print(f"  Total Commands Available: {len(TV_COMMANDS)}")
    print("═"*60)
    print("  Press Ctrl+C to quit\n")

def print_result(result, show_performance=False):
    if result["command"]:
        print("\n┌──────────────────────────────────────────┐")
        print(f"│  🎯  Heard    : {result['input']:<29} │")
        print(f"│  ✅  Command  : {result['command']:<29} │")
        print(f"│  📟  Code     : {result['code']:<29} │")
        print(f"│  🔍  Method   : {result['method']:<29} │")
        print(f"│  📊  Confidence: {result['confidence']*100:.1f}% {'':<20} │")
        print(f"│  📝  Action   : {result['description']:<29} │")
        if show_performance:
            print(f"│  ⏱️  Processing: {result['processing_time']*1000:.2f}ms {'':<17} │")
        print("└──────────────────────────────────────────┘\n")
    else:
        print(f"\n  ❌  Could not match: \"{result['input']}\"")
        if show_performance:
            print(f"  ⏱️  Processing time: {result['processing_time']*1000:.2f}ms")
        print("     Try rephrasing or speak more clearly.\n")

def print_performance_stats(performance_tracker):
    stats = performance_tracker.get_stats()
    print("\n" + "═"*60)
    print("   📊  PERFORMANCE STATISTICS")
    print("═"*60)
    print(f"  Session Duration      : {stats['session_duration']:.2f} seconds")
    print(f"  Total Commands Spoken : {stats['total_commands']}")
    print(f"  Matched Commands      : {stats['matched_commands']}")
    print(f"  Accuracy Rate         : {stats['accuracy']:.1f}%")
    print("─"*60)
    print(f"  Avg Audio Recording   : {stats['avg_recording_time']*1000:.2f} ms")
    print(f"  Avg Speech-to-Text    : {stats['avg_stt_time']*1000:.2f} ms")
    print(f"  Avg Matching Time     : {stats['avg_matching_time']*1000:.2f} ms")
    print(f"  Avg Total Processing  : {stats['avg_total_time']*1000:.2f} ms")
    print("═"*60 + "\n")
    
    return stats

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def run(language="en-US", show_performance=False):
    print_header()

    matcher = TVCommandMatcher(TV_COMMANDS, show_progress=False)
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    performance_tracker = PerformanceTracker() if show_performance else None
    
    lang_names = {
        "en-US": "English (US)",
        "fr-FR": "French",
        "ar-DZ": "Arabic (Algerian Dialect)"
    }
    
    print(f"[Info] Language       : {lang_names.get(language, language)}")
    print(f"[Info] Performance    : {'ON' if show_performance else 'OFF'}")
    print(f"[Info] Mode           : Continuous (press Ctrl+C to stop)\n")

    triggered = False
    
    while True:
        try:
            stats_start = time.time()
            
            spoken = listen_and_recognize(recognizer, language, performance_tracker)

            if spoken is None:
                continue

            print(f'🗣️  You said: "{spoken}"')
            
            # Check for TV trigger word
            is_triggered, remainder = matcher.detect_trigger(spoken)
            
            if is_triggered:
                print("✅ TV trigger detected!")
                
                # If there's a command after TV, process it
                if remainder:
                    print(f"📝 Processing command: \"{remainder}\"")
                    match_start = time.time()
                    result = matcher.match(remainder)
                    match_time = time.time() - match_start
                    if performance_tracker:
                        performance_tracker.record_matching_time(match_time)
                    triggered = True
                else:
                    print("⏱️  Waiting for command after 'TV'...\n")
                    triggered = True
                    continue
            else:
                # Try to match even without TV trigger (backwards compatibility)
                print("⚠️  No 'TV' trigger detected. Attempting direct command match...\n")
                match_start = time.time()
                result = matcher.match(spoken)
                match_time = time.time() - match_start
                if performance_tracker:
                    performance_tracker.record_matching_time(match_time)
                triggered = False

            total_time = time.time() - stats_start
            if performance_tracker:
                performance_tracker.record_processing_time(total_time)
                performance_tracker.command_attempted(result["command"] is not None)
            
            print_result(result, show_performance=show_performance)

        except KeyboardInterrupt:
            print("\n\n[Assistant] Session ended. Goodbye! 👋\n")
            
            if performance_tracker:
                print_performance_stats(performance_tracker)
            
            sys.exit(0)

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TV Voice Assistant - Enhanced with Performance Monitoring"
    )
    parser.add_argument(
        "--lang", default="en-US",
        choices=["en-US", "fr-FR", "ar-DZ"],
        help="Speech language (default: en-US)"
    )
    parser.add_argument(
        "--performance", action="store_true",
        help="Enable performance monitoring and statistics"
    )
    
    args = parser.parse_args()
    run(language=args.lang, show_performance=args.performance)
