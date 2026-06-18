"""
MULTI-DEVICE VOICE ASSISTANT - ENHANCED VERSION
================================================
Real-time voice control for MULTIPLE devices with performance monitoring.

Command format:   <device name> + <command>
  • "tv  <command>"    → controls the TV          → returns "11" + command code
  • "clem <command>"   → controls Air Conditioner → returns "00" + command code

Supported languages: English + Arabic (Algerian Darija)

Features:
  • Device-name activation (TV / Air Conditioner "clem")
  • Per-device command databases & output prefixes
  • Performance metrics (recording time, processing time)
  • Semantic NLP matching (multilingual)
  • Arabic Darija (Algerian dialect) support
  • Real-time monitoring

Usage:
    python tv_voice_assistant_enhanced.py --lang en-US
    python tv_voice_assistant_enhanced.py --lang ar-DZ
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
#  TV COMMANDS DATABASE  (output prefix = "11")
# ─────────────────────────────────────────────
TV_COMMANDS = {
    # Power Controls
    "open": {
        "code": "0001",
        "aliases": [
            "open", "turn on", "power on", "start", "on", "switch on", "wake up",
            "boot", "launch", "activate", "begin", "fire it up", "wake", "go",
            "افتح", "شغّل", "تشغيل", "دوّر", "ولّع", "اضغط", "شمّع",
            "فتح الجهاز", "شغيل التليفزيون", "حل التلفزيون",
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
            "أغلق", "إيقاف", "أوقف", "طفّي", "اطفي", "قفّل", "نام",
            "إطفاء التليفزيون", "قفل الجهاز", "حبس التلفزيون",
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
            "كتم", "صامت", "أسكت", "اسكت الصوت", "سكوت", "قطع الصوت", "حبس الصوت",
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
            "ارفع الصوت", "صوت أعلى", "زيادة الصوت", "أرفع", "أعلى شوي", "زيد الصوت",
        ],
        "category": "Volume",
        "description": "Increase Volume"
    },
    "volume down": {
        "code": "0006",
        "aliases": [
            "volume down", "minus volume", "decrease volume", "quieter", "turn down",
            "lower volume", "less sound", "reduce volume", "sound down", "softer",
            "اخفض الصوت", "صوت أقل", "تخفيض الصوت", "اخفت", "خفيف", "نقّص الصوت",
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
            "القناة التالية", "القناة فوق", "الزر التالي", "زين", "القناة اللي بعد",
        ],
        "category": "Channel",
        "description": "Next Channel"
    },
    "channel down": {
        "code": "0008",
        "aliases": [
            "channel down", "previous channel", "channel back", "back", "reverse",
            "channel minus", "skip back", "go back", "prior",
            "القناة السابقة", "القناة تحت", "الزر السابق", "راجع", "القناة اللي قبل",
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
            "المصدر", "الإدخال", "الموجة", "الكابل", "بدّل المصدر",
        ],
        "category": "Navigation",
        "description": "Switch Input Source"
    },
    "settings": {
        "code": "0003",
        "aliases": [
            "settings", "setting", "options", "preferences", "configure",
            "configuration", "setup", "adjust", "customize",
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
            "let's go", "play now",
            "تشغيل", "استئناف", "ابدأ", "اضغط التشغيل", "شغل الفيديو",
        ],
        "category": "Playback",
        "description": "Play / Resume"
    },
    "pause": {
        "code": "0012",
        "aliases": [
            "pause", "freeze", "hold", "stop playing", "wait",
            "pause it", "hold on", "suspend",
            "إيقاف مؤقت", "توقف", "انتظر", "قف", "جمّد",
        ],
        "category": "Playback",
        "description": "Pause"
    },
    "stop": {
        "code": "0013",
        "aliases": [
            "stop", "cease", "end", "finish", "cut", "exit playback",
            "إيقاف", "توقف نهائي", "انهي", "خرج", "نهاية",
        ],
        "category": "Playback",
        "description": "Stop Playback"
    },
    "rewind": {
        "code": "0014",
        "aliases": [
            "rewind", "go back", "backward", "back up", "previous",
            "ارجع", "للخلف", "رجّع", "اللي فات",
        ],
        "category": "Playback",
        "description": "Rewind"
    },
    "fast forward": {
        "code": "0015",
        "aliases": [
            "fast forward", "skip forward", "skip", "ahead", "forward",
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
            "سجّل", "تسجيل", "ابدأ التسجيل", "بدّا التسجيل",
        ],
        "category": "Recording",
        "description": "Start Recording"
    },
    "screenshot": {
        "code": "0031",
        "aliases": [
            "screenshot", "screen capture", "capture", "snap", "take screenshot",
            "صورة الشاشة", "التقط صورة", "خذ صورة",
        ],
        "category": "Capture",
        "description": "Take Screenshot"
    },
    "guide": {
        "code": "0032",
        "aliases": [
            "guide", "tv guide", "program guide", "what's on", "schedule",
            "الدليل", "دليل البرامج", "البرامج", "الجدول",
        ],
        "category": "Navigation",
        "description": "Open TV Guide"
    },
    "search": {
        "code": "0033",
        "aliases": [
            "search", "find", "look for", "search for", "hunt",
            "ابحث", "البحث", "لقّي", "شوف",
        ],
        "category": "Navigation",
        "description": "Open Search"
    },
}

# ─────────────────────────────────────────────
#  AIR CONDITIONER COMMANDS DATABASE  (output prefix = "00")
# ─────────────────────────────────────────────
AC_COMMANDS = {
    # Power
    "power on": {
        "code": "0001",
        "aliases": [
            "power on", "turn on", "start", "on", "switch on", "open",
            "ولّع الكليما", "شعل الكليما", "خدّم الكليما", "حل الكليما",
            "ولّع", "شعل", "خدّم", "شغّل المكيف", "شغّل",
        ],
        "category": "Power",
        "description": "Turn ON the Air Conditioner"
    },
    "power off": {
        "code": "0002",
        "aliases": [
            "power off", "turn off", "off", "switch off", "shutdown", "stop",
            "طفّي الكليما", "حبس الكليما", "سكّر الكليما", "طفّي", "حبس",
            "سكّر", "أطفئ المكيف", "وقّف الكليما",
        ],
        "category": "Power",
        "description": "Turn OFF the Air Conditioner"
    },

    # Temperature
    "temp up": {
        "code": "0003",
        "aliases": [
            "temperature up", "warmer", "hotter", "increase temperature",
            "raise temperature", "heat up", "more heat",
            "زيد السخانة", "زيد الحرارة", "سخّن شوية", "زيد", "رفع الحرارة",
        ],
        "category": "Temperature",
        "description": "Increase Temperature"
    },
    "temp down": {
        "code": "0004",
        "aliases": [
            "temperature down", "cooler", "colder", "decrease temperature",
            "lower temperature", "cool down", "more cold",
            "نقّص الحرارة", "برّد شوية", "اخفض الحرارة", "نقّص", "خفّض الحرارة",
        ],
        "category": "Temperature",
        "description": "Decrease Temperature"
    },

    # Modes
    "cool mode": {
        "code": "0005",
        "aliases": [
            "cool", "cooling", "cool mode", "cold mode", "air cool",
            "تبريد", "وضع التبريد", "برّد", "كول", "حط تبريد",
        ],
        "category": "Mode",
        "description": "Cooling Mode"
    },
    "heat mode": {
        "code": "0006",
        "aliases": [
            "heat", "heating", "heat mode", "warm mode",
            "تسخين", "وضع التسخين", "سخّن", "حط تسخين", "سخانة",
        ],
        "category": "Mode",
        "description": "Heating Mode"
    },
    "fan mode": {
        "code": "0007",
        "aliases": [
            "fan", "fan mode", "ventilation", "ventilate", "air",
            "مروحة", "فان", "تهوية", "وضع المروحة", "هوا",
        ],
        "category": "Mode",
        "description": "Fan / Ventilation Mode"
    },
    "dry mode": {
        "code": "0008",
        "aliases": [
            "dry", "dry mode", "dehumidify", "dehumidifier",
            "تجفيف", "وضع التجفيف", "نشّف", "إزالة الرطوبة",
        ],
        "category": "Mode",
        "description": "Dry / Dehumidify Mode"
    },
    "auto mode": {
        "code": "0009",
        "aliases": [
            "auto", "auto mode", "automatic", "automatic mode",
            "أوتوماتيك", "تلقائي", "وضع تلقائي", "أوتو",
        ],
        "category": "Mode",
        "description": "Automatic Mode"
    },

    # Fan speed
    "fan up": {
        "code": "0010",
        "aliases": [
            "fan up", "faster fan", "increase fan", "fan speed up", "stronger fan",
            "زيد المروحة", "زيد الهوا", "سرّع المروحة", "قوّي الهوا", "زيد سرعة المروحة",
        ],
        "category": "Fan",
        "description": "Increase Fan Speed"
    },
    "fan down": {
        "code": "0011",
        "aliases": [
            "fan down", "slower fan", "decrease fan", "fan speed down", "weaker fan",
            "نقّص المروحة", "بطّئ المروحة", "نقّص الهوا", "خفّف الهوا", "نقّص سرعة المروحة",
        ],
        "category": "Fan",
        "description": "Decrease Fan Speed"
    },

    # Swing
    "swing on": {
        "code": "0012",
        "aliases": [
            "swing", "swing on", "oscillate", "move air", "auto swing",
            "حرّك الهوا", "سوينغ", "وجّه الهوا", "حرّك", "دوّر الهوا",
        ],
        "category": "Swing",
        "description": "Turn ON Swing"
    },
    "swing off": {
        "code": "0013",
        "aliases": [
            "swing off", "stop swing", "no swing", "fix air",
            "حبس الحركة", "وقّف السوينغ", "ثبّت الهوا", "حبس السوينغ",
        ],
        "category": "Swing",
        "description": "Turn OFF Swing"
    },

    # Extras
    "timer": {
        "code": "0014",
        "aliases": [
            "timer", "set timer", "sleep timer", "schedule",
            "مؤقت", "تايمر", "وقّت", "حط مؤقت",
        ],
        "category": "Timer",
        "description": "Set Timer"
    },
    "eco mode": {
        "code": "0015",
        "aliases": [
            "eco", "eco mode", "sleep mode", "energy saving", "save energy",
            "وضع النوم", "اقتصادي", "إيكو", "وضع توفير الطاقة", "وفّر الطاقة",
        ],
        "category": "Mode",
        "description": "Eco / Sleep Mode"
    },
}

# ─────────────────────────────────────────────
#  DEVICE REGISTRY
# ─────────────────────────────────────────────
#  Each device: human name, output prefix, trigger words, command database.
#  Output sent to hardware = prefix + command code  (e.g. TV open → "11" + "0001").
DEVICES = {
    "tv": {
        "name": "TV",
        "prefix": "11",
        "triggers": [
            "tv", "tee vee", "t v", "tiv", "the tv", "television", "telly",
            "تي في", "تيفي", "تلفاز", "التلفزيون", "تلفزيون", "تيلي",
        ],
        "commands": TV_COMMANDS,
    },
    "ac": {
        "name": "Air Conditioner",
        "prefix": "00",
        "triggers": [
            "clem", "clim", "klim", "klima", "kleem", "la clim", "the ac",
            "ac", "air conditioner", "air conditioning", "air con",
            "كليم", "كليما", "الكليما", "المكيف", "مكيف", "تكييف", "المكيّف",
        ],
        "commands": AC_COMMANDS,
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
#  DEVICE COMMAND MATCHER  (one per device, shares the NLP model)
# ─────────────────────────────────────────────
class DeviceCommandMatcher:
    def __init__(self, device_key, device, model, show_progress=False):
        self.device_key = device_key
        self.name = device["name"]
        self.prefix = device["prefix"]
        self.commands = device["commands"]
        self.model = model
        self.alias_map = build_alias_map(self.commands)
        self.all_aliases = list(self.alias_map.keys())

        print(f"[NLP] Encoding aliases for {self.name}...")
        self.alias_embeddings = self.model.encode(
            self.all_aliases,
            convert_to_tensor=True,
            show_progress_bar=show_progress
        )

    def _build_result(self, spoken_text, cmd, method, confidence, match_start):
        cmd_data = self.commands[cmd]
        return {
            "input": spoken_text,
            "device": self.name,
            "device_key": self.device_key,
            "command": cmd,
            "code": cmd_data["code"],
            "full_code": self.prefix + cmd_data["code"],
            "method": method,
            "confidence": confidence,
            "description": cmd_data["description"],
            "category": cmd_data["category"],
            "processing_time": time.time() - match_start,
        }

    def match(self, spoken_text, fuzzy_threshold=70, semantic_threshold=0.55):
        match_start = time.time()
        text = spoken_text.lower().strip()

        # 1. Exact match
        if text in self.alias_map:
            return self._build_result(spoken_text, self.alias_map[text], "exact", 1.0, match_start)

        # 2. Substring match
        for alias, cmd in self.alias_map.items():
            if alias in text or text in alias:
                return self._build_result(spoken_text, cmd, "substring", 0.95, match_start)

        # 3. Fuzzy matching
        best_fuzzy, score, _ = process.extractOne(
            text, self.all_aliases, scorer=fuzz.WRatio
        )
        if score >= fuzzy_threshold:
            return self._build_result(
                spoken_text, self.alias_map[best_fuzzy],
                f"fuzzy ({score:.0f}%)", score / 100, match_start
            )

        # 4. Semantic NLP
        query_emb = self.model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.alias_embeddings)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= semantic_threshold:
            return self._build_result(
                spoken_text, self.alias_map[self.all_aliases[best_idx]],
                f"semantic ({best_score:.2f})", best_score, match_start
            )

        # No match
        return {
            "input": spoken_text,
            "device": self.name,
            "device_key": self.device_key,
            "command": None,
            "code": None,
            "full_code": None,
            "method": "no match",
            "confidence": 0.0,
            "description": None,
            "category": None,
            "processing_time": time.time() - match_start,
        }

# ─────────────────────────────────────────────
#  MULTI-DEVICE ASSISTANT  (device detection + routing)
# ─────────────────────────────────────────────
class MultiDeviceAssistant:
    def __init__(self, devices, show_progress=False):
        print("[NLP] Loading semantic model...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        self.devices = devices
        self.matchers = {
            key: DeviceCommandMatcher(key, dev, self.model, show_progress)
            for key, dev in devices.items()
        }
        print("[NLP] ✓ All devices ready!\n")

    def detect_device(self, spoken_text, fuzzy_threshold=80):
        """Identify which device the command targets.

        Returns (device_key, remainder) where remainder is the command text
        after the device name, or (None, None) if no device recognized.
        """
        text = spoken_text.lower().strip()
        if not text:
            return None, None

        # 1. Exact prefix match — longest triggers first (e.g. "air conditioner" before "ac")
        candidates = []
        for key, dev in self.devices.items():
            for trigger in dev["triggers"]:
                candidates.append((trigger.lower(), key))
        candidates.sort(key=lambda x: len(x[0]), reverse=True)

        for trigger, key in candidates:
            if text == trigger:
                return key, None
            if text.startswith(trigger + " "):
                return key, text[len(trigger):].strip()

        # 2. Fuzzy match on the first word (handles STT noise / dialect variants)
        first_word = text.split()[0]
        best_key, best_score = None, 0
        for key, dev in self.devices.items():
            for trigger in dev["triggers"]:
                score = fuzz.ratio(first_word, trigger.lower())
                if score > best_score:
                    best_score, best_key = score, key
        if best_score >= fuzzy_threshold:
            remainder = " ".join(text.split()[1:])
            return best_key, (remainder if remainder else None)

        return None, None

    def process(self, spoken_text):
        """Detect device then match the command. Returns (device_key, result)."""
        device_key, remainder = self.detect_device(spoken_text)
        if device_key is None:
            return None, None
        if not remainder:
            return device_key, None
        result = self.matchers[device_key].match(remainder)
        return device_key, result

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
    print("   🎛️   MULTI-DEVICE VOICE ASSISTANT  🎙️")
    print("═"*60)
    print("  Say: '<device name>' followed by a command")
    print("  📺  TV  → 'tv open', 'tv volume up', 'tv netflix'")
    print("  ❄️   AC  → 'clem power on', 'clem cool', 'clem temp up'")
    print("─"*60)
    for key, dev in DEVICES.items():
        print(f"  • {dev['name']:<18} prefix '{dev['prefix']}' · "
              f"{len(dev['commands'])} commands")
    print("═"*60)
    print("  Press Ctrl+C to quit\n")

def print_result(device_name, result, show_performance=False):
    if result and result["command"]:
        print("\n┌──────────────────────────────────────────┐")
        print(f"│  🎯  Heard     : {result['input']:<28} │")
        print(f"│  🎛️   Device    : {device_name:<28} │")
        print(f"│  ✅  Command   : {result['command']:<28} │")
        print(f"│  📟  Code      : {result['code']:<28} │")
        print(f"│  📤  Output    : {result['full_code']:<28} │")
        print(f"│  🔍  Method    : {result['method']:<28} │")
        print(f"│  📊  Confidence: {result['confidence']*100:.1f}% {'':<20} │")
        print(f"│  📝  Action    : {result['description']:<28} │")
        if show_performance:
            print(f"│  ⏱️   Processing: {result['processing_time']*1000:.2f}ms {'':<16} │")
        print("└──────────────────────────────────────────┘\n")
    elif result:
        print(f"\n  ❌  [{device_name}] Could not match: \"{result['input']}\"")
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

    assistant = MultiDeviceAssistant(DEVICES, show_progress=False)
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
    print(f"[Info] Devices        : {', '.join(d['name'] for d in DEVICES.values())}")
    print(f"[Info] Mode           : Continuous (press Ctrl+C to stop)\n")

    while True:
        try:
            stats_start = time.time()

            spoken = listen_and_recognize(recognizer, language, performance_tracker)

            if spoken is None:
                continue

            print(f'🗣️  You said: "{spoken}"')

            # Identify the target device from the spoken text
            device_key, remainder = assistant.detect_device(spoken)

            if device_key is None:
                print("⚠️  No device name detected. Start with a device name "
                      "(e.g. 'tv ...' or 'clem ...').\n")
                continue

            device = DEVICES[device_key]
            print(f"✅ Device detected: {device['name']} (prefix '{device['prefix']}')")

            if not remainder:
                print(f"⏱️  Waiting for a command after '{device['name']}'...\n")
                continue

            print(f"📝 Processing command: \"{remainder}\"")
            match_start = time.time()
            result = assistant.matchers[device_key].match(remainder)
            match_time = time.time() - match_start
            if performance_tracker:
                performance_tracker.record_matching_time(match_time)

            total_time = time.time() - stats_start
            if performance_tracker:
                performance_tracker.record_processing_time(total_time)
                performance_tracker.command_attempted(result["command"] is not None)

            print_result(device["name"], result, show_performance=show_performance)

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
        description="Multi-Device Voice Assistant (TV + Air Conditioner)"
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
