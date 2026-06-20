"""
MULTI-DEVICE VOICE ASSISTANT - ENHANCED VERSION
================================================
Controls TV (LG) and Air Conditioner (Midea) by voice.
Sends raw hardware command over USB serial (e.g. ac_25, tv_v+).

Command format:  <device name> + <command>
  • "tv  <command>"   → sends  tv_v+   via serial
  • "clem <command>"  → sends  ac_25   via serial

Temperature shortcut:  "clem 25" → ac_25

Languages: English · French · Arabic Algerian Darija

Usage:
    python tv_voice_assistant_enhanced.py --port /dev/cu.usbserial-XXX
    python tv_voice_assistant_enhanced.py --port /dev/cu.usbserial-XXX --baud 115200
    python tv_voice_assistant_enhanced.py --port /dev/cu.usbserial-XXX --lang ar-DZ
    python tv_voice_assistant_enhanced.py  (no serial — prints to screen only)
"""

import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
import argparse
import sys
import time
from collections import defaultdict

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ─────────────────────────────────────────────
#  PERFORMANCE TRACKER
# ─────────────────────────────────────────────
class PerformanceTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.session_start = time.time()
        self.command_count = 0
        self.matched_count = 0

    def record(self, key, duration):
        self.metrics[key].append(duration)

    def record_audio_time(self, d):       self.record("audio", d)
    def record_stt_time(self, d):         self.record("stt", d)
    def record_matching_time(self, d):    self.record("match", d)
    def record_processing_time(self, d):  self.record("total", d)

    def command_attempted(self, matched=True):
        self.command_count += 1
        if matched:
            self.matched_count += 1

    def _avg(self, key):
        v = self.metrics[key]
        return sum(v) / len(v) if v else 0

    def get_stats(self):
        return {
            "session_duration": time.time() - self.session_start,
            "total_commands":   self.command_count,
            "matched_commands": self.matched_count,
            "accuracy":         self.matched_count / max(1, self.command_count) * 100,
            "avg_audio":        self._avg("audio"),
            "avg_stt":          self._avg("stt"),
            "avg_match":        self._avg("match"),
            "avg_total":        self._avg("total"),
        }

# ─────────────────────────────────────────────
#  TV COMMANDS  (hardware codes = tv_*)
#  Output sent to device = "11" + code
# ─────────────────────────────────────────────
TV_COMMANDS = {

    # ── POWER ──────────────────────────────────────────────────────
    "power": {
        "code": "tv_p",
        "aliases": [
            "power", "toggle", "tv power", "toggle power",
            "الطاقة", "تبديل", "زر الطاقة",
        ],
        "category": "Power", "description": "Power Toggle"
    },
    "power on": {
        "code": "tv_pon",
        "aliases": [
            "power on", "turn on", "open", "start", "on", "switch on", "wake up",
            "ولّع التيفي", "شعل", "حل", "افتح", "شغّل",
        ],
        "category": "Power", "description": "Power ON"
    },
    "power off": {
        "code": "tv_poff",
        "aliases": [
            "power off", "turn off", "close", "off", "switch off", "sleep", "shutdown",
            "clothes", "cloths",
            "طفّي التيفي", "سكّر", "قفّل", "نام", "اطفي",
        ],
        "category": "Power", "description": "Power OFF"
    },

    # ── VOLUME ─────────────────────────────────────────────────────
    "volume up": {
        "code": "tv_v+",
        "aliases": [
            "volume up", "louder", "turn up", "raise volume", "sound up",
            "increase volume", "amplify", "more sound",
            "ارفع الصوت", "زيد الصوت", "صوت أعلى", "قوّي الصوت",
            "augmenter le volume", "plus fort",
        ],
        "category": "Volume", "description": "Volume Up"
    },
    "volume down": {
        "code": "tv_v-",
        "aliases": [
            "volume down", "quieter", "turn down", "lower volume", "sound down",
            "decrease volume", "softer", "less sound",
            "اخفض الصوت", "نقّص الصوت", "صوت أقل", "خفّت الصوت",
            "diminuer le volume", "moins fort",
        ],
        "category": "Volume", "description": "Volume Down"
    },
    "mute": {
        "code": "tv_mute",
        "aliases": [
            "mute", "silent", "quiet", "no sound", "shh", "sound off",
            "كتم", "سكوت", "اسكت", "حبس الصوت", "بلا صوت",
            "sourdine", "couper le son",
        ],
        "category": "Volume", "description": "Mute / Unmute"
    },

    # ── CHANNEL ────────────────────────────────────────────────────
    "channel up": {
        "code": "tv_c+",
        "aliases": [
            "channel up", "next channel", "channel forward", "channel plus",
            "القناة التالية", "القناة فوق", "زين", "القناة اللي بعد",
            "chaine suivante", "chaine plus",
        ],
        "category": "Channel", "description": "Channel Up"
    },
    "channel down": {
        "code": "tv_c-",
        "aliases": [
            "channel down", "previous channel", "channel back", "channel minus",
            "القناة السابقة", "القناة تحت", "راجع", "القناة اللي قبل",
            "chaine precedente", "chaine moins",
        ],
        "category": "Channel", "description": "Channel Down"
    },

    # ── NAVIGATION ─────────────────────────────────────────────────
    "up": {
        "code": "tv_up",
        "aliases": ["up", "arrow up", "go up", "فوق", "لفوق", "haut"],
        "category": "Navigation", "description": "Arrow Up"
    },
    "down": {
        "code": "tv_down",
        "aliases": ["down", "arrow down", "go down", "تحت", "لتحت", "bas"],
        "category": "Navigation", "description": "Arrow Down"
    },
    "left": {
        "code": "tv_left",
        "aliases": ["left", "arrow left", "go left", "يسار", "ليسار", "gauche"],
        "category": "Navigation", "description": "Arrow Left"
    },
    "right": {
        "code": "tv_right",
        "aliases": ["right", "arrow right", "go right", "يمين", "ليمين", "droite"],
        "category": "Navigation", "description": "Arrow Right"
    },
    "ok": {
        "code": "tv_ok",
        "aliases": [
            "ok", "okay", "confirm", "select", "enter", "press ok",
            "موافق", "تأكيد", "اختار", "valider",
        ],
        "category": "Navigation", "description": "OK / Select"
    },
    "back": {
        "code": "tv_back",
        "aliases": [
            "back", "go back", "return",
            "رجع", "ارجع", "رجوع", "retour",
        ],
        "category": "Navigation", "description": "Back"
    },
    "exit": {
        "code": "tv_exit",
        "aliases": [
            "exit", "close menu", "quit", "leave",
            "خرج", "اخرج", "quitter",
        ],
        "category": "Navigation", "description": "Exit"
    },
    "home": {
        "code": "tv_home",
        "aliases": [
            "home", "home screen", "main menu", "go home",
            "الرئيسية", "القائمة الرئيسية", "بيت",
            "accueil", "menu principal",
        ],
        "category": "Navigation", "description": "Home Screen"
    },
    "recent": {
        "code": "tv_recent",
        "aliases": [
            "recent", "recent apps", "last apps", "multitask",
            "التطبيقات الأخيرة", "آخر التطبيقات", "recents",
        ],
        "category": "Navigation", "description": "Recent Apps"
    },
    "live": {
        "code": "tv_live",
        "aliases": [
            "live", "live tv", "live broadcast",
            "مباشر", "التلفزيون المباشر", "بث مباشر", "en direct",
        ],
        "category": "Navigation", "description": "Live TV"
    },

    # ── INPUT & SETTINGS ───────────────────────────────────────────
    "tv source": {
        "code": "tv_tv",
        "aliases": [
            "tv source", "antenna", "terrestrial", "air tv",
            "هوائي", "المصدر التلفزيوني", "antenne",
        ],
        "category": "Input", "description": "TV Source (Antenna)"
    },
    "input": {
        "code": "tv_input",
        "aliases": [
            "input", "source", "change input", "change source", "switch input",
            "المصدر", "الإدخال", "بدّل المصدر", "entree", "source entree",
        ],
        "category": "Input", "description": "Switch Input"
    },
    "settings": {
        "code": "tv_settings",
        "aliases": [
            "settings", "options", "preferences", "configure", "setup",
            "الإعدادات", "ضبط", "إعدادات", "parametres", "reglages",
        ],
        "category": "Settings", "description": "Settings Menu"
    },
    "zoom": {
        "code": "tv_zoom",
        "aliases": [
            "zoom", "zoom in", "zoom out", "aspect", "picture size",
            "زوم", "كبّر الصورة", "حجم الصورة",
        ],
        "category": "Settings", "description": "Zoom / Aspect"
    },
    "subtitles": {
        "code": "tv_sub",
        "aliases": [
            "subtitles", "subtitle", "sub", "captions",
            "ترجمة", "ترجمات", "sous-titres", "sous titres",
        ],
        "category": "Settings", "description": "Subtitles"
    },

    # ── INFO / GUIDE ────────────────────────────────────────────────
    "info": {
        "code": "tv_info",
        "aliases": [
            "info", "information", "show info", "channel info",
            "معلومات", "المعلومات", "informations",
        ],
        "category": "Guide", "description": "Info"
    },
    "guide": {
        "code": "tv_guide",
        "aliases": [
            "guide", "tv guide", "program guide", "schedule",
            "الدليل", "دليل البرامج", "الجدول", "guide tv",
        ],
        "category": "Guide", "description": "TV Guide"
    },
    "list": {
        "code": "tv_list",
        "aliases": [
            "list", "channel list", "favorite list",
            "قائمة", "قائمة القنوات", "liste", "liste des chaines",
        ],
        "category": "Guide", "description": "Channel List"
    },
    "search": {
        "code": "tv_search",
        "aliases": [
            "search", "find", "look for", "search for",
            "ابحث", "البحث", "لقّي", "chercher", "rechercher",
        ],
        "category": "Guide", "description": "Search"
    },

    # ── MEDIA PLAYBACK ─────────────────────────────────────────────
    "play": {
        "code": "tv_play",
        "aliases": [
            "play", "resume", "continue", "start playing",
            "تشغيل", "شغّل", "استئناف", "jouer", "reprendre",
        ],
        "category": "Media", "description": "Play / Resume"
    },
    "pause": {
        "code": "tv_pause",
        "aliases": [
            "pause", "freeze", "hold", "wait",
            "إيقاف مؤقت", "وقّف", "جمّد", "pauser",
        ],
        "category": "Media", "description": "Pause"
    },
    "stop": {
        "code": "tv_stop",
        "aliases": [
            "stop", "end", "finish", "stop playing",
            "إيقاف", "خلّص", "arreter",
        ],
        "category": "Media", "description": "Stop"
    },
    "record": {
        "code": "tv_rec",
        "aliases": [
            "record", "recording", "start recording", "rec",
            "سجّل", "تسجيل", "ابدأ التسجيل", "enregistrer",
        ],
        "category": "Media", "description": "Record"
    },
    "rewind": {
        "code": "tv_rew",
        "aliases": [
            "rewind", "go back", "backward", "back up",
            "ارجع", "للخلف", "رجّع", "rembobiner",
        ],
        "category": "Media", "description": "Rewind"
    },
    "fast forward": {
        "code": "tv_ff",
        "aliases": [
            "fast forward", "forward", "skip ahead", "skip forward",
            "قدّم", "ابعد للأمام", "تخطي", "avance rapide",
        ],
        "category": "Media", "description": "Fast Forward"
    },

    # ── COLOR BUTTONS ──────────────────────────────────────────────
    "red": {
        "code": "tv_red",
        "aliases": ["red", "red button", "احمر", "rouge"],
        "category": "Color", "description": "Red Button"
    },
    "green": {
        "code": "tv_green",
        "aliases": ["green", "green button", "اخضر", "vert"],
        "category": "Color", "description": "Green Button"
    },
    "yellow": {
        "code": "tv_yellow",
        "aliases": ["yellow", "yellow button", "اصفر", "jaune"],
        "category": "Color", "description": "Yellow Button"
    },
    "blue": {
        "code": "tv_blue",
        "aliases": ["blue", "blue button", "ازرق", "bleu"],
        "category": "Color", "description": "Blue Button"
    },

    # ── HDMI INPUTS ────────────────────────────────────────────────
    "hdmi1": {
        "code": "tv_hdmi1",
        "aliases": [
            "hdmi 1", "hdmi1", "hdmi one", "input 1",
            "hdmi واحد", "المدخل الاول",
        ],
        "category": "Input", "description": "HDMI 1"
    },
    "hdmi2": {
        "code": "tv_hdmi2",
        "aliases": [
            "hdmi 2", "hdmi2", "hdmi two", "input 2",
            "hdmi جوج", "المدخل الثاني",
        ],
        "category": "Input", "description": "HDMI 2"
    },
    "hdmi3": {
        "code": "tv_hdmi3",
        "aliases": [
            "hdmi 3", "hdmi3", "hdmi three", "input 3",
            "hdmi تلاتة", "المدخل الثالث",
        ],
        "category": "Input", "description": "HDMI 3"
    },
    "hdmi4": {
        "code": "tv_hdmi4",
        "aliases": [
            "hdmi 4", "hdmi4", "hdmi four", "input 4",
            "hdmi ربعة", "المدخل الرابع",
        ],
        "category": "Input", "description": "HDMI 4"
    },

    # ── ADVANCED ───────────────────────────────────────────────────
    "energy": {
        "code": "tv_energy",
        "aliases": ["energy", "energy saving", "eco", "توفير الطاقة", "energie"],
        "category": "Advanced", "description": "Energy Saving"
    },
    "quick menu": {
        "code": "tv_quick",
        "aliases": ["quick menu", "quick settings", "قائمة سريعة", "menu rapide"],
        "category": "Advanced", "description": "Quick Menu"
    },
    "favorites": {
        "code": "tv_fav",
        "aliases": ["favorites", "favourite", "fav", "المفضلة", "favoris"],
        "category": "Advanced", "description": "Favorites"
    },
    "ratio": {
        "code": "tv_ratio",
        "aliases": ["ratio", "aspect ratio", "screen ratio", "نسبة العرض", "format"],
        "category": "Advanced", "description": "Aspect Ratio"
    },
    "picture mode": {
        "code": "tv_picture",
        "aliases": ["picture", "picture mode", "image mode", "وضع الصورة", "image"],
        "category": "Advanced", "description": "Picture Mode"
    },
    "sleep timer": {
        "code": "tv_sleep",
        "aliases": ["sleep", "sleep timer", "auto off", "مؤقت النوم", "minuterie"],
        "category": "Advanced", "description": "Sleep Timer"
    },
    "3d": {
        "code": "tv_3d",
        "aliases": ["3d", "3d mode", "three d", "ثلاثي الابعاد"],
        "category": "Advanced", "description": "3D Mode"
    },
    "teletext": {
        "code": "tv_teletext",
        "aliases": ["teletext", "text", "تيليتكست", "teletext"],
        "category": "Advanced", "description": "Teletext"
    },
    "teletext options": {
        "code": "tv_ttopt",
        "aliases": ["teletext options", "text options", "tt options", "خيارات تيليتكست"],
        "category": "Advanced", "description": "Teletext Options"
    },
    "audio description": {
        "code": "tv_ad",
        "aliases": ["audio description", "audio desc", "ad", "الوصف الصوتي"],
        "category": "Advanced", "description": "Audio Description"
    },
}

# ── TV: Number keys 0–9 ────────────────────────────────────────────
_TV_DIGIT_EN = ["zero","one","two","three","four","five","six","seven","eight","nine"]
_TV_DIGIT_AR = ["صفر","واحد","جوج","تلاتة","ربعة","خمسة","ستة","سبعة","تمانية","تسعة"]
_TV_DIGIT_FR = ["zero","un","deux","trois","quatre","cinq","six","sept","huit","neuf"]

for _d in range(10):
    TV_COMMANDS[f"digit_{_d}"] = {
        "code": f"tv_{_d}",
        "aliases": [
            str(_d),
            _TV_DIGIT_EN[_d],
            _TV_DIGIT_FR[_d],
            _TV_DIGIT_AR[_d],
            f"number {_d}",
            f"channel {_d}",
        ],
        "category": "Number",
        "description": f"Key {_d}"
    }

# ─────────────────────────────────────────────
#  AC COMMANDS  (hardware codes = ac_*)
#  Output sent to device = "00" + code
# ─────────────────────────────────────────────
AC_COMMANDS = {

    # ── POWER ──────────────────────────────────────────────────────
    "power on": {
        "code": "ac_on",
        "aliases": [
            "power on", "turn on", "on", "start", "open", "switch on",
            "ولّع الكليما", "ولّع الكليم", "شعل الكليما", "حل الكليما",
            "شغّل المكيف", "ولّع", "خدّم", "شعل",
            "allumer le clim", "allumer",
        ],
        "category": "Power", "description": "AC Power ON (cool 24°C)"
    },
    "power off": {
        "code": "ac_off",
        "aliases": [
            "power off", "turn off", "off", "stop", "shutdown", "switch off",
            "طفّي الكليما", "طفّي الكليم", "حبس الكليما", "سكّر الكليما",
            "وقّف المكيف", "طفّي", "حبس", "سكّر",
            "eteindre le clim", "eteindre",
        ],
        "category": "Power", "description": "AC Power OFF"
    },

    # ── MODES ──────────────────────────────────────────────────────
    "cool mode": {
        "code": "ac_cool",
        "aliases": [
            "cool", "cooling", "cool mode", "cold", "cold mode",
            "تبريد", "وضع تبريد", "برّد", "برودة", "كول",
            "froid", "mode froid", "refroidissement",
        ],
        "category": "Mode", "description": "Cooling Mode (22°C)"
    },
    "heat mode": {
        "code": "ac_hot",
        "aliases": [
            "heat", "heating", "heat mode", "warm", "hot",
            "تسخين", "وضع تسخين", "سخّن", "حرارة", "دفا",
            "chaud", "mode chaud", "chauffage",
        ],
        "category": "Mode", "description": "Heating Mode (26°C)"
    },
    "fan mode": {
        "code": "ac_fan",
        "aliases": [
            "fan", "fan mode", "ventilation", "ventilate", "air only",
            "مروحة", "فان", "تهوية", "وضع مروحة", "هوا بس",
            "ventilateur", "mode ventilation",
        ],
        "category": "Mode", "description": "Fan Only Mode"
    },
    "dry mode": {
        "code": "ac_dry",
        "aliases": [
            "dry", "dry mode", "dehumidify",
            "تجفيف", "وضع تجفيف", "نشّف", "جاف",
            "sec", "mode sec",
        ],
        "category": "Mode", "description": "Dry Mode"
    },

    # ── TEMPERATURE STEP ───────────────────────────────────────────
    "temp up": {
        "code": "ac_temp_up",
        "aliases": [
            "temp up", "temperature up", "warmer", "hotter", "increase temperature",
            "raise temperature", "heat up", "degree up",
            "زيد الحرارة", "زيد درجة", "سخّن شوية", "ارفع الحرارة",
            "augmenter la temperature", "plus chaud",
        ],
        "category": "Temperature", "description": "Temperature +1°C"
    },
    "temp down": {
        "code": "ac_temp_down",
        "aliases": [
            "temp down", "temperature down", "cooler", "colder", "decrease temperature",
            "lower temperature", "cool down", "degree down",
            "نقّص الحرارة", "نقّص درجة", "برّد شوية", "اخفض الحرارة",
            "diminuer la temperature", "plus froid",
        ],
        "category": "Temperature", "description": "Temperature -1°C"
    },

    # ── FAN SPEED ──────────────────────────────────────────────────
    "fan auto": {
        "code": "ac_fan_auto",
        "aliases": [
            "fan auto", "automatic fan", "auto fan", "auto",
            "مروحة اوتو", "ريح اوتو", "تلقائي",
            "ventilateur auto", "vitesse auto",
        ],
        "category": "Fan", "description": "Fan Speed Auto"
    },
    "fan low": {
        "code": "ac_fan_low",
        "aliases": [
            "fan low", "low fan", "slow fan", "fan 1",
            "مروحة هادية", "ريح هادية", "ريح خفيفة",
            "ventilateur faible", "vitesse lente",
        ],
        "category": "Fan", "description": "Fan Speed Low"
    },
    "fan medium": {
        "code": "ac_fan_med",
        "aliases": [
            "fan medium", "medium fan", "fan med", "fan 2",
            "مروحة متوسطة", "ريح متوسطة", "وسط",
            "ventilateur moyen", "vitesse moyenne",
        ],
        "category": "Fan", "description": "Fan Speed Medium"
    },
    "fan high": {
        "code": "ac_fan_high",
        "aliases": [
            "fan high", "high fan", "fast fan", "fan 3", "max fan",
            "مروحة قوية", "ريح قوية", "اقوى ريح",
            "ventilateur fort", "vitesse elevee",
        ],
        "category": "Fan", "description": "Fan Speed High"
    },

    # ── SWING ──────────────────────────────────────────────────────
    "swing": {
        "code": "ac_swing",
        "aliases": [
            "swing", "swing toggle", "oscillate", "move air", "auto swing",
            "سوينغ", "حرّك الهوا", "دوّر الريش", "تحريك",
        ],
        "category": "Swing", "description": "Swing Toggle"
    },
}

# ── AC: Temperature commands 16–30 ────────────────────────────────
_TEMP_EN = {
    16: ["sixteen"], 17: ["seventeen"], 18: ["eighteen"], 19: ["nineteen"],
    20: ["twenty"], 21: ["twenty one", "twenty-one"],
    22: ["twenty two", "twenty-two"], 23: ["twenty three", "twenty-three"],
    24: ["twenty four", "twenty-four"], 25: ["twenty five", "twenty-five"],
    26: ["twenty six", "twenty-six"], 27: ["twenty seven", "twenty-seven"],
    28: ["twenty eight", "twenty-eight"], 29: ["twenty nine", "twenty-nine"],
    30: ["thirty"],
}
_TEMP_FR = {
    16: ["seize"], 17: ["dix-sept"], 18: ["dix-huit"], 19: ["dix-neuf"],
    20: ["vingt"], 21: ["vingt et un"], 22: ["vingt-deux"],
    23: ["vingt-trois"], 24: ["vingt-quatre"], 25: ["vingt-cinq"],
    26: ["vingt-six"], 27: ["vingt-sept"], 28: ["vingt-huit"],
    29: ["vingt-neuf"], 30: ["trente"],
}
_TEMP_AR = {
    16: ["ستاش", "سيتاش", "ستة عشر"],
    17: ["سبعتاش", "سبعطاش", "سبعة عشر"],
    18: ["تمنتاش", "ثمانتاش", "ثمانية عشر"],
    19: ["تسعتاش", "تسعة عشر"],
    20: ["عشرين"],
    21: ["واحد وعشرين"],
    22: ["جوج وعشرين", "اثنين وعشرين"],
    23: ["تلاتة وعشرين", "ثلاثة وعشرين"],
    24: ["ربعة وعشرين", "اربعة وعشرين"],
    25: ["خمسة وعشرين", "حمسة وعشرين"],
    26: ["ستة وعشرين"],
    27: ["سبعة وعشرين"],
    28: ["تمانية وعشرين", "ثمانية وعشرين"],
    29: ["تسعة وعشرين"],
    30: ["تلاتين"],
}

for _n in range(16, 31):
    AC_COMMANDS[f"temp_{_n}"] = {
        "code": f"ac_{_n}",
        "aliases": (
            [str(_n), f"set {_n}", f"temperature {_n}", f"{_n} degrees",
             f"درجة {_n}", f"حط {_n}"]
            + _TEMP_EN.get(_n, [])
            + _TEMP_FR.get(_n, [])
            + _TEMP_AR.get(_n, [])
        ),
        "category": "Temperature",
        "description": f"Set temperature to {_n}°C"
    }

# ─────────────────────────────────────────────
#  DEVICE REGISTRY
# ─────────────────────────────────────────────
DEVICES = {
    "tv": {
        "name": "TV",
        "prefix": "11",
        "triggers": [
            "tv", "t v", "tee vee", "tiv", "tivi", "the tv", "television", "telly",
            "تيفي", "تي في", "تلفاز", "التلفزيون", "تلفزيون", "تيلي",
        ],
        "commands": TV_COMMANDS,
    },
    "ac": {
        "name": "Air Conditioner",
        "prefix": "00",
        "triggers": [
            "clem", "clim", "klim", "klima", "kleem", "la clim", "le clim",
            "ac", "air conditioner", "air conditioning", "air con",
            "كليم", "كليما", "الكليما", "الكليم", "المكيف", "مكيف",
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
#  DEVICE COMMAND MATCHER
# ─────────────────────────────────────────────
class DeviceCommandMatcher:
    def __init__(self, device_key, device, model):
        self.device_key  = device_key
        self.name        = device["name"]
        self.prefix      = device["prefix"]
        self.commands    = device["commands"]
        self.model       = model
        self.alias_map   = build_alias_map(self.commands)
        self.all_aliases = list(self.alias_map.keys())

        print(f"[NLP] Encoding aliases for {self.name}...")
        self.alias_embeddings = self.model.encode(
            self.all_aliases, convert_to_tensor=True, show_progress_bar=False
        )

    def _result(self, spoken, cmd, method, confidence, t0):
        d = self.commands[cmd]
        return {
            "input":           spoken,
            "device":          self.name,
            "command":         cmd,
            "code":            d["code"],
            "full_code":       self.prefix + d["code"],
            "method":          method,
            "confidence":      confidence,
            "description":     d["description"],
            "category":        d["category"],
            "processing_time": time.time() - t0,
        }

    def _no_match(self, spoken, t0):
        return {
            "input": spoken, "device": self.name,
            "command": None, "code": None, "full_code": None,
            "method": "no match", "confidence": 0.0,
            "description": None, "category": None,
            "processing_time": time.time() - t0,
        }

    def match(self, spoken_text, fuzzy_threshold=70, semantic_threshold=0.55):
        t0   = time.time()
        text = spoken_text.lower().strip()

        if not text:
            return self._no_match(spoken_text, t0)

        # 1. Exact match
        if text in self.alias_map:
            return self._result(spoken_text, self.alias_map[text], "exact", 1.0, t0)

        # 2. Substring match
        for alias, cmd in self.alias_map.items():
            if alias in text or text in alias:
                return self._result(spoken_text, cmd, "substring", 0.95, t0)

        # 3. Fuzzy match
        best_alias, score, _ = process.extractOne(
            text, self.all_aliases, scorer=fuzz.WRatio
        )
        if score >= fuzzy_threshold:
            return self._result(spoken_text, self.alias_map[best_alias],
                                f"fuzzy ({score:.0f}%)", score / 100, t0)

        # 4. Semantic NLP
        q_emb  = self.model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.alias_embeddings)[0]
        best_i = int(scores.argmax())
        best_s = float(scores[best_i])
        if best_s >= semantic_threshold:
            return self._result(spoken_text, self.alias_map[self.all_aliases[best_i]],
                                f"semantic ({best_s:.2f})", best_s, t0)

        return self._no_match(spoken_text, t0)

# ─────────────────────────────────────────────
#  MULTI-DEVICE ASSISTANT
# ─────────────────────────────────────────────
class MultiDeviceAssistant:
    def __init__(self, devices):
        print("[NLP] Loading semantic model...")
        self.model   = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.devices = devices
        self.matchers = {
            key: DeviceCommandMatcher(key, dev, self.model)
            for key, dev in devices.items()
        }
        print("[NLP] All devices ready!\n")

    def detect_device(self, spoken_text, fuzzy_threshold=80):
        """Return (device_key, remainder_command) or (None, None)."""
        text = spoken_text.lower().strip()
        if not text:
            return None, None

        # Longest triggers first (so "air conditioner" beats "ac")
        candidates = sorted(
            [(t.lower(), k)
             for k, d in self.devices.items()
             for t in d["triggers"]],
            key=lambda x: -len(x[0])
        )

        for trigger, key in candidates:
            if text == trigger:
                return key, None
            if text.startswith(trigger + " "):
                rem = text[len(trigger):].strip()
                return key, rem or None

        # Fuzzy fallback on first word
        first = text.split()[0]
        best_key, best_score = None, 0
        for trigger, key in candidates:
            s = fuzz.ratio(first, trigger)
            if s > best_score:
                best_score, best_key = s, key
        if best_score >= fuzzy_threshold:
            rem = " ".join(text.split()[1:])
            return best_key, (rem if rem else None)

        return None, None

# ─────────────────────────────────────────────
#  MICROPHONE LISTENER
# ─────────────────────────────────────────────
def listen_and_recognize(recognizer, language, tracker=None):
    with sr.Microphone() as source:
        print("Listening... (speak now)")
        rec_start = time.time()
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            print("Timeout — no speech detected\n")
            return None

        if tracker:
            tracker.record_audio_time(time.time() - rec_start)

        print("Processing...")
        stt_start = time.time()
        try:
            text = recognizer.recognize_google(audio, language=language)
            if tracker:
                tracker.record_stt_time(time.time() - stt_start)
            return text
        except sr.UnknownValueError:
            print("Could not understand — please try again\n")
        except sr.RequestError as e:
            print(f"Network error: {e}\n")
        return None

# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def open_serial(port, baud):
    """Open serial port. Returns serial object or None if unavailable."""
    if not port:
        return None
    if not SERIAL_AVAILABLE:
        print("[Serial] pyserial not installed — run: pip install pyserial")
        return None
    try:
        conn = pyserial.Serial(port, baud, timeout=1)
        time.sleep(2)  # wait for device to reset after connection
        print(f"[Serial] Connected: {port} @ {baud} baud")
        return conn
    except Exception as e:
        print(f"[Serial] Could not open {port}: {e}")
        return None

def send_serial(conn, code):
    """Send command code over serial with newline terminator."""
    if conn is None:
        return
    try:
        conn.write((code + "\n").encode("utf-8"))
        conn.flush()
        print(f"[Serial] Sent -> {code}")
    except Exception as e:
        print(f"[Serial] Send error: {e}")

def print_header(serial_port=None, baud=9600):
    print("\n" + "="*62)
    print("   MULTI-DEVICE VOICE ASSISTANT")
    print("="*62)
    print("  TV  -> 'tv open'  'tv volume up'  'tv hdmi 1'  'tv 5'")
    print("  AC  -> 'clem power on'  'clem cool'  'clem 25'  'clem fan low'")
    print("-"*62)
    for k, d in DEVICES.items():
        print(f"  {d['name']:<22} {len(d['commands'])} commands")
    if serial_port:
        print(f"  Serial port : {serial_port} @ {baud} baud")
    else:
        print("  Serial port : NOT connected (screen only)")
    print("="*62)
    print("  Press Ctrl+C to quit\n")

def print_result(device_name, result, show_performance=False):
    if result and result["command"]:
        print("\n" + "-"*46)
        print(f"  Heard      : {result['input']}")
        print(f"  Device     : {device_name}")
        print(f"  Command    : {result['command']}")
        print(f"  Serial OUT : {result['code']}")
        print(f"  Method     : {result['method']}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Action     : {result['description']}")
        if show_performance:
            print(f"  Processing : {result['processing_time']*1000:.2f}ms")
        print("-"*46 + "\n")
    elif result:
        print(f"\n  [NO MATCH] [{device_name}] '{result['input']}'")
        print("  Try rephrasing or speak more clearly.\n")

def print_performance_stats(tracker):
    s = tracker.get_stats()
    print("\n" + "="*50)
    print("  PERFORMANCE STATISTICS")
    print("="*50)
    print(f"  Session   : {s['session_duration']:.1f}s")
    print(f"  Commands  : {s['total_commands']} total / {s['matched_commands']} matched")
    print(f"  Accuracy  : {s['accuracy']:.1f}%")
    print(f"  Avg STT   : {s['avg_stt']*1000:.0f} ms")
    print(f"  Avg Match : {s['avg_match']*1000:.1f} ms")
    print("="*50 + "\n")

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def run(language="en-US", show_performance=False, serial_port=None, baud=9600):
    print_header(serial_port, baud)

    conn       = open_serial(serial_port, baud)
    assistant  = MultiDeviceAssistant(DEVICES)
    recognizer = sr.Recognizer()
    recognizer.energy_threshold         = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold          = 0.8

    tracker = PerformanceTracker() if show_performance else None

    lang_names = {
        "en-US": "English",
        "fr-FR": "French",
        "ar-DZ": "Arabic (Algerian Darija)"
    }
    print(f"Language : {lang_names.get(language, language)}")
    print(f"Devices  : {', '.join(d['name'] for d in DEVICES.values())}\n")

    while True:
        try:
            t_start = time.time()
            spoken  = listen_and_recognize(recognizer, language, tracker)

            if spoken is None:
                continue

            print(f'You said: "{spoken}"')

            device_key, remainder = assistant.detect_device(spoken)

            if device_key is None:
                print("No device detected. Start with 'tv ...' or 'clem ...'\n")
                continue

            dev = DEVICES[device_key]
            print(f"Device: {dev['name']}")

            if not remainder:
                print(f"No command after '{dev['name']}' — speak again.\n")
                continue

            print(f"Command: \"{remainder}\"")
            t_match = time.time()
            result  = assistant.matchers[device_key].match(remainder)

            if tracker:
                tracker.record_matching_time(time.time() - t_match)
                tracker.record_processing_time(time.time() - t_start)
                tracker.command_attempted(result["command"] is not None)

            # Send raw hardware code over serial (no prefix)
            if result and result["code"]:
                send_serial(conn, result["code"])

            print_result(dev["name"], result, show_performance)

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye!\n")
            if tracker:
                print_performance_stats(tracker)
            if conn:
                conn.close()
            sys.exit(0)

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Device Voice Assistant (TV + AC)")
    parser.add_argument("--lang", default="en-US",
                        choices=["en-US", "fr-FR", "ar-DZ"],
                        help="Speech language (default: en-US)")
    parser.add_argument("--port", default=None,
                        help="Serial port (e.g. /dev/cu.usbserial-110)")
    parser.add_argument("--baud", default=9600, type=int,
                        help="Serial baud rate (default: 9600)")
    parser.add_argument("--performance", action="store_true",
                        help="Show timing stats")
    args = parser.parse_args()
    run(language=args.lang, show_performance=args.performance,
        serial_port=args.port, baud=args.baud)
