"""
SMART HOME VOICE ASSISTANT - OFFLINE / FAST VERSION
====================================================
Fully offline — no internet needed after first model download.
Controls: TV (تيفي) and Air Conditioner (كليم).
Language: Arabic Algerian Darija + French + English.

Speed optimisations vs previous version:
  • Shared SentenceTransformer (loaded once, used for both devices)
  • Whisper tiny by default  (4x faster than small, good for short commands)
  • beam_size=1  (greedy decode, 3-4x faster than beam_size=5)
  • NumPy audio buffer  (no temp-file disk I/O)
  • Ambient noise calibration once at startup, not per listen
  • phrase_time_limit=4s  (Darija commands are 1-3 words)
  • pause_threshold=0.5s  (faster end-of-speech detection)
  • Whisper pre-warmed on startup to avoid cold-start latency

Usage:
    python tv_voice_assistant_enhanced.py
    python tv_voice_assistant_enhanced.py --model small   # more accurate
    python tv_voice_assistant_enhanced.py --performance

Install:
    pip install faster-whisper speechrecognition pyaudio sentence-transformers rapidfuzz
"""

import sys
import time
import argparse
import numpy as np
from collections import defaultdict

import speech_recognition as sr
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process


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

    def command_attempted(self, matched=True):
        self.command_count += 1
        if matched:
            self.matched_count += 1

    def get_stats(self):
        return {
            "session_duration":  time.time() - self.session_start,
            "total_commands":    self.command_count,
            "matched_commands":  self.matched_count,
            "accuracy":          self.matched_count / max(1, self.command_count) * 100,
            "avg_recording":     self._avg("recording"),
            "avg_stt":           self._avg("stt"),
            "avg_matching":      self._avg("matching"),
            "avg_total":         self._avg("total"),
        }

    def _avg(self, key):
        v = self.metrics[key]
        return sum(v) / len(v) if v else 0


# ─────────────────────────────────────────────
#  TV COMMANDS DATABASE  (Darija + FR + EN)
# ─────────────────────────────────────────────
TV_COMMANDS = {
    # ── Power ──────────────────────────────────
    "open": {
        "code": "0001",
        "aliases": [
            # English
            "open", "turn on", "power on", "start", "on", "switch on", "wake up", "boot",
            # French
            "allumer", "allume", "ouvrir", "ouvre", "démarrer", "démarre",
            "mettre en marche", "brancher",
            # Darija — power on TV
            "شعل", "شعلو", "شعل التيفي", "شعل التلفاز", "شعل التلفزيون",
            "ولع", "ولعو", "ولع التيفي", "ولع التلفزيون",
            "خدم", "خدمو", "خدم التيفي", "دير التيفي يخدم", "دير يخدم",
            "شمّع", "شمّعو", "شمّع التيفي",
            "فتح", "فتحو", "افتح", "افتح التيفي",
            "شغّل", "شغّلو", "شغّل التيفي",
            "دور", "ديره يخدم", "حيّه",
        ],
        "category": "Power",
        "description": "Turn ON the TV",
    },
    "close": {
        "code": "0002",
        "aliases": [
            "close", "turn off", "power off", "shutdown", "off", "switch off", "sleep",
            "éteindre", "éteins", "fermer", "ferme", "arrêter", "arrête",
            "طفي", "طفيو", "طفي التيفي", "طفي التلفزيون",
            "حبس", "حبسو", "حبس التيفي",
            "سكر", "سكرو", "سكر التيفي",
            "قفل", "قفلو", "قفل التيفي",
            "بطّل", "بطّلو", "بطّل التيفي",
            "أطفي", "أطفيه", "طفيه",
            "وقّف", "وقّفو",
        ],
        "category": "Power",
        "description": "Turn OFF the TV",
    },

    # ── Volume ─────────────────────────────────
    "mute": {
        "code": "0004",
        "aliases": [
            "mute", "silent", "silence", "quiet", "no sound", "shh",
            "sourdine", "couper le son", "silencieux", "sans son",
            "اسكت", "سكات", "صامت", "كتم",
            "حبس الصوت", "قطع الصوت", "بلا صوت",
            "ما نسمعش", "ما حبيتش نسمع", "سكّت الصوت",
            "كتم الصوت", "اقطع الصوت", "صفر صوت",
        ],
        "category": "Volume",
        "description": "Mute / Unmute",
    },
    "volume up": {
        "code": "0005",
        "aliases": [
            "volume up", "louder", "increase volume", "turn up", "sound up",
            "raise volume", "more sound", "amplify", "crank it up",
            "plus fort", "augmenter le volume", "monter le son", "plus de son",
            "زيد الصوت", "زيد في الصوت", "زيدو الصوت",
            "رفع الصوت", "ارفع الصوت", "رفعو الصوت",
            "عالي الصوت", "أعلى", "أعلى شوية",
            "قوي الصوت", "قوّيه", "الصوت مسمعش", "ما نسمعش وين",
            "صوت أكثر", "زيد شوية", "قوّي شوية",
        ],
        "category": "Volume",
        "description": "Increase Volume",
    },
    "volume down": {
        "code": "0006",
        "aliases": [
            "volume down", "quieter", "decrease volume", "turn down", "sound down",
            "lower volume", "softer", "less sound", "reduce volume",
            "moins fort", "baisser le son", "diminuer le volume", "réduire le son",
            "نقص الصوت", "نقص في الصوت", "نقصو الصوت",
            "هادي الصوت", "هاديه شوية", "خفض الصوت", "اخفض الصوت",
            "أهدى", "أهدى شوية", "الصوت عالي برك",
            "الصوت يقتلني", "صوت أقل", "شوية أهدى",
        ],
        "category": "Volume",
        "description": "Decrease Volume",
    },

    # ── Channel ────────────────────────────────
    "channel up": {
        "code": "0007",
        "aliases": [
            "channel up", "next channel", "next", "forward channel",
            "chaîne suivante", "avancer la chaîne", "prochaine chaîne",
            "القناة الجاية", "الجاية", "قناة لفوق",
            "بدل للجاية", "بدّل قناة", "دور على قناة",
            "شوف اش كاين", "شوف أخرى", "روح لقدام",
            "قناة أخرى", "عدّي", "قناة أحسن",
        ],
        "category": "Channel",
        "description": "Next Channel",
    },
    "channel down": {
        "code": "0008",
        "aliases": [
            "channel down", "previous channel", "back channel", "prior channel",
            "chaîne précédente", "reculer la chaîne", "chaîne d'avant",
            "القناة الفايتة", "الفايتة", "قناة لتحت",
            "ارجع قناة", "راجع قناة", "القناة اللي كنا فيها",
            "راجع للفايتة", "الفايتة كانت حلوة",
        ],
        "category": "Channel",
        "description": "Previous Channel",
    },

    # ── Navigation ─────────────────────────────
    "home": {
        "code": "0009",
        "aliases": [
            "home", "home screen", "main menu", "dashboard", "go home",
            "accueil", "menu principal", "écran d'accueil", "page principale",
            "الصفحة الرئيسية", "روح للبداية", "الشاشة الرئيسية",
            "المنيو", "ارجع للمنيو", "رئيسي", "البداية",
            "روح لاكيا", "الواجهة",
        ],
        "category": "Navigation",
        "description": "Go to Home Screen",
    },
    "settings": {
        "code": "0003",
        "aliases": [
            "settings", "options", "configure", "preferences", "setup",
            "paramètres", "réglages", "options", "configurer",
            "الإعدادات", "السيتينغ", "الضبط",
            "روح للإعدادات", "حل الإعدادات", "التوازنات",
            "ضبط التيفي", "إعدادات التلفزيون",
        ],
        "category": "Navigation",
        "description": "Open Settings Menu",
    },
    "input": {
        "code": "0010",
        "aliases": [
            "input", "source", "hdmi", "change input", "switch input", "change source",
            "entrée", "source d'entrée", "changer d'entrée", "hdmi",
            "المصدر", "الإدخال", "بدل المصدر",
            "hdmi", "الكابل", "الموجة", "بدّل المصدر",
        ],
        "category": "Navigation",
        "description": "Switch Input Source",
    },

    # ── Playback ───────────────────────────────
    "play": {
        "code": "0011",
        "aliases": [
            "play", "resume", "continue", "start playing", "go",
            "jouer", "reprendre", "continuer", "lancer",
            "شغّل", "دوّر", "استمر", "استئناف",
            "دور الفيديو", "شغّل الفيديو", "ابدأ",
            "واصل", "واصل التشغيل",
        ],
        "category": "Playback",
        "description": "Play / Resume",
    },
    "pause": {
        "code": "0012",
        "aliases": [
            "pause", "freeze", "hold", "wait", "stop for a sec",
            "pauser", "mettre en pause", "attendre", "suspendre",
            "وقف", "وقّف", "سكن", "انتظر",
            "إيقاف مؤقت", "توقف لحظة", "علّق", "فريّز",
            "جمّد", "سكن شوية",
        ],
        "category": "Playback",
        "description": "Pause",
    },
    "stop": {
        "code": "0013",
        "aliases": [
            "stop", "end playback", "finish", "exit playback",
            "arrêter la lecture", "terminer", "quitter",
            "إيقاف", "حبس التشغيل", "توقف نهائي", "بطّل التشغيل",
        ],
        "category": "Playback",
        "description": "Stop Playback",
    },
    "rewind": {
        "code": "0014",
        "aliases": [
            "rewind", "go back", "backward", "back up",
            "rembobiner", "retourner en arrière",
            "ارجع", "رجّع", "للخلف", "اللي فات",
            "ارجع شوية", "رجعلي شوية",
        ],
        "category": "Playback",
        "description": "Rewind",
    },
    "fast forward": {
        "code": "0015",
        "aliases": [
            "fast forward", "skip forward", "ahead", "forward",
            "avance rapide", "avancer",
            "قدّم", "روح لقدام", "تخطي", "ابعد للأمام",
            "قدّم شوية", "تجاوز",
        ],
        "category": "Playback",
        "description": "Fast Forward",
    },

    # ── Apps ───────────────────────────────────
    "netflix": {
        "code": "0020",
        "aliases": [
            "netflix", "open netflix", "watch netflix", "netflix please",
            "ouvrir netflix", "mettre netflix",
            "نيتفليكس", "افتح نيتفليكس", "حل نيتفليكس",
            "دير نيتفليكس", "بغيت نيتفليكس", "شوف نيتفليكس",
        ],
        "category": "App",
        "description": "Open Netflix",
    },
    "youtube": {
        "code": "0021",
        "aliases": [
            "youtube", "open youtube", "watch youtube", "you tube",
            "ouvrir youtube", "mettre youtube",
            "يوتيوب", "افتح يوتيوب", "حل يوتيوب",
            "دير يوتيوب", "بغيت يوتيوب", "شوف يوتيوب",
        ],
        "category": "App",
        "description": "Open YouTube",
    },
    "spotify": {
        "code": "0022",
        "aliases": [
            "spotify", "open spotify", "music", "listen spotify",
            "ouvrir spotify", "musique", "écouter spotify",
            "سبوتيفاي", "افتح سبوتيفاي", "حل سبوتيفاي",
            "موسيقى", "موزيكا", "بغيت نسمع موزيكا",
        ],
        "category": "App",
        "description": "Open Spotify",
    },
    "search": {
        "code": "0033",
        "aliases": [
            "search", "find", "look for", "search for",
            "chercher", "rechercher", "trouver",
            "ابحث", "لقّي", "دور على", "شوف",
            "قلّب على", "فتش على",
        ],
        "category": "Navigation",
        "description": "Open Search",
    },
    "guide": {
        "code": "0032",
        "aliases": [
            "guide", "tv guide", "what's on", "program guide", "schedule",
            "guide tv", "programme", "grille des programmes",
            "الدليل", "شو كاين", "البرامج", "جدول البرامج",
            "شو عندهم", "الجدول", "دليل القنوات",
        ],
        "category": "Navigation",
        "description": "Open TV Guide",
    },

    # ── Zoom ───────────────────────────────────
    "zoom": {
        "code": "0034",
        "aliases": [
            # English
            "zoom", "zoom in", "zoom out", "picture zoom", "aspect ratio",
            "fit screen", "fill screen", "picture size", "screen size",
            # French
            "zoom", "zoomer", "agrandir", "agrandir l'image", "réduire",
            "plein écran", "taille de l'image", "format d'image",
            "rapport d'aspect", "ajuster l'écran",
            # Darija / Arabic
            "زووم", "زوم", "كبر الصورة", "صغر الصورة",
            "كبرها", "كبّر", "كبّر الصورة", "وسع الشاشة",
            "الصورة صغيرة", "الشاشة ما تكملش", "دير زووم",
            "حجم الصورة", "ملء الشاشة", "املأ الشاشة",
        ],
        "category": "Picture",
        "description": "Zoom / Picture Size",
    },

    # ── Subtitles ──────────────────────────────
    "subtitles": {
        "code": "0035",
        "aliases": [
            # English
            "subtitles", "sub", "subs", "captions", "cc", "closed captions",
            "turn on subtitles", "show subtitles", "text on screen", "toggle subtitles",
            # French
            "sous-titres", "sous-titrage", "légendes", "activer sous-titres",
            "afficher sous-titres", "mettre les sous-titres",
            # Darija / Arabic
            "ترجمة", "الترجمة", "السبتيتل", "سبتيتل",
            "حط الترجمة", "شعل الترجمة", "دير الترجمة",
            "الكتابة", "الكتابة في الشاشة", "الكلام مكتوب",
            "ما نفهمش بلا ترجمة", "افتح الترجمة",
            "ترجمة عربي", "ترجمة فرنساوي",
        ],
        "category": "Display",
        "description": "Toggle Subtitles",
    },

    # ── Info ───────────────────────────────────
    "info": {
        "code": "0036",
        "aliases": [
            # English
            "info", "information", "details", "channel info", "show info",
            "what is this", "program info", "what are we watching",
            # French
            "info", "informations", "détails", "infos",
            "qu'est-ce qu'on regarde", "informations sur le programme",
            "détails du programme",
            # Darija / Arabic
            "معلومات", "إنفو", "المعلومات", "التفاصيل",
            "شو هذا", "إيش هذا", "واش كاين", "شو البرنامج",
            "عطيني معلومات", "شو القناة هذي",
            "أعطيني المعلومات", "بيانات",
        ],
        "category": "Navigation",
        "description": "Show Info",
    },

    # ── List ───────────────────────────────────
    "list": {
        "code": "0037",
        "aliases": [
            # English
            "list", "channel list", "program list", "favorites list",
            "my channels", "favorites", "favourite channels",
            # French
            "liste", "liste des chaînes", "mes chaînes", "mes favoris",
            "liste des favoris", "chaînes favorites",
            # Darija / Arabic
            "اللايسط", "اللستة", "القائمة", "ليستي",
            "قائمة القنوات", "القنوات المفضلة", "المفضلة",
            "حب ديالي", "قنواتي", "شوف اللايسط",
            "افتح القائمة", "القائمة ديالتي",
        ],
        "category": "Navigation",
        "description": "Open Channel List",
    },

    # ── Recent ─────────────────────────────────
    "recent": {
        "code": "0038",
        "aliases": [
            # English
            "recent", "recently watched", "watch history", "last watched",
            "history", "last channels", "previously watched",
            # French
            "récent", "historique", "dernières chaînes",
            "ce que j'ai regardé", "vu récemment",
            # Darija / Arabic
            "الأخير", "اللي شفت", "التاريخ", "السابق",
            "القنوات الأخيرة", "اللي كنت نشوفه", "اللي زرت",
            "الشاشة اللي كانت", "رسنت", "ارجع للأخير",
            "اللي فات", "القنوات السابقة",
        ],
        "category": "Navigation",
        "description": "Open Recent Channels",
    },

    # ── Live ───────────────────────────────────
    "live": {
        "code": "0039",
        "aliases": [
            # English
            "live", "live tv", "live broadcast", "direct", "broadcast",
            "live channel", "watch live", "go live",
            # French
            "direct", "en direct", "télévision en direct", "live",
            "chaîne en direct", "regarder en direct",
            # Darija / Arabic
            "مباشر", "البث المباشر", "لايف", "مباشرة",
            "البث الحي", "التلفزيون المباشر", "القناة المباشرة",
            "شاهد مباشر", "خدني للمباشر", "اللايف",
            "بث مباشر", "حي",
        ],
        "category": "Navigation",
        "description": "Live TV",
    },

    # ── HDMI Inputs ────────────────────────────
    "hdmi1": {
        "code": "0050",
        "aliases": [
            # English
            "hdmi 1", "hdmi1", "hdmi one", "input 1", "source 1", "source one",
            "input one", "switch to hdmi 1",
            # French
            "hdmi 1", "entrée 1", "source 1", "mettre sur hdmi 1",
            # Darija / Arabic
            "hdmi واحد", "hdmi 1", "إيتش دي إم أي واحد",
            "المصدر الأول", "الأول", "المصدر 1",
            "غيّر لـ hdmi واحد", "hdmi الأول",
        ],
        "category": "Input",
        "description": "Switch to HDMI 1",
    },
    "hdmi2": {
        "code": "0051",
        "aliases": [
            # English
            "hdmi 2", "hdmi2", "hdmi two", "input 2", "source 2", "source two",
            "input two", "switch to hdmi 2",
            # French
            "hdmi 2", "entrée 2", "source 2", "mettre sur hdmi 2",
            # Darija / Arabic
            "hdmi جوج", "hdmi اثنين", "hdmi 2",
            "المصدر الثاني", "الثاني", "المصدر 2",
            "غيّر لـ hdmi جوج", "hdmi الثاني",
        ],
        "category": "Input",
        "description": "Switch to HDMI 2",
    },
    "hdmi3": {
        "code": "0052",
        "aliases": [
            # English
            "hdmi 3", "hdmi3", "hdmi three", "input 3", "source 3", "source three",
            "input three", "switch to hdmi 3",
            # French
            "hdmi 3", "entrée 3", "source 3", "mettre sur hdmi 3",
            # Darija / Arabic
            "hdmi تلاتة", "hdmi ثلاثة", "hdmi 3",
            "المصدر الثالث", "الثالث", "المصدر 3",
            "غيّر لـ hdmi تلاتة", "hdmi الثالث",
        ],
        "category": "Input",
        "description": "Switch to HDMI 3",
    },
    "hdmi4": {
        "code": "0053",
        "aliases": [
            # English
            "hdmi 4", "hdmi4", "hdmi four", "input 4", "source 4", "source four",
            "input four", "switch to hdmi 4",
            # French
            "hdmi 4", "entrée 4", "source 4", "mettre sur hdmi 4",
            # Darija / Arabic
            "hdmi ربعة", "hdmi أربعة", "hdmi 4",
            "المصدر الرابع", "الرابع", "المصدر 4",
            "غيّر لـ hdmi ربعة", "hdmi الرابع",
        ],
        "category": "Input",
        "description": "Switch to HDMI 4",
    },

    # ── Digit Keys 0-9 ─────────────────────────
    "digit_0": {
        "code": "0040",
        "aliases": [
            "0", "zero", "channel 0",
            "zéro", "chaîne zéro",
            "صفر", "زيرو", "قناة صفر",
        ],
        "category": "Channel",
        "description": "Key 0",
    },
    "digit_1": {
        "code": "0041",
        "aliases": [
            "1", "one", "channel 1", "number one",
            "un", "chaîne un", "chaîne 1",
            "واحد", "قناة واحد", "قناة 1",
        ],
        "category": "Channel",
        "description": "Key 1",
    },
    "digit_2": {
        "code": "0042",
        "aliases": [
            "2", "two", "channel 2", "number two",
            "deux", "chaîne deux", "chaîne 2",
            "جوج", "اثنين", "قناة جوج", "قناة اثنين", "قناة 2",
        ],
        "category": "Channel",
        "description": "Key 2",
    },
    "digit_3": {
        "code": "0043",
        "aliases": [
            "3", "three", "channel 3", "number three",
            "trois", "chaîne trois", "chaîne 3",
            "تلاتة", "ثلاثة", "قناة تلاتة", "قناة ثلاثة", "قناة 3",
        ],
        "category": "Channel",
        "description": "Key 3",
    },
    "digit_4": {
        "code": "0044",
        "aliases": [
            "4", "four", "channel 4", "number four",
            "quatre", "chaîne quatre", "chaîne 4",
            "ربعة", "أربعة", "قناة ربعة", "قناة أربعة", "قناة 4",
        ],
        "category": "Channel",
        "description": "Key 4",
    },
    "digit_5": {
        "code": "0045",
        "aliases": [
            "5", "five", "channel 5", "number five",
            "cinq", "chaîne cinq", "chaîne 5",
            "خمسة", "قناة خمسة", "قناة 5",
        ],
        "category": "Channel",
        "description": "Key 5",
    },
    "digit_6": {
        "code": "0046",
        "aliases": [
            "6", "six", "channel 6", "number six",
            "six", "chaîne six", "chaîne 6",
            "ستة", "قناة ستة", "قناة 6",
        ],
        "category": "Channel",
        "description": "Key 6",
    },
    "digit_7": {
        "code": "0047",
        "aliases": [
            "7", "seven", "channel 7", "number seven",
            "sept", "chaîne sept", "chaîne 7",
            "سبعة", "قناة سبعة", "قناة 7",
        ],
        "category": "Channel",
        "description": "Key 7",
    },
    "digit_8": {
        "code": "0048",
        "aliases": [
            "8", "eight", "channel 8", "number eight",
            "huit", "chaîne huit", "chaîne 8",
            "تمانية", "ثمانية", "قناة تمانية", "قناة ثمانية", "قناة 8",
        ],
        "category": "Channel",
        "description": "Key 8",
    },
    "digit_9": {
        "code": "0049",
        "aliases": [
            "9", "nine", "channel 9", "number nine",
            "neuf", "chaîne neuf", "chaîne 9",
            "تسعة", "قناة تسعة", "قناة 9",
        ],
        "category": "Channel",
        "description": "Key 9",
    },
}

# ─────────────────────────────────────────────
#  AC COMMANDS DATABASE  (Darija + FR + EN)
# ─────────────────────────────────────────────
AC_COMMANDS = {
    # ── Power ──────────────────────────────────
    "open": {
        "code": "1001",
        "aliases": [
            "open", "turn on", "start", "on", "activate",
            "allumer le climatiseur", "allumer le clim", "démarrer le clim",
            "mettre en marche le clim",
            # Darija — AC is "كليم" (from French "clim")
            # bare short forms — used when trigger word already stripped
            "شعل", "شعلو", "ولع", "ولعو", "خدم", "خدمو",
            "افتح", "افتحو", "فتح", "شغّل", "شغّلو",
            "حيّه", "مشّيه", "ديرو يخدم",
            # with device name
            "شعل الكليم", "شعلو الكليم", "شعل الكليماتيزور",
            "ولع الكليم", "ولعو الكليم",
            "خدم الكليم", "خدمو الكليم", "دير الكليم يخدم",
            "شغّل الكليم", "شغّلو الكليم",
            "افتح الكليم", "فتح الكليم",
            "مشّي الكليم", "ديماري الكليم",
            "حيّ الكليم", "شمّع الكليم",
        ],
        "category": "Power",
        "description": "Turn ON the Air Conditioner",
    },
    "close": {
        "code": "1002",
        "aliases": [
            "close", "turn off", "shutdown", "off", "stop",
            "éteindre le clim", "éteindre le climatiseur", "arrêter le clim",
            # bare short forms
            "طفي", "طفيو", "حبس", "حبسو", "سكر", "سكرو",
            "قفل", "قفلو", "بطّل", "بطّلو", "وقّف",
            "أطفيه", "طفيه", "حبسه",
            # with device name
            "طفي الكليم", "طفيو الكليم",
            "حبس الكليم", "حبسو الكليم",
            "سكر الكليم", "سكرو الكليم",
            "قفل الكليم", "قفلو الكليم",
            "بطّل الكليم", "بطّلو الكليم",
            "وقّف الكليم",
        ],
        "category": "Power",
        "description": "Turn OFF the Air Conditioner",
    },

    # ── Modes ──────────────────────────────────
    "cool": {
        "code": "1003",
        "aliases": [
            "cool", "cooling", "cold", "freeze", "air conditioning",
            "froid", "mode froid", "refroidir", "mettre en froid", "fraîcheur",
            # Darija — cold = برد / كلاص / قلاصي
            "برد", "بردني", "ردها برد",
            "كلاص", "ردها كلاص", "وضع الكلاص",
            "قلاصي", "وضع القلاص",
            "برد الحال", "برد الدار", "برّد",
            "حال بارد", "تكييف بارد",
            "وضع البرودة", "mode برد",
            "الصيف جاي برد", "سقسيني",
        ],
        "category": "Mode",
        "description": "Switch to Cooling Mode",
    },
    "heat": {
        "code": "1004",
        "aliases": [
            "heat", "heating", "warm", "hot", "heater",
            "chaud", "chauffage", "chauffer", "mode chaud", "mettre en chaud",
            # Darija — hot = سخن / دافي
            "سخن", "سخّن", "دافي", "دفيني",
            "سخانة", "ردها سخانة", "وضع السخانة",
            "سخن الدار", "سخن الحال",
            "حال سخون", "تدفئة",
            "وضع التدفئة", "mode سخانة",
            "شتا برد درك", "دفيني شوية",
        ],
        "category": "Mode",
        "description": "Switch to Heating Mode",
    },
    "dry": {
        "code": "1011",
        "aliases": [
            "dry", "dehumidify", "dry mode", "humidity",
            "sécher", "mode séchage", "déshumidifier",
            "جاف", "تجفيف", "وضع التجفيف", "نشّف الهواء",
        ],
        "category": "Mode",
        "description": "Dry Mode (Dehumidify)",
    },
    "fan only": {
        "code": "1012",
        "aliases": [
            "fan only", "ventilation", "fan mode", "only fan", "air flow",
            "ventilation seulement", "mode ventilation", "souffler",
            "ريح بس", "فقط الريح", "مروحة بس",
            "دير الريح بلا برد", "وضع المروحة",
            "هواء بلا تكييف",
        ],
        "category": "Mode",
        "description": "Fan Only Mode",
    },
    "auto": {
        "code": "1013",
        "aliases": [
            "auto", "automatic", "auto mode", "smart mode",
            "automatique", "mode auto", "mode intelligent",
            "أوتو", "تلقائي", "وضع أوتو", "خليه يدير وحده",
            "وضع تلقائي", "دير أوتو",
        ],
        "category": "Mode",
        "description": "Auto Mode",
    },

    # ── Temperature ────────────────────────────
    "temp up": {
        "code": "1005",
        "aliases": [
            "temperature up", "warmer", "increase temperature", "hotter", "temp up",
            "plus chaud", "augmenter la température", "monter la température",
            # bare forms
            "زيد", "زيدو", "طلع", "أعلى",
            # with context
            "زيد الدرجة", "زيد في الدرجة", "زيدلو درجة",
            "طلع الدرجة", "طلع شوية",
            "سخنو أكثر", "زيد في الحرارة",
            "أكثر سخانة", "الدرجة أعلى",
            "درجة فوق",
        ],
        "category": "Temperature",
        "description": "Increase Temperature",
    },
    "temp down": {
        "code": "1006",
        "aliases": [
            "temperature down", "cooler", "decrease temperature", "colder", "temp down",
            "moins chaud", "baisser la température", "descendre la température",
            # bare forms
            "نقص", "نقصو", "هبط", "أقل",
            # with context
            "نقص الدرجة", "نقص في الدرجة", "نقصلو درجة",
            "هبط الدرجة", "هبط شوية",
            "بردو أكثر", "نقص في الحرارة",
            "أقل سخانة", "الدرجة أقل",
            "درجة تحت",
        ],
        "category": "Temperature",
        "description": "Decrease Temperature",
    },

    # ── Fan ────────────────────────────────────
    "fan up": {
        "code": "1007",
        "aliases": [
            "fan up", "increase fan", "stronger wind", "faster fan", "more wind",
            "plus de vent", "augmenter la ventilation", "ventilateur plus fort",
            # bare forms (fan = ريح / فنتيلاتور)
            "قوي", "قوّيها",
            # with context
            "زيد الريح", "زيد في الريح", "قوي الريح",
            "زيد الفنتيلاتور", "قوي الفنتيلاتور",
            "ريح أكثر", "أكثر هواء", "قوّيها",
            "الريح ضعيفة", "زيدو الريح",
        ],
        "category": "Fan",
        "description": "Increase Fan Speed",
    },
    "fan down": {
        "code": "1008",
        "aliases": [
            "fan down", "decrease fan", "weaker wind", "slower fan", "less wind",
            "moins de vent", "baisser la ventilation", "ventilateur moins fort",
            # bare forms
            "هادي", "هاديها",
            # with context
            "نقص الريح", "نقص في الريح", "هادي الريح",
            "نقص الفنتيلاتور", "هادي الفنتيلاتور",
            "ريح أقل", "أقل هواء", "هاديها",
            "الريح قوية برك", "نقصو الريح",
        ],
        "category": "Fan",
        "description": "Decrease Fan Speed",
    },

    # ── Extras ─────────────────────────────────
    "swing": {
        "code": "1009",
        "aliases": [
            "swing", "rotate", "oscillate", "move flaps",
            "oscillation", "balancer", "faire bouger",
            "دوّر", "سوانغ", "حرك الريش",
            "خليه يدور", "دوّر الريش",
            "تأرجح", "سويينغ",
        ],
        "category": "Motion",
        "description": "Toggle Swing Mode",
    },
    "silent": {
        "code": "1010",
        "aliases": [
            "silent", "quiet mode", "night mode", "no noise", "sleep mode",
            "silencieux", "mode nuit", "mode silencieux", "mode veille",
            "سكات", "صامت", "وضع الليل",
            "مايديرش الحس", "بلا ضجيج",
            "هادي", "نقص الضجيج", "ما نسمعش منه",
            "وضع النوم", "نايمين",
        ],
        "category": "Mode",
        "description": "Toggle Silent / Night Mode",
    },
    "turbo": {
        "code": "1014",
        "aliases": [
            "turbo", "turbo mode", "maximum", "max power", "full power",
            "turbo mode", "puissance maximale", "plein régime",
            "توربو", "أقصى قوة", "ماكسيمو",
            "على قد ما تقدر", "قوة قصوى",
            "بلا حدود", "برد بسرعة",
        ],
        "category": "Mode",
        "description": "Turbo / Max Power Mode",
    },
    "timer": {
        "code": "1015",
        "aliases": [
            "timer", "set timer", "schedule off", "auto off", "time",
            "minuterie", "programmer", "mettre un timer",
            "تايمر", "وقت", "اضبط التايمر",
            "حبس بعد ساعة", "حبس بعد وقت", "برمجة",
        ],
        "category": "Timer",
        "description": "Set Timer",
    },

    # ── Fan Speed Presets ──────────────────────
    "fan_auto": {
        "code": "1016",
        "aliases": [
            # English
            "fan auto", "fan automatic", "auto fan", "automatic fan speed",
            "auto speed", "fan auto mode",
            # French
            "ventilateur auto", "vitesse auto", "ventilateur automatique",
            "mode ventilation automatique", "vitesse automatique",
            # Darija / Arabic
            "ريح أوتو", "الريح التلقائي", "أوتو للريح",
            "الريح أوتو", "وضع الريح التلقائي",
            "الفنتيلاتور أوتو", "سرعة الريح تلقائية",
            "خليها تدير وحدها", "أوتو فان",
        ],
        "category": "Fan",
        "description": "Fan Auto Speed",
    },
    "fan_low": {
        "code": "1017",
        "aliases": [
            # English
            "fan low", "low fan", "low speed", "slow fan", "minimum fan",
            "fan minimum", "quiet fan", "soft fan",
            # French
            "ventilateur faible", "vitesse basse", "faible ventilation",
            "petit ventilateur", "vitesse minimale", "souffle faible",
            # Darija / Arabic
            "ريح هادية", "ريح خفيفة", "ريح واطة",
            "فنتيلاتور هادي", "ريح قليلة", "ريح صغيرة",
            "ريح وطية", "بطيء الريح", "ريح هادي",
            "هادي الفنتيلاتور", "ريح أقل وحدة",
        ],
        "category": "Fan",
        "description": "Fan Low Speed",
    },
    "fan_med": {
        "code": "1018",
        "aliases": [
            # English
            "fan medium", "fan med", "medium fan", "mid fan", "medium speed",
            "medium fan speed", "moderate fan",
            # French
            "ventilateur moyen", "vitesse moyenne", "moyenne ventilation",
            "souffle moyen", "ventilation modérée",
            # Darija / Arabic
            "ريح متوسطة", "موسطة ريح", "ريح موسطة",
            "فنتيلاتور متوسط", "ريح وسطية",
            "ريح ميديوم", "سرعة متوسطة للريح",
            "الريح لا قوية لا هادية",
        ],
        "category": "Fan",
        "description": "Fan Medium Speed",
    },
    "fan_high": {
        "code": "1019",
        "aliases": [
            # English
            "fan high", "high fan", "high speed", "fast fan", "maximum fan",
            "fan maximum", "max fan", "strong fan", "powerful fan", "turbo fan",
            # French
            "ventilateur fort", "vitesse élevée", "grande vitesse",
            "ventilateur maximum", "souffle fort", "ventilation forte",
            # Darija / Arabic
            "ريح قوية", "ريح عالية", "فنتيلاتور قوي",
            "قوّي الريح", "الريح بالقصوى", "ريح كثيرة",
            "ريح هاي", "سرعة عالية للريح", "أقصى ريح",
            "ريح على أقصى سرعة",
        ],
        "category": "Fan",
        "description": "Fan High Speed",
    },

    # ── Specific Temperature Presets ───────────
    "temp_16": {
        "code": "1020",
        "aliases": [
            "16", "16 degrees", "set to 16", "temperature 16", "sixteen",
            "seize degrés", "mettre à 16", "température seize", "16 degrés",
            "ستاش", "ستة عشر", "ستاش درجة", "حط عند ستاش",
            "ستاش درجة مئوية", "درجة ستاش",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 16°C",
    },
    "temp_18": {
        "code": "1021",
        "aliases": [
            "18", "18 degrees", "set to 18", "temperature 18", "eighteen",
            "dix-huit degrés", "mettre à 18", "température dix-huit", "18 degrés",
            "تمنتاش", "ثمانية عشر", "تمنتاش درجة", "حط عند تمنتاش",
            "تمنتاش درجة مئوية", "درجة تمنتاش",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 18°C",
    },
    "temp_20": {
        "code": "1022",
        "aliases": [
            "20", "20 degrees", "set to 20", "temperature 20", "twenty",
            "vingt degrés", "mettre à 20", "température vingt", "20 degrés",
            "عشرين", "عشرين درجة", "حط عند عشرين",
            "عشرين درجة مئوية", "درجة عشرين",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 20°C",
    },
    "temp_22": {
        "code": "1023",
        "aliases": [
            "22", "22 degrees", "set to 22", "temperature 22", "twenty two",
            "vingt-deux degrés", "mettre à 22", "température vingt-deux", "22 degrés",
            "جوج وعشرين", "اثنين وعشرين", "درجة جوج وعشرين",
            "حط عند جوج وعشرين", "اثنين وعشرين درجة",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 22°C",
    },
    "temp_24": {
        "code": "1024",
        "aliases": [
            "24", "24 degrees", "set to 24", "temperature 24", "twenty four",
            "vingt-quatre degrés", "mettre à 24", "température vingt-quatre", "24 degrés",
            "ربعة وعشرين", "أربعة وعشرين", "درجة ربعة وعشرين",
            "حط عند ربعة وعشرين", "أربعة وعشرين درجة",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 24°C",
    },
    "temp_26": {
        "code": "1025",
        "aliases": [
            "26", "26 degrees", "set to 26", "temperature 26", "twenty six",
            "vingt-six degrés", "mettre à 26", "température vingt-six", "26 degrés",
            "ستة وعشرين", "درجة ستة وعشرين",
            "حط عند ستة وعشرين", "ستة وعشرين درجة",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 26°C",
    },
    "temp_28": {
        "code": "1026",
        "aliases": [
            "28", "28 degrees", "set to 28", "temperature 28", "twenty eight",
            "vingt-huit degrés", "mettre à 28", "température vingt-huit", "28 degrés",
            "تمانية وعشرين", "ثمانية وعشرين", "درجة تمانية وعشرين",
            "حط عند تمانية وعشرين", "ثمانية وعشرين درجة",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 28°C",
    },
    "temp_30": {
        "code": "1027",
        "aliases": [
            "30", "30 degrees", "set to 30", "temperature 30", "thirty",
            "trente degrés", "mettre à 30", "température trente", "30 degrés",
            "تلاتين", "ثلاثين", "تلاتين درجة", "ثلاثين درجة",
            "حط عند تلاتين", "درجة تلاتين",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 30°C",
    },
    "temp_17": {
        "code": "1028",
        "aliases": [
            "17", "17 degrees", "set to 17", "temperature 17", "seventeen",
            "dix-sept degrés", "mettre à 17", "17 degrés",
            "سبعتاش", "سبعة عشر", "سبعتاش درجة", "حط عند سبعتاش",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 17°C",
    },
    "temp_19": {
        "code": "1029",
        "aliases": [
            "19", "19 degrees", "set to 19", "temperature 19", "nineteen",
            "dix-neuf degrés", "mettre à 19", "19 degrés",
            "تسعتاش", "تسعة عشر", "تسعتاش درجة", "حط عند تسعتاش",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 19°C",
    },
    "temp_21": {
        "code": "1030",
        "aliases": [
            "21", "21 degrees", "set to 21", "temperature 21", "twenty one",
            "vingt et un degrés", "mettre à 21", "21 degrés",
            "واحد وعشرين", "درجة واحد وعشرين", "حط عند واحد وعشرين",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 21°C",
    },
    "temp_23": {
        "code": "1031",
        "aliases": [
            "23", "23 degrees", "set to 23", "temperature 23", "twenty three",
            "vingt-trois degrés", "mettre à 23", "23 degrés",
            "تلاتة وعشرين", "ثلاثة وعشرين", "درجة تلاتة وعشرين",
            "حط عند تلاتة وعشرين",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 23°C",
    },
    "temp_25": {
        "code": "1032",
        "aliases": [
            "25", "25 degrees", "set to 25", "temperature 25", "twenty five",
            "vingt-cinq degrés", "mettre à 25", "25 degrés",
            "خمسة وعشرين", "درجة خمسة وعشرين", "حط عند خمسة وعشرين",
            "خمسة وعشرين درجة",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 25°C",
    },
    "temp_27": {
        "code": "1033",
        "aliases": [
            "27", "27 degrees", "set to 27", "temperature 27", "twenty seven",
            "vingt-sept degrés", "mettre à 27", "27 degrés",
            "سبعة وعشرين", "درجة سبعة وعشرين", "حط عند سبعة وعشرين",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 27°C",
    },
    "temp_29": {
        "code": "1034",
        "aliases": [
            "29", "29 degrees", "set to 29", "temperature 29", "twenty nine",
            "vingt-neuf degrés", "mettre à 29", "29 degrés",
            "تسعة وعشرين", "درجة تسعة وعشرين", "حط عند تسعة وعشرين",
        ],
        "category": "Temperature",
        "description": "Set Temperature to 29°C",
    },
}

# ─────────────────────────────────────────────
#  LEARNED ALIAS LOADER
# ─────────────────────────────────────────────
def load_learned_aliases(tv_commands, ac_commands):
    """
    Load learned_aliases.json and:
      1. Inject user-recorded command aliases into TV/AC command dicts
         (before matchers encode embeddings — gives exact-match = 100%).
      2. Inject user-recorded trigger words into _TRIGGERS so the
         device detector recognises how Whisper hears the user's voice.
    Called once at startup inside run().
    """
    import json
    from pathlib import Path
    path = Path(__file__).parent / "learned_aliases.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Training] Warning: could not load learned_aliases.json — {e}")
        return

    alias_count = 0
    for device_key, commands in [("TV", tv_commands), ("AC", ac_commands)]:
        for cmd_key, learned in data.get(device_key, {}).items():
            if cmd_key not in commands:
                continue
            existing_lower = {a.lower() for a in commands[cmd_key]["aliases"]}
            for alias in learned:
                if alias.lower() not in existing_lower:
                    commands[cmd_key]["aliases"].append(alias)
                    existing_lower.add(alias.lower())
                    alias_count += 1

    trigger_count = 0
    for device in ("TV", "AC"):
        existing_lower = [t.lower() for t in _TRIGGERS[device]]
        for t in data.get("_triggers", {}).get(device, []):
            if t.lower() not in existing_lower:
                _TRIGGERS[device].append(t.lower())
                existing_lower.append(t.lower())
                trigger_count += 1

    if alias_count or trigger_count:
        print(f"[Training] Loaded {alias_count} alias(es) + "
              f"{trigger_count} trigger word(s) from learned_aliases.json\n")


# ─────────────────────────────────────────────
#  ALIAS MAP BUILDER
# ─────────────────────────────────────────────
def build_alias_map(commands):
    alias_map = {}
    for cmd_name, cmd_data in commands.items():
        for alias in cmd_data["aliases"]:
            alias_map[alias.lower()] = cmd_name
    return alias_map


# ─────────────────────────────────────────────
#  DEVICE TRIGGER DETECTOR
# ─────────────────────────────────────────────
# These are the wake-words per device. Matched on the first word(s) of the utterance.
_TRIGGERS = {
    "TV": [
        # English / Latin
        "tv", "tivi", "tiv", "t v", "television", "tele", "télé",
        "the tv", "my tv", "le tv", "la tv",
        # Arabic standard
        "تيفي", "التيفي", "تي في", "تيف",
        "تلفاز", "التلفاز", "تلفزيون", "التلفزيون",
        "التلفاز", "التلفة",
        # Whisper Darija misreadings of "تيفي"
        "تبي", "تيبي", "ديبي", "تبية", "تيبية",
        "تيبه", "تيبة", "تبه", "تيبو",
    ],
    "AC": [
        # English / Latin
        "ac", "a c", "clim", "clime", "cleem", "clima",
        "climatiseur", "le climatiseur", "la clim", "la clime",
        "air conditioner", "aircon", "air con",
        "climaseur", "climaseur", "climatizor", "climatizeur",
        # Arabic standard
        "كليم", "الكليم", "كليماتيزور", "الكليماتيزور",
        "المكيف", "مكيف", "التكييف", "تكييف",
        # Whisper Darija misreadings of "كليم"
        "كلم", "كيلم", "كليمة", "كليماتيزر",
    ],
}


# ─────────────────────────────────────────────
#  NUMBER NORMALISATION
# ─────────────────────────────────────────────
# Maps spoken number words (all 4 languages) → digit string.
# Applied before command matching so "AC vingt-cinq" → "ac 25" → temp_25.
_NUMBER_WORDS = {
    # ── English ──────────────────────────────
    "zero":"0","one":"1","two":"2","three":"3","four":"4",
    "five":"5","six":"6","seven":"7","eight":"8","nine":"9",
    "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19",
    "twenty":"20",
    "twenty one":"21","twenty-one":"21",
    "twenty two":"22","twenty-two":"22",
    "twenty three":"23","twenty-three":"23",
    "twenty four":"24","twenty-four":"24",
    "twenty five":"25","twenty-five":"25",
    "twenty six":"26","twenty-six":"26",
    "twenty seven":"27","twenty-seven":"27",
    "twenty eight":"28","twenty-eight":"28",
    "twenty nine":"29","twenty-nine":"29",
    "thirty":"30",
    # ── French ───────────────────────────────
    "zéro":"0","un":"1","deux":"2","trois":"3","quatre":"4",
    "cinq":"5","sept":"7","huit":"8","neuf":"9","dix":"10",
    "seize":"16","dix-sept":"17","dix-huit":"18","dix-neuf":"19",
    "vingt":"20",
    "vingt et un":"21","vingt-et-un":"21",
    "vingt deux":"22","vingt-deux":"22",
    "vingt trois":"23","vingt-trois":"23",
    "vingt quatre":"24","vingt-quatre":"24",
    "vingt cinq":"25","vingt-cinq":"25",
    "vingt six":"26","vingt-six":"26",
    "vingt sept":"27","vingt-sept":"27",
    "vingt huit":"28","vingt-huit":"28",
    "vingt neuf":"29","vingt-neuf":"29",
    "trente":"30",
    # ── Arabic MSA ────────────────────────────
    "صفر":"0","واحد":"1","اثنين":"2","ثلاثة":"3","أربعة":"4",
    "خمسة":"5","ستة":"6","سبعة":"7","ثمانية":"8","تسعة":"9",
    "ستة عشر":"16","سبعة عشر":"17","ثمانية عشر":"18","تسعة عشر":"19",
    "عشرون":"20","عشرين":"20",
    "واحد وعشرون":"21","واحد وعشرين":"21",
    "اثنان وعشرون":"22","اثنين وعشرين":"22",
    "ثلاثة وعشرون":"23","ثلاثة وعشرين":"23",
    "أربعة وعشرون":"24","أربعة وعشرين":"24",
    "خمسة وعشرون":"25","خمسة وعشرين":"25",
    "ستة وعشرون":"26","ستة وعشرين":"26",
    "سبعة وعشرون":"27","سبعة وعشرين":"27",
    "ثمانية وعشرون":"28","ثمانية وعشرين":"28",
    "تسعة وعشرون":"29","تسعة وعشرين":"29",
    "ثلاثون":"30","ثلاثين":"30",
    # ── Algerian Darija ───────────────────────
    "جوج":"2","ربعة":"4","تمانية":"8",
    "ستاش":"16","سبعتاش":"17","تمنتاش":"18","تسعتاش":"19",
    "واحد وعشرين":"21",
    "جوج وعشرين":"22","اثنين وعشرين":"22",
    "تلاتة وعشرين":"23","ثلاثة وعشرين":"23",
    "ربعة وعشرين":"24","أربعة وعشرين":"24",
    "خمسة وعشرين":"25",
    "ستة وعشرين":"26",
    "سبعة وعشرين":"27",
    "تمانية وعشرين":"28","ثمانية وعشرين":"28",
    "تسعة وعشرين":"29",
    "تلاتين":"30","تلاتة وثلاثين":"30",
}


def normalize_numbers(text: str) -> str:
    """Replace spoken number words with digit strings (longest match first)."""
    result = text.lower()
    for phrase in sorted(_NUMBER_WORDS, key=len, reverse=True):
        if phrase in result:
            result = result.replace(phrase, _NUMBER_WORDS[phrase])
    return result


# ─────────────────────────────────────────────
#  COMPREHENSIVE NATURAL-LANGUAGE ALIASES
#  (merged into TV/AC commands at startup)
# ─────────────────────────────────────────────
_EXTRA_ALIASES = {
    "TV": {
        "open": [
            # English
            "put the tv on","start the tv","i want to watch tv",
            "switch the tv on","fire up the tv","tv please",
            "power up the tv","bring up the tv","turn the television on",
            # French
            "mets la télé","allume la télé","démarre la télé",
            "ouvre la télé","télé s'il te plaît","mets la télévision",
            "allume le téléviseur","lance la télé",
            # Arabic MSA
            "شغّل التلفاز","افتح التلفزيون","ابدأ التلفاز","فعّل التلفاز",
            # Darija
            "بغيت نشعل التيفي","دير التيفي","شعّل التيفي",
            "فتحلي التيفي","خدّم التيفي","حيّ التيفي",
            "ولّع التيفي","شمّع التيفي",
        ],
        "close": [
            # English
            "shut off the tv","power down the tv","turn the tv off",
            "kill the tv","tv off","stop the tv","tv sleep",
            "turn off the television","switch off the tv",
            # French
            "éteins la télé","coupe la télé","ferme la télé",
            "arrête la télé","télé off","coupe le téléviseur",
            "éteins le téléviseur",
            # Arabic MSA
            "أطفئ التلفاز","أغلق التلفزيون","أوقف التلفاز",
            # Darija
            "طفيلي التيفي","حبس التيفي","سكّر التيفي",
            "بطّل التيفي","قفل التيفي","وقّف التيفي",
        ],
        "volume up": [
            # English
            "make it louder","raise the volume","increase the sound",
            "turn the volume up","more volume","louder please",
            "i can't hear","can't hear","speak up","too quiet",
            "crank it up","bump up the volume","volume higher",
            # French
            "monte le son","augmente le volume","plus fort s'il te plaît",
            "je n'entends pas","on n'entend rien","son plus fort",
            # Arabic MSA
            "ارفع الصوت","زيادة الصوت","اجعله أعلى",
            # Darija
            "زيدلي الصوت","قوّيلي الصوت","الصوت ما نسمعش",
            "ارفعلي الصوت","زيد في الصوت بزاف",
        ],
        "volume down": [
            # English
            "make it quieter","lower the volume","decrease the sound",
            "turn it down","too loud","not so loud","quieter please",
            "volume lower","softer please","reduce the sound",
            # French
            "baisse le son","moins fort s'il te plaît","diminue le volume",
            "c'est trop fort","réduire le son",
            # Arabic MSA
            "اخفض الصوت","خفّض الصوت","قلّل الصوت",
            # Darija
            "نقصلي الصوت","هادلي الصوت","الصوت عالي بزاف",
            "خففلي الصوت","صوت أقل شوية",
        ],
        "mute": [
            # English
            "no sound","silence please","shut it up","shush",
            "i need quiet","be quiet","quiet please","stop the sound",
            # French
            "coupe le son","silence s'il te plaît","sourdine s'il te plaît",
            "pas de son","plus de son",
            # Arabic MSA
            "اسكت","بدون صوت","اقطع الصوت",
            # Darija
            "بلا صوت","سكاتو","ما نبغيش نسمع",
            "اقطعلي الصوت","كتّملي الصوت",
        ],
        "channel up": [
            # English
            "go to next channel","next tv channel","flip up","channel forward",
            "switch to next channel","next one please","forward channel",
            # French
            "prochaine chaîne","chaîne d'après","passe à la chaîne suivante",
            # Darija
            "القناة الجاية","بدّل للجاية","شوف اش كاين",
            "روح للقناة الجاية","عدّي قناة",
        ],
        "channel down": [
            # English
            "go back channel","previous tv channel","flip down","channel back",
            "go to previous channel","last channel","prior channel",
            # French
            "chaîne précédente","chaîne d'avant","revenir à la chaîne précédente",
            # Darija
            "ارجع للقناة","القناة الفايتة","راجع قناة",
        ],
        "home": [
            # English
            "go to home","take me home","main screen","go to main menu",
            "home page","back to start","start screen",
            # French
            "retour à l'accueil","page principale","écran principal",
            "retourne à l'accueil",
            # Darija
            "روح للبداية","الواجهة الرئيسية","ارجع للرئيسية",
            "روح للمنيو",
        ],
        "settings": [
            # English
            "open settings","go to settings","tv settings","configure",
            "open preferences","open options","show settings",
            # French
            "ouvre les paramètres","va dans les réglages",
            "ouvre les options","affiche les paramètres",
            # Darija
            "روح للإعدادات","افتح الإعدادات","حل الإعدادات",
            "الضبط","ضبط التيفي",
        ],
        "input": [
            # English
            "change input","switch source","change source","external input",
            "change hdmi","switch to hdmi","select input",
            # French
            "changer d'entrée","changer la source","sélectionner l'entrée",
            # Darija
            "بدّل المصدر","غيّر المصدر","الكابل",
        ],
        "play": [
            "start playing","resume playback","play video","continue watching",
            "lance la lecture","reprendre la lecture",
            "شغّل الفيديو","استمر في التشغيل",
        ],
        "pause": [
            "stop for a second","freeze it","hold on","wait a moment",
            "mettre en pause","gèle l'image",
            "وقّف لحظة","جمّد","فريّز",
        ],
        "netflix": [
            "open netflix","put on netflix","i want netflix","launch netflix",
            "go to netflix","watch netflix","netflix app",
            "ouvre netflix","mets netflix","lance netflix",
            "بغيت نيتفليكس","حل نيتفليكس","افتح نيتفليكس",
        ],
        "youtube": [
            "open youtube","put on youtube","i want youtube","launch youtube",
            "go to youtube","watch youtube","youtube please",
            "ouvre youtube","mets youtube",
            "بغيت يوتيوب","حل يوتيوب","افتح يوتيوب",
        ],
        "subtitles": [
            "turn on subtitles","show subtitles","enable captions",
            "i want subtitles","add subtitles","subtitle on",
            "affiche les sous-titres","active les sous-titres",
            "بغيت ترجمة","حط الترجمة","شعل الترجمة",
        ],
        "hdmi1": [
            "switch to hdmi 1","go to hdmi 1","select hdmi 1",
            "entrée hdmi 1","source hdmi 1",
            "روح لـ hdmi 1","بدّل لـ hdmi واحد",
        ],
        "hdmi2": [
            "switch to hdmi 2","go to hdmi 2","select hdmi 2",
            "entrée hdmi 2","source hdmi 2",
            "روح لـ hdmi 2","بدّل لـ hdmi جوج",
        ],
        "hdmi3": [
            "switch to hdmi 3","go to hdmi 3","select hdmi 3",
            "entrée hdmi 3","روح لـ hdmi 3",
        ],
        "hdmi4": [
            "switch to hdmi 4","go to hdmi 4","select hdmi 4",
            "entrée hdmi 4","روح لـ hdmi 4",
        ],
        "search": [
            "search for something","find a show","look something up",
            "cherche quelque chose","lancer une recherche",
            "دور على شيء","ابحث عن",
        ],
        "guide": [
            "show me the guide","what's on tv","program schedule",
            "show program guide","what's playing",
            "affiche le guide","qu'est-ce qu'il y a",
            "شو كاين اليوم","البرامج",
        ],
        "info": [
            "what is this","show info","what are we watching",
            "tell me about this","programme info",
            "qu'est-ce qu'on regarde","infos sur le programme",
            "شو هذا البرنامج","عطيني معلومات",
        ],
        "live": [
            "live tv","watch live","go to live","direct tv",
            "television en direct","regarder en direct",
            "التلفزيون المباشر","خدني للمباشر",
        ],
        "recent": [
            "recently watched","watch history","show history",
            "what did i watch","last channels",
            "historique","mes dernières chaînes",
            "اللي شفت من قبل","التاريخ",
        ],
    },
    "AC": {
        "open": [
            # English
            "start the ac","put on the ac","turn the ac on",
            "ac on please","i want the ac","start the air conditioner",
            "switch on the ac","activate the ac","ac start",
            # French
            "démarre la clim","mets la clim","lance la clim",
            "allume la climatisation","mets le climatiseur en marche",
            "clim s'il te plaît","mets la clime",
            # Arabic MSA
            "شغّل المكيف","افتح التكييف","ابدأ المكيف",
            # Darija
            "شعّل الكليم","دير الكليم","خدّم الكليم",
            "بغيت الكليم","فتحلي الكليم","ولّع الكليم",
        ],
        "close": [
            # English
            "stop the ac","turn the ac off","ac off",
            "shut off the ac","i'm cold enough","stop the air conditioner",
            "switch off the ac","ac please stop",
            # French
            "éteins la clim","coupe la clim","arrête la clim",
            "la clim off","éteins le climatiseur","stoppe la clim",
            # Arabic MSA
            "أطفئ المكيف","أوقف التكييف","أغلق المكيف",
            # Darija
            "طفيلي الكليم","حبس الكليم","بطّل الكليم",
            "قفل الكليم","وقّف الكليم","سكّر الكليم",
        ],
        "cool": [
            # English
            "i'm hot","it's too hot","make it cool","cooling please",
            "set to cool","cooling mode","i need cool air",
            "it's burning here","so hot","cool me down",
            # French
            "il fait chaud","trop chaud","mets en froid",
            "mode refroidissement","mets en mode froid","il fait une chaleur",
            # Arabic MSA
            "حار جداً","الجو حار","اجعله بارداً","وضع التبريد",
            # Darija
            "حار بزاف","حالة حارة","برّد عليا","سقسيني",
            "الحال ساخن","برّد الدار","وضع البرودة",
        ],
        "heat": [
            # English
            "i'm cold","it's too cold","make it warm","heating please",
            "set to heat","heating mode","warm me up","heat up",
            # French
            "il fait froid","trop froid","mets en chaud",
            "mode chauffage","mets en mode chaud","il fait un froid",
            # Arabic MSA
            "بارد جداً","الجو بارد","اجعله دافئاً","وضع التدفئة",
            # Darija
            "بارد بزاف","الحال باردة","سخّن عليا","دفّيني",
            "الجو قارص","سخّن الدار",
        ],
        "dry": [
            "dehumidify please","remove humidity","dry the air",
            "mode séchage","déshumidifier",
            "نشّف الهواء","رطوبة عالية",
        ],
        "fan only": [
            "just the fan","fan without cooling","ventilation only",
            "juste le ventilateur","ventilation sans froid",
            "الريح بس","مروحة بس","هواء بدون تبريد",
        ],
        "auto": [
            "auto mode please","let it decide","smart mode",
            "mode automatique","laisse décider",
            "خليه يدير وحده","وضع تلقائي",
        ],
        "temp up": [
            "increase temperature","make it warmer","higher temperature",
            "a bit warmer","raise the temperature",
            "augmente la température","plus chaud",
            "زيد الحرارة","ارفع الدرجة","درجة أعلى",
            "زيدلي درجة","أكثر دفئاً",
        ],
        "temp down": [
            "decrease temperature","make it cooler","lower temperature",
            "a bit cooler","reduce the temperature",
            "baisse la température","moins chaud",
            "نقص الحرارة","اخفض الدرجة","درجة أقل",
            "نقصلي درجة","أقل حرارة",
        ],
        "fan up": [
            "stronger fan","more wind","increase fan speed",
            "fan faster","blow harder",
            "plus de ventilation","ventilateur plus fort",
            "زيد الريح","قوّي الريح","ريح أكثر",
        ],
        "fan down": [
            "weaker fan","less wind","decrease fan speed",
            "fan slower","blow softer",
            "moins de ventilation","ventilateur moins fort",
            "نقص الريح","هادي الريح","ريح أقل",
        ],
        "fan_low": [
            "low fan speed","slow fan","quiet fan","gentle fan",
            "minimum fan speed","fan at minimum",
            "vitesse de ventilateur basse","petit ventilateur",
            "الريح الهادية","ريح بطيئة","فنتيلاتور هادي",
        ],
        "fan_med": [
            "medium fan speed","mid fan","moderate fan",
            "vitesse moyenne du ventilateur",
            "الريح المتوسطة","فنتيلاتور موسط",
        ],
        "fan_high": [
            "high fan speed","fast fan","maximum fan","full fan",
            "blow at max","strong fan please",
            "vitesse maximale du ventilateur","ventilateur à fond",
            "الريح القوية","فنتيلاتور على أقصى سرعة",
        ],
        "fan_auto": [
            "automatic fan speed","let fan decide","fan on auto",
            "vitesse automatique du ventilateur",
            "الريح التلقائية","الفنتيلاتور أوتو",
        ],
        "swing": [
            "turn on swing","rotate the fan","oscillate",
            "move the flaps","swing the air",
            "activer l'oscillation","faire osciller",
            "دوّر الريش","حرّك الريش","سوانغ",
        ],
        "silent": [
            "quiet mode","night mode","sleep mode","no noise",
            "mode nuit","mode silencieux","sans bruit",
            "وضع النوم","وضع الليل","بلا ضجيج","هادي",
        ],
        "turbo": [
            "maximum power","full blast","turbo mode","max cooling",
            "mode turbo","puissance maximale",
            "أقصى قوة","توربو","برد بسرعة قصوى",
        ],
        # ── Temperature presets — natural phrases ─────────────
        "temp_16": [
            "set temperature 16","put it on 16","16 degrees please",
            "mets à 16","température à 16",
            "حط عند 16","اضبط على 16",
        ],
        "temp_17": [
            "set temperature 17","17 degrees please",
            "mets à 17","حط عند 17",
        ],
        "temp_18": [
            "set temperature 18","18 degrees please",
            "mets à 18","حط عند 18",
        ],
        "temp_19": [
            "set temperature 19","19 degrees please",
            "mets à 19","حط عند 19",
        ],
        "temp_20": [
            "set temperature 20","20 degrees please","put it on 20",
            "mets à 20","température vingt",
            "حط عند 20","اضبط على 20","درجة عشرين",
        ],
        "temp_21": [
            "set temperature 21","21 degrees","mets à 21","حط عند 21",
        ],
        "temp_22": [
            "set temperature 22","22 degrees please","put it on 22",
            "mets à 22","حط عند 22",
        ],
        "temp_23": [
            "set temperature 23","23 degrees","mets à 23","حط عند 23",
        ],
        "temp_24": [
            "set temperature 24","24 degrees please","put it on 24",
            "mets à 24","حط عند 24",
        ],
        "temp_25": [
            "set temperature 25","25 degrees please","put it on 25",
            "mets à 25","température vingt-cinq",
            "حط عند 25","اضبط على 25","درجة خمسة وعشرين",
            "i want 25","clim 25","ac 25","25 s'il te plaît",
        ],
        "temp_26": [
            "set temperature 26","26 degrees","mets à 26","حط عند 26",
        ],
        "temp_27": [
            "set temperature 27","27 degrees","mets à 27","حط عند 27",
        ],
        "temp_28": [
            "set temperature 28","28 degrees","mets à 28","حط عند 28",
        ],
        "temp_29": [
            "set temperature 29","29 degrees","mets à 29","حط عند 29",
        ],
        "temp_30": [
            "set temperature 30","30 degrees please",
            "mets à 30","حط عند 30","اضبط على 30",
        ],
    },
}


def merge_extra_aliases():
    """
    Merge _EXTRA_ALIASES into TV_COMMANDS / AC_COMMANDS at startup,
    before the matchers encode embeddings.
    """
    mapping = {"TV": TV_COMMANDS, "AC": AC_COMMANDS}
    total = 0
    for device, cmd_dict in _EXTRA_ALIASES.items():
        commands = mapping[device]
        for cmd_key, extras in cmd_dict.items():
            if cmd_key not in commands:
                continue
            existing_lower = {a.lower() for a in commands[cmd_key]["aliases"]}
            for alias in extras:
                if alias.lower() not in existing_lower:
                    commands[cmd_key]["aliases"].append(alias)
                    existing_lower.add(alias.lower())
                    total += 1
    return total

def detect_device(spoken_text):
    """
    Scan the ENTIRE utterance for a device keyword — not just the prefix.
    "turn up the volume on the TV" works just as well as "TV volume up".

    Returns (device, command_remainder) or (None, None).

    Pass 1 — exact prefix match (fastest).
    Pass 2 — exact word match anywhere in the utterance.
    Pass 3 — fuzzy word match (≥75) anywhere; min word length 3
             to avoid short false-positives.
    The matched keyword is removed; everything else is the command.
    """
    text = spoken_text.lower().strip()
    words = text.split()

    for device, variants in _TRIGGERS.items():

        # ── Pass 1: exact prefix ─────────────────────────────────────────
        for v in variants:
            vl = v.lower()
            if text.startswith(vl + " ") or text == vl:
                return device, text[len(vl):].strip() or None

        # ── Pass 2: exact word anywhere ──────────────────────────────────
        for i, word in enumerate(words):
            bare = word.lstrip("ال")
            for v in variants:
                vl = v.lower()
                if word == vl or (bare and bare == vl):
                    remainder = " ".join(words[:i] + words[i + 1:]).strip()
                    return device, remainder or None

        # ── Pass 3: fuzzy anywhere (threshold 75, min len 4) ─────────────
        # Min length 4 prevents "كيف" (how) from matching "مكيف" (AC)
        # at 86% because it's a 3-char suffix of a 4-char word.
        # 3-char Darija variants (تبي, كلم, …) are already exact in the list.
        for i, word in enumerate(words):
            if len(word) < 4:
                continue
            bare = word.lstrip("ال")
            for v in variants:
                vl = v.lower()
                if len(vl) < 4:
                    continue
                if (fuzz.ratio(word, vl) >= 75
                        or (bare and len(bare) >= 4
                            and fuzz.ratio(bare, vl) >= 75)):
                    remainder = " ".join(words[:i] + words[i + 1:]).strip()
                    return device, remainder or None

    return None, None


# ─────────────────────────────────────────────
#  DEVICE COMMAND MATCHER
# ─────────────────────────────────────────────
class DeviceCommandMatcher:
    # model is shared between TV and AC instances — load once
    _shared_model = None

    @classmethod
    def _get_model(cls):
        if cls._shared_model is None:
            print("[NLP] Loading multilingual model (shared)...")
            cls._shared_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("[NLP] Model ready!\n")
        return cls._shared_model

    def __init__(self, device_name, commands):
        self.device_name = device_name
        self.commands = commands
        self.alias_map = build_alias_map(commands)
        self.all_aliases = list(self.alias_map.keys())

        model = DeviceCommandMatcher._get_model()
        print(f"[NLP] Encoding {device_name} aliases ({len(self.all_aliases)})...")
        self.alias_embeddings = model.encode(
            self.all_aliases, convert_to_tensor=True, show_progress_bar=False
        )
        print(f"[NLP] {device_name} ready!\n")

    def match(self, spoken_text, fuzzy_threshold=65, semantic_threshold=0.42):
        t0 = time.time()
        # Normalize spoken numbers ("vingt-cinq" → "25", "عشرين" → "20")
        text = normalize_numbers(spoken_text.lower().strip())
        result = {
            "device": self.device_name,
            "input": spoken_text,
            "command": None, "code": None,
            "method": None, "confidence": 0.0,
            "description": None, "category": None,
            "processing_time": 0,
        }

        # 1. Exact
        if text in self.alias_map:
            result.update(self._fill(self.alias_map[text], "exact", 1.0))
            result["processing_time"] = time.time() - t0
            return result

        # 2. Substring
        for alias, cmd in self.alias_map.items():
            if alias in text or text in alias:
                result.update(self._fill(cmd, "substring", 0.95))
                result["processing_time"] = time.time() - t0
                return result

        # 3. Fuzzy
        best_fuzzy, score, _ = process.extractOne(
            text, self.all_aliases, scorer=fuzz.WRatio
        )
        if score >= fuzzy_threshold:
            result.update(self._fill(self.alias_map[best_fuzzy], f"fuzzy {score:.0f}%", score / 100))
            result["processing_time"] = time.time() - t0
            return result

        # 4. Semantic NLP
        model = DeviceCommandMatcher._get_model()
        q_emb = model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.alias_embeddings)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= semantic_threshold:
            cmd = self.alias_map[self.all_aliases[best_idx]]
            result.update(self._fill(cmd, f"semantic {best_score:.2f}", best_score))
            result["processing_time"] = time.time() - t0
            return result

        result["method"] = "no match"
        result["processing_time"] = time.time() - t0
        return result

    def _fill(self, cmd, method, confidence):
        d = self.commands[cmd]
        return {
            "command": cmd, "code": d["code"],
            "method": method, "confidence": confidence,
            "description": d["description"], "category": d["category"],
        }


# ─────────────────────────────────────────────
#  OFFLINE STT  (faster-whisper, beam_size=1)
# ─────────────────────────────────────────────
class OfflineSTT:
    def __init__(self, model_size="tiny"):
        print(f"[STT] Loading Whisper '{model_size}' (offline, int8)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        # Pre-warm: eliminate cold-start latency on first real command
        print("[STT] Pre-warming model...")
        silence = np.zeros(16000, dtype=np.float32)
        list(self.model.transcribe(silence, language="ar", beam_size=1)[0])
        print("[STT] Ready!\n")

    def transcribe(self, audio: sr.AudioData) -> str | None:
        # Convert to float32 numpy — no disk I/O needed
        raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self.model.transcribe(
            samples,
            language="ar",          # Darija transcribed in Arabic script
            beam_size=1,            # greedy — 3-4x faster than beam_size=5
            vad_filter=True,        # skip silent segments
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text if text else None


# ─────────────────────────────────────────────
#  MICROPHONE LISTENER
# ─────────────────────────────────────────────
def calibrate_mic(recognizer: sr.Recognizer):
    """Run once at startup to calibrate ambient noise threshold."""
    print("[Mic] Calibrating ambient noise — please be quiet for 1 second...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
    print(f"[Mic] Energy threshold set to {recognizer.energy_threshold:.0f}\n")


def listen(recognizer: sr.Recognizer, stt: OfflineSTT, tracker=None):
    with sr.Microphone() as source:
        print("Listening...  (speak now)")
        t0 = time.time()
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=4)
        except sr.WaitTimeoutError:
            print("Timeout — no speech.\n")
            return None
        if tracker:
            tracker.record("recording", time.time() - t0)

        print("Transcribing...")
        t1 = time.time()
        text = stt.transcribe(audio)
        if tracker:
            tracker.record("stt", time.time() - t1)

        if not text:
            print("Could not understand — try again.\n")
        return text


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def print_header():
    w = 58
    print("\n" + "=" * w)
    print("   SMART HOME VOICE ASSISTANT  (OFFLINE)".center(w))
    print("=" * w)
    print("  Triggers  :  'تيفي ...'  /  'TV ...'")
    print("              'كليم ...'  /  'Cleem ...'  /  'AC ...'")
    print()
    print("  Examples  :  تيفي شعل       →  TV on")
    print("               تيفي زيد الصوت →  Volume up")
    print("               تيفي نيتفليكس  →  Open Netflix")
    print("               كليم شعل       →  AC on")
    print("               كليم برد الحال →  Cooling mode")
    print("               كليم نقص الدرجة→  Temp down")
    print("-" * w)
    print(f"  TV: {len(TV_COMMANDS)} commands  |  AC: {len(AC_COMMANDS)} commands")
    print("=" * w)
    print("  More examples:")
    print("               تيفي hdmi1         →  HDMI 1")
    print("               تيفي ترجمة         →  Subtitles")
    print("               تيفي مباشر         →  Live TV")
    print("               كليم عشرين         →  Set 20°C")
    print("               كليم ريح هادية     →  Fan Low")
    print("               كليم ريح قوية      →  Fan High")
    print("=" * w)
    print("  Ctrl+C to quit\n")


def print_result(result, show_perf=False):
    if result["command"]:
        icon = "[TV]" if result.get("device") == "TV" else "[AC]"
        line = "-" * 52
        print(f"\n{line}")
        print(f"  Heard      : {result['input']}")
        print(f"  Device     : {icon}  {result.get('device','')}")
        print(f"  Command    : {result['command']}")
        print(f"  Code       : {result['code']}")
        print(f"  Method     : {result['method']}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Action     : {result['description']}")
        if show_perf:
            print(f"  Match time : {result['processing_time']*1000:.1f} ms")
        print(f"{line}\n")
    else:
        print(f'\n  No match for: "{result["input"]}"')
        print("  Tip: start with 'تيفي' for TV or 'كليم' for AC\n")


def print_stats(tracker: PerformanceTracker):
    s = tracker.get_stats()
    print("\n" + "=" * 50)
    print("  PERFORMANCE STATISTICS")
    print("=" * 50)
    print(f"  Session        : {s['session_duration']:.1f} s")
    print(f"  Commands       : {s['total_commands']}  matched: {s['matched_commands']}")
    print(f"  Accuracy       : {s['accuracy']:.1f} %")
    print("-" * 50)
    print(f"  Avg recording  : {s['avg_recording']*1000:.0f} ms")
    print(f"  Avg STT        : {s['avg_stt']*1000:.0f} ms")
    print(f"  Avg matching   : {s['avg_matching']*1000:.0f} ms")
    print(f"  Avg total      : {s['avg_total']*1000:.0f} ms")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def run(model_size="tiny", show_perf=False):
    print_header()

    # 1. Merge comprehensive natural-language aliases
    n_extra = merge_extra_aliases()
    print(f"[NLP] Merged {n_extra} extra aliases from _EXTRA_ALIASES\n")

    # 2. Merge user-trained aliases from voice training
    load_learned_aliases(TV_COMMANDS, AC_COMMANDS)

    stt        = OfflineSTT(model_size=model_size)
    tv_matcher = DeviceCommandMatcher("TV", TV_COMMANDS)
    ac_matcher = DeviceCommandMatcher("AC", AC_COMMANDS)

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5   # faster end-of-speech detection

    calibrate_mic(recognizer)          # once at startup, not per listen

    tracker = PerformanceTracker() if show_perf else None

    print(f"[Info] STT     : Whisper {model_size} offline (beam_size=1)")
    print(f"[Info] NLP     : paraphrase-multilingual-MiniLM-L12-v2")
    print(f"[Info] Perf    : {'ON' if show_perf else 'OFF'}\n")

    while True:
        try:
            t_start = time.time()
            spoken = listen(recognizer, stt, tracker)
            if not spoken:
                continue

            print(f'You said: "{spoken}"')

            device, remainder = detect_device(spoken)

            if device:
                matcher = tv_matcher if device == "TV" else ac_matcher
                print(f"[{device}] trigger detected!")
                if remainder:
                    t_m = time.time()
                    result = matcher.match(remainder)
                    result["device"] = device
                    if tracker:
                        tracker.record("matching", time.time() - t_m)
                else:
                    print(f"[{device}] Say the command now.\n")
                    continue
            else:
                # No trigger word found — refuse to guess the device.
                # Guessing caused TV to run for AC commands.
                print("  No device trigger detected.")
                print("  Say 'تيفي <command>' for TV  or  'كليم <command>' for AC.\n")
                continue

            if tracker:
                tracker.record("total", time.time() - t_start)
                tracker.command_attempted(result["command"] is not None)

            print_result(result, show_perf=show_perf)

        except KeyboardInterrupt:
            print("\nSession ended. Goodbye!\n")
            if tracker:
                print_stats(tracker)
            sys.exit(0)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline Smart Home Voice Assistant — TV & AC in Darija"
    )
    parser.add_argument(
        "--model", default="tiny",
        choices=["tiny", "small", "medium", "large-v2", "large-v3"],
        help=(
            "Whisper model size (default: tiny = fastest).\n"
            "Use 'small' or 'medium' for better accuracy in noisy environments."
        ),
    )
    parser.add_argument(
        "--performance", action="store_true",
        help="Show timing statistics after each command and on exit",
    )
    args = parser.parse_args()
    run(model_size=args.model, show_perf=args.performance)
