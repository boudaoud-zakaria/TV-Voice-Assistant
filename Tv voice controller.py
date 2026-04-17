"""
TV Voice Assistant - LG via ESP8266 + KY-005 IR
================================================
Python envoie un code court -> ESP8266 -> KY-005 -> TV LG

Install:
    pip install SpeechRecognition sentence-transformers rapidfuzz pyserial

Run:
    python tv_voice_lg.py
    python tv_voice_lg.py --lang fr-FR
    python tv_voice_lg.py --port COM3 --lang fr-FR
"""

import argparse
import sys
import time

import serial
import serial.tools.list_ports
import speech_recognition as sr
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
#  TV COMMANDS
#  "code" doit correspondre EXACTEMENT aux
#  commandes dans le sketch ESP8266
# ─────────────────────────────────────────────
TV_COMMANDS = {
    "power": {
        "code": "p",#p f tv
        "aliases": [
            "power", "open", "turn on", "power on", "on", "allumer",
            "close", "turn off", "power off", "off", "eteindre",
            "افتح", "أغلق", "تشغيل", "إيقاف",
        ],
        "description": "Power ON/OFF"
    },
    "close": {
        "code": "p",#p f tv
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
    "volume up": {
        "code": "v+",
        "aliases": [
            "volume up", "increase volume", "louder", "turn up",
            "raise volume", "more sound", "amplify", "plus fort",
            "augmenter le volume", "ارفع الصوت", "صوت اعلى",
        ],
        "description": "Volume +"
    },
    "volume down": {
        "code": "v-",
        "aliases": [
            "volume down", "decrease volume", "quieter", "turn down",
            "lower volume", "less sound", "softer", "moins fort",
            "diminuer le volume", "اخفض الصوت", "صوت اقل",
        ],
        "description": "Volume -"
    },
    "mute": {
        "code": "m",
        "aliases": [
            "mute", "silent", "silence", "quiet", "no sound", "shh",
            "sourdine", "couper le son", "muet",
            "كتم", "صامت", "اسكت",
        ],
        "description": "Mute / Unmute"
    },
    "input": {
        "code": "input",
        "aliases": [
            "input", "source", "hdmi", "change input", "change source",
            "entree", "المصدر", "الإدخال",
        ],
        "description": "Source / HDMI"
    },
    "channel up": {
        "code": "c+",
        "aliases": [
            "channel up", "next channel", "next",
            "chaine suivante", "القناة التالية",
        ],
        "description": "Chaine +"
    },
    "channel down": {
        "code": "c-",
        "aliases": [
            "channel down", "previous channel", "back",
            "chaine precedente", "القناة السابقة",
        ],
        "description": "Chaine -"
    },
    "home": {
        "code": "home",
        "aliases": [
            "home", "home screen", "main menu", "go home",
            "accueil", "ecran principal", "الرئيسية",
        ],
        "description": "Home"
    },
    "settings": {
        "code": "settings",
        "aliases": [
            "settings", "setting", "options", "menu", "configure",
            "parametres", "reglages", "الإعدادات", "ضبط",
        ],
        "description": "Settings"
    },
    "ok": {
        "code": "ok",
        "aliases": [
            "ok", "okay", "confirm", "select", "enter",
            "valider", "confirmer", "موافق", "تاكيد",
        ],
        "description": "OK / Confirmer"
    },
    "play": {
        "code": "play",
        "aliases": [
            "play", "resume", "continue", "start playing",
            "jouer", "reprendre", "تشغيل", "استئناف",
        ],
        "description": "Play"
    },
    "netflix": {
        "code": "netflix",
        "aliases": [
            "netflix", "netflix app", "open netflix", "start netflix",
            "netflix", "ouvrir netflix", "lancer netflix",
            "نيتفليكس", "نتفليكس", "افتح نيتفليكس", "شغل نيتفليكس"
    ],
    "description": "Open Netflix"
    },
    "youtube": {
        "code": "yt",
        "aliases": [
            "youtube", "youtube app", "open youtube", "start youtube",
            "youtube", "app youtube", "ouvrir youtube", "lancer youtube",
            "يوتيوب", "افتح يوتيوب", "شغل يوتيوب"
    ],
    "description": "Open YouTube"
    },
    "pause": {
        "code": "pause",
        "aliases": [
            "pause", "freeze", "hold", "stop playing", "wait",
            "pauser", "mettre en pause", "إيقاف مؤقت",
        ],
        "description": "Pause"
    },
    "recording": {
        "code": "recordlist",
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
        "code": "guide",
        "aliases": [
            "guide", "tv guide", "program guide", "what's on", "schedule",
            "guide tv", "guide des programmes", "horaire", "ce qui passe",
            "الدليل", "دليل البرامج", "البرامج", "الجدول",
        ],
        "category": "Navigation",
        "description": "Open TV Guide"
    },
    "search": {
        "code": "search",
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
#  SERIAL MANAGER
# ─────────────────────────────────────────────
class SerialManager:
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def auto_detect_port(self):
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            return None
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ("usb", "uart", "ch340", "cp210", "ftdi", "esp", "wemos", "nodemcu")):
                return p.device
        return ports[0].device

    def connect(self):
        if self.port is None:
            self.port = self.auto_detect_port()
            if self.port is None:
                print("[Serial] Aucun ESP8266 trouve - mode simulation.")
                return False
            print(f"[Serial] ESP8266 detecte : {self.port}")

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=2,
            )
            print("[Serial] Attente demarrage ESP8266 (2s)...")
            time.sleep(2)
            self.ser.flushInput()
            print(f"[Serial] Connecte -> {self.port} @ {self.baudrate} baud\n")
            return True
        except serial.SerialException as e:
            print(f"[Serial] Impossible d'ouvrir {self.port} : {e}")
            self.ser = None
            return False

    def send_code(self, code: str):
        """Envoie le code court + newline vers l'ESP8266."""
        message = f"{code}\n"

        if self.ser is None or not self.ser.is_open:
            print(f"[Serial] (simulation) -> {message.strip()}")
            return

        try:
            self.ser.write(message.encode("utf-8"))
            self.ser.flush()
            print(f"[Serial] Envoye -> ESP8266 : '{message.strip()}'")

            # Lire la reponse de l'ESP8266
            time.sleep(0.05)
            if self.ser.in_waiting:
                reply = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if reply:
                    print(f"[Serial] ESP8266 : {reply}")

        except serial.SerialException as e:
            print(f"[Serial] Erreur : {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[Serial] Port ferme.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────
#  ALIAS MAP + NLP MATCHER
# ─────────────────────────────────────────────
def build_alias_map(commands):
    alias_map = {}
    for cmd_name, cmd_data in commands.items():
        for alias in cmd_data["aliases"]:
            alias_map[alias.lower()] = cmd_name
    return alias_map


class TVCommandMatcher:
    def __init__(self, commands):
        self.commands = commands
        self.alias_map = build_alias_map(commands)
        self.all_aliases = list(self.alias_map.keys())

        print("[NLP] Chargement du modele semantique...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("[NLP] Encodage des alias...")
        self.alias_embeddings = self.model.encode(
            self.all_aliases, convert_to_tensor=True, show_progress_bar=False
        )
        print("[NLP] Pret !\n")

    def match(self, spoken_text, fuzzy_threshold=70, semantic_threshold=0.55):
        text = spoken_text.lower().strip()
        result = {"input": spoken_text, "command": None, "code": None,
                  "method": None, "confidence": 0.0, "description": None}

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

        best_fuzzy, score, _ = process.extractOne(
            text, self.all_aliases, scorer=fuzz.WRatio)
        if score >= fuzzy_threshold:
            cmd = self.alias_map[best_fuzzy]
            result.update({"command": cmd, "code": self.commands[cmd]["code"],
                           "method": f"fuzzy ({score:.0f}%)", "confidence": score / 100,
                           "description": self.commands[cmd]["description"]})
            return result

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
    print("\n" + "="*54)
    print("   LG TV VOICE - ESP8266 + KY-005 (D5)")
    print("="*54)
    print(f"  {'Commande':<14} | {'Code':<4} | Description")
    print("-"*54)
    for name, data in TV_COMMANDS.items():
        print(f"  {name:<14} | {data['code']:<4} | {data['description']}")
    print("="*54)
    print("  Parle une commande - Ctrl+C pour quitter\n")

def print_result(result):
    if result["command"]:
        print("\n+----------------------------------------+")
        print(f"|  Entendu  : {result['input']:<26} |")
        print(f"|  Commande : {result['command']:<26} |")
        print(f"|  Code ESP : '{result['code']}'  {'':<22}|")
        print(f"|  Methode  : {result['method']:<26} |")
        print(f"|  Action   : {result['description']:<26} |")
        print("+----------------------------------------+\n")
    else:
        print(f"\n  Non reconnu : \"{result['input']}\"")
        print("  Reformule ou parle plus clairement.\n")


# ─────────────────────────────────────────────
#  MIC LISTENER
# ─────────────────────────────────────────────
def listen_and_recognize(recognizer, language):
    with sr.Microphone() as source:
        print("Ecoute... (parle maintenant)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        except sr.WaitTimeoutError:
            print("Timeout - aucune parole detectee\n")
            return None

    print("Traitement...")
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        print("Incomprehensible - reessaie\n")
        return None
    except sr.RequestError as e:
        print(f"Erreur reseau : {e}\n")
        return None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run(language="fr-FR", serial_port=None, baudrate=115200):
    print_header()
    matcher = TVCommandMatcher(TV_COMMANDS)
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    print(f"[Info] Langue : {language}")
    print(f"[Info] Baud   : {baudrate}\n")

    with SerialManager(port=serial_port, baudrate=baudrate) as serial_mgr:
        while True:
            try:
                spoken = listen_and_recognize(recognizer, language)
                if spoken is None:
                    continue

                print(f'Tu as dit : "{spoken}"')
                result = matcher.match(spoken)
                print_result(result)

                if result["code"]:
                    serial_mgr.send_code(result["code"])

            except KeyboardInterrupt:
                print("\n\nArrete. Au revoir !\n")
                sys.exit(0)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LG TV Voice - ESP8266 KY-005")
    parser.add_argument(
        "--lang", default="ar-DZ",
        help="Langue : en-US | fr-FR | ar-DZ  (defaut: fr-FR)"
    )
    parser.add_argument(
        "--port", default=None,
        help="Port ESP8266 : COM3 (Windows) | /dev/ttyUSB0 (Linux)"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (defaut: 115200)"
    )
    args = parser.parse_args()
    run(language=args.lang, serial_port=args.port, baudrate=args.baud)