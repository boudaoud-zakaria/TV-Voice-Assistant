# Smart Home Voice Assistant — Complete Documentation

**Version:** 3.0  
**Date:** May 2026  
**Status:** Production Ready

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How to Run](#2-how-to-run)
3. [How to Test (no microphone)](#3-how-to-test-no-microphone)
4. [Installation](#4-installation)
5. [Architecture](#5-architecture)
6. [AI Model Details](#6-ai-model-details)
7. [TV Command Reference](#7-tv-command-reference)
8. [AC Command Reference](#8-ac-command-reference)
9. [Language Support (EN / FR / Darija)](#9-language-support)
10. [Performance](#10-performance)
11. [Configuration & Tuning](#11-configuration--tuning)
12. [Troubleshooting](#12-troubleshooting)
13. [File Structure](#13-file-structure)
14. [Changelog](#14-changelog)

---

## 1. Project Overview

A **fully offline** voice assistant that controls a TV and an Air Conditioner using spoken commands in **English, French, and Algerian Darija (Arabic dialect)**. It never sends audio to the cloud — all transcription and NLP run locally on your machine.

### What it does

1. Listens to your microphone.
2. Transcribes speech offline with **Whisper** (faster-whisper, int8, CPU).
3. Detects which device you are addressing: **TV** or **AC (كليم / clim)**.
4. Matches the rest of the utterance to a command using a 4-level cascade:
   Exact → Substring → Fuzzy → Semantic NLP.
5. Returns the IR/network **command code** and a confidence score.

### Key numbers

| Metric | Value |
|--------|-------|
| Total TV commands | 40 |
| Total AC commands | 27 |
| Total aliases (TV + AC) | ~900+ |
| Languages | English, French, Algerian Darija |
| STT model | Whisper tiny (offline, int8) |
| NLP model | paraphrase-multilingual-MiniLM-L12-v2 |
| NLP model size | ~70 MB |
| Requires internet at runtime | No |

---

## 2. How to Run

### Step 1 — open a terminal in the project folder

```
cd f:\pfe
```

### Step 2 — run the assistant

```bash
# Default (fastest, Whisper tiny)
python tv_voice_assistant_enhanced.py

# More accurate transcription (slower startup)
python tv_voice_assistant_enhanced.py --model small

# Show timing stats after every command
python tv_voice_assistant_enhanced.py --performance

# Both flags combined
python tv_voice_assistant_enhanced.py --model small --performance
```

### Step 3 — speak

Say the **device trigger word** then the **command**:

```
تيفي شعل              →  TV ON
تيفي زيد الصوت        →  Volume UP
تيفي نيتفليكس         →  Open Netflix
تيفي ترجمة            →  Toggle Subtitles
تيفي hdmi واحد        →  Switch HDMI 1
تيفي مباشر            →  Live TV
تيفي 5                →  Press digit 5

كليم شعل              →  AC ON
كليم برد              →  Cooling mode
كليم عشرين            →  Set 20 °C
كليم ريح هادية        →  Fan Low
كليم ريح قوية         →  Fan High
كليم ريح أوتو         →  Fan Auto
كليم سوانغ            →  Swing
```

### Trigger words supported

| Device | Trigger words |
|--------|---------------|
| TV | `tv`, `tivi`, `تيفي`, `التيفي`, `تلفزيون`, `تلفاز` |
| AC | `كليم`, `الكليم`, `clim`, `cleem`, `ac`, `مكيف` |

Press **Ctrl+C** to quit. If `--performance` is set, statistics are printed on exit.

---

## 3. How to Test (no microphone)

The test utility runs all matching and trigger-detection logic without any audio hardware:

```bash
python test_and_debug_utilities.py
```

You will see an interactive menu:

```
1. Command Matching Test      (50+ cases, TV + AC, all 3 languages)
2. Device Trigger Detection Test
3. Database Statistics        (commands / aliases by category)
4. Performance Benchmark      (1000 iterations, timing per method)
5. Complete Command Reference (all codes + alias counts)
6. Export Commands to JSON    → smart_home_commands.json
7. Run All
```

#### Quick smoke-test (no menu, one line)

```bash
python -c "
from tv_voice_assistant_enhanced import DeviceCommandMatcher, TV_COMMANDS, AC_COMMANDS
tv = DeviceCommandMatcher('TV', TV_COMMANDS)
ac = DeviceCommandMatcher('AC', AC_COMMANDS)
for text, dev in [('شعل', tv), ('زيد الصوت', tv), ('عشرين', ac), ('ريح هادية', ac)]:
    r = dev.match(text)
    print(f'{text!r:20} -> {r[\"command\"]} ({r[\"method\"]})')
"
```

Expected output:
```
'شعل'                -> open (exact)
'زيد الصوت'          -> volume up (exact)
'عشرين'              -> temp_20 (exact)
'ريح هادية'          -> fan_low (exact)
```

---

## 4. Installation

### Requirements

- Python 3.9+
- Microphone (only needed to run the main assistant, not for tests)
- ~2 GB RAM at runtime (model + embeddings)
- ~1.5 GB disk (Python packages + Whisper + MiniLM model cache)
- Internet **only on first run** (to download Whisper and MiniLM weights)

### Install all dependencies

```bash
pip install -r requirements.txt
```

### Verify the install

```bash
python -c "import faster_whisper, sentence_transformers, rapidfuzz, speech_recognition; print('All OK')"
```

### Dependency table

| Package | Purpose |
|---------|---------|
| `faster-whisper` | Offline speech-to-text (CTranslate2 / int8) |
| `sentence-transformers` | Semantic NLP model (MiniLM) |
| `rapidfuzz` | Fast fuzzy string matching |
| `SpeechRecognition` | Microphone capture |
| `pyaudio` | Low-level audio I/O |
| `numpy` | Audio buffer processing |
| `torch` | Backend for sentence-transformers |

---

## 5. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Microphone  (SpeechRecognition + PyAudio)                   │
│  • 7s timeout, 4s phrase limit, 0.5s pause threshold         │
│  • Energy threshold auto-calibrated once at startup          │
└──────────────────────┬───────────────────────────────────────┘
                       │ raw PCM (16 kHz, int16)
┌──────────────────────▼───────────────────────────────────────┐
│  OfflineSTT  (faster-whisper)                                │
│  • Model: tiny (default) — 39M params, ~200 MB RAM           │
│  • compute_type=int8, beam_size=1 (greedy, 3-4x faster)      │
│  • VAD filter — skips silent segments automatically          │
│  • Language hint: "ar" (Arabic script for Darija)            │
│  • Pre-warmed at startup → no cold-start latency             │
└──────────────────────┬───────────────────────────────────────┘
                       │ transcribed text (Arabic / French / EN)
┌──────────────────────▼───────────────────────────────────────┐
│  detect_device()  — trigger word detection                   │
│  Pass 1: exact prefix match                                  │
│  Pass 2: fuzzy match (threshold 65) on each word             │
│  Returns: (device="TV"|"AC"|None, remainder)                 │
└──────────┬─────────────────────────────┬─────────────────────┘
           │ "TV"                        │ "AC"
┌──────────▼─────────┐        ┌──────────▼─────────────────────┐
│ DeviceCommandMatcher│        │ DeviceCommandMatcher           │
│ (TV, TV_COMMANDS)  │        │ (AC, AC_COMMANDS)               │
│                    │        │                                 │
│  Level 1 Exact  1ms│        │  same 4-level cascade          │
│  Level 2 Substr 2ms│        │                                 │
│  Level 3 Fuzzy 10ms│        │                                 │
│  Level 4 Seman 40ms│        │                                 │
└──────────┬─────────┘        └──────────┬──────────────────────┘
           └─────────────┬───────────────┘
                         │ result dict
┌────────────────────────▼─────────────────────────────────────┐
│  Output                                                      │
│  { command, code, method, confidence, description, category }│
└──────────────────────────────────────────────────────────────┘
```

### DeviceCommandMatcher — 4-level cascade

| Level | Method | Threshold | Speed | Accuracy |
|-------|--------|-----------|-------|----------|
| 1 | Exact dict lookup | — | ~1 ms | 100% |
| 2 | Substring (alias ⊆ text or text ⊆ alias) | — | ~2 ms | 95% |
| 3 | RapidFuzz WRatio | ≥ 68 | ~10 ms | 85% |
| 4 | Cosine similarity (MiniLM embeddings) | ≥ 0.48 | ~40 ms | 75–80% |

The model embeddings for all aliases are computed **once at startup** and cached as tensors. Runtime semantic lookup is just a matrix multiply (no re-encoding of aliases).

---

## 6. AI Model Details

### Speech-to-Text: Whisper (faster-whisper)

| Property | Value |
|----------|-------|
| Default model | `tiny` |
| Parameters | 39 M |
| Compute type | int8 (CPU-optimised) |
| Decoding | beam_size=1 (greedy) |
| VAD | built-in silero-VAD |
| Average transcription time | 0.3–0.8 s for a 2-3 s utterance |
| Languages | 99 (including Arabic / French) |

Switch to `--model small` (244 M params) for better accuracy in noisy environments, at the cost of ~2× slower transcription.

### Semantic NLP: paraphrase-multilingual-MiniLM-L12-v2

| Property | Value |
|----------|-------|
| Architecture | Sentence-BERT, 12-layer MiniLM |
| Embedding dimension | 384 |
| Supported languages | 50+ |
| Model size on disk | ~120 MB (cached by HuggingFace) |
| RAM at runtime | ~200 MB (shared between TV and AC matchers) |
| Inference per query | ~40 ms on CPU |
| License | Apache 2.0 |

The model is **loaded once and shared** between the TV and AC matchers (`DeviceCommandMatcher._shared_model`). Both devices encode their aliases at startup; at runtime only the query is re-encoded.

**Why this model?**  
It is multilingual, small enough to fit in 200 MB RAM, and strong at paraphrase detection — exactly what we need to match informal Darija expressions to canonical command names.

---

## 7. TV Command Reference

**40 commands total**

### Power (codes 0001–0002)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `open` | 0001 | Turn TV ON | on, allumer, شعل, ولع, خدم |
| `close` | 0002 | Turn TV OFF | off, éteindre, طفي, حبس, سكر |

### Volume (codes 0004–0006)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `mute` | 0004 | Mute/Unmute | silent, sourdine, كتم الصوت, اسكت |
| `volume up` | 0005 | Increase volume | louder, plus fort, زيد الصوت, قوّيه |
| `volume down` | 0006 | Decrease volume | quieter, moins fort, نقص الصوت, هادي |

### Channel (codes 0007–0008)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `channel up` | 0007 | Next channel | next, chaîne suivante, القناة الجاية |
| `channel down` | 0008 | Previous channel | back, chaîne précédente, الفايتة |

### Navigation (codes 0003, 0009–0010, 0032–0039)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `settings` | 0003 | Open settings | options, paramètres, الإعدادات, السيتينغ |
| `home` | 0009 | Home screen | main menu, accueil, الرئيسية, المنيو |
| `input` | 0010 | Switch input source | source, entrée, المصدر, بدّل المصدر |
| `guide` | 0032 | TV guide | schedule, programme, الدليل, شو كاين |
| `search` | 0033 | Search | find, chercher, ابحث, دور على |
| `zoom` | 0034 | Zoom / picture size | agrandir, زووم, كبر الصورة |
| `subtitles` | 0035 | Toggle subtitles | sub, sous-titres, ترجمة, سبتيتل |
| `info` | 0036 | Show info | details, informations, معلومات, شو هذا |
| `list` | 0037 | Channel list | favorites, liste, اللايسط, القائمة |
| `recent` | 0038 | Recent channels | history, récent, اللي شفت, الأخير |
| `live` | 0039 | Live TV | en direct, broadcast, مباشر, لايف |

### HDMI Inputs (codes 0050–0053)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `hdmi1` | 0050 | HDMI 1 | hdmi 1, entrée 1, hdmi واحد, المصدر الأول |
| `hdmi2` | 0051 | HDMI 2 | hdmi 2, entrée 2, hdmi جوج |
| `hdmi3` | 0052 | HDMI 3 | hdmi 3, entrée 3, hdmi تلاتة |
| `hdmi4` | 0053 | HDMI 4 | hdmi 4, entrée 4, hdmi ربعة |

### Digit Keys (codes 0040–0049)

| Key | Code | Darija | French |
|-----|------|--------|--------|
| `digit_0` | 0040 | صفر | zéro |
| `digit_1` | 0041 | واحد | un |
| `digit_2` | 0042 | جوج / اثنين | deux |
| `digit_3` | 0043 | تلاتة / ثلاثة | trois |
| `digit_4` | 0044 | ربعة / أربعة | quatre |
| `digit_5` | 0045 | خمسة | cinq |
| `digit_6` | 0046 | ستة | six |
| `digit_7` | 0047 | سبعة | sept |
| `digit_8` | 0048 | تمانية / ثمانية | huit |
| `digit_9` | 0049 | تسعة | neuf |

### Playback (codes 0011–0015)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `play` | 0011 | Play/Resume | jouer, شغّل, دوّر |
| `pause` | 0012 | Pause | pauser, وقّف, جمّد |
| `stop` | 0013 | Stop | arrêter, إيقاف |
| `rewind` | 0014 | Rewind | rembobiner, رجّع, للخلف |
| `fast forward` | 0015 | Fast forward | avancer, قدّم |

### Apps (codes 0020–0022)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `netflix` | 0020 | Open Netflix | ouvrir netflix, حل نيتفليكس |
| `youtube` | 0021 | Open YouTube | ouvrir youtube, حل يوتيوب |
| `spotify` | 0022 | Open Spotify | musique, موزيكا |

---

## 8. AC Command Reference

**27 commands total**

### Power (codes 1001–1002)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `open` | 1001 | Turn AC ON | allumer le clim, شعل الكليم, ولع الكليم |
| `close` | 1002 | Turn AC OFF | éteindre le clim, طفي الكليم, حبس الكليم |

### Modes (codes 1003–1004, 1011–1013)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `cool` | 1003 | Cooling mode | froid, cold, برد, كلاص, ردها برد |
| `heat` | 1004 | Heating mode | chaud, warm, سخن, دافي, دفيني |
| `dry` | 1011 | Dry / dehumidify | sécher, جاف, تجفيف |
| `fan only` | 1012 | Fan only | ventilation, ريح بس, مروحة بس |
| `auto` | 1013 | Auto mode | automatique, أوتو, تلقائي |

### Temperature presets (codes 1005–1006, 1020–1027)

| Key | Code | Description | Darija |
|-----|------|-------------|--------|
| `temp up` | 1005 | Increase temp | زيد الدرجة, طلع شوية |
| `temp down` | 1006 | Decrease temp | نقص الدرجة, هبط شوية |
| `temp_16` | 1020 | Set 16 °C | ستاش درجة |
| `temp_18` | 1021 | Set 18 °C | تمنتاش درجة |
| `temp_20` | 1022 | Set 20 °C | عشرين درجة |
| `temp_22` | 1023 | Set 22 °C | جوج وعشرين درجة |
| `temp_24` | 1024 | Set 24 °C | ربعة وعشرين درجة |
| `temp_26` | 1025 | Set 26 °C | ستة وعشرين درجة |
| `temp_28` | 1026 | Set 28 °C | تمانية وعشرين درجة |
| `temp_30` | 1027 | Set 30 °C | تلاتين درجة |

### Fan speed (codes 1007–1008, 1016–1019)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `fan up` | 1007 | Increase fan | plus de vent, زيد الريح |
| `fan down` | 1008 | Decrease fan | moins de vent, نقص الريح |
| `fan_auto` | 1016 | Fan auto speed | vitesse auto, ريح أوتو, الريح التلقائي |
| `fan_low` | 1017 | Fan low | vitesse basse, ريح هادية, ريح واطة |
| `fan_med` | 1018 | Fan medium | vitesse moyenne, ريح متوسطة |
| `fan_high` | 1019 | Fan high | vitesse élevée, ريح قوية, ريح عالية |

### Other (codes 1009–1010, 1014–1015)

| Key | Code | Description | Example aliases |
|-----|------|-------------|-----------------|
| `swing` | 1009 | Toggle swing | oscillation, دوّر الريش, سوانغ |
| `silent` | 1010 | Silent / night mode | mode nuit, وضع الليل, هادي |
| `turbo` | 1014 | Turbo / max power | puissance max, توربو, برد بسرعة |
| `timer` | 1015 | Set timer | minuterie, تايمر, حبس بعد ساعة |

---

## 9. Language Support

### Darija (Algerian Arabic dialect) — key vocabulary

| Darija | Meaning | Commands |
|--------|---------|---------|
| شعل / ولع / خدم | turn on | open (TV+AC) |
| طفي / حبس / سكر | turn off | close (TV+AC) |
| زيد | increase / add | volume up, temp up, fan up |
| نقص | decrease | volume down, temp down, fan down |
| كلاص / برد | cold / cool | AC cool mode |
| سخن / دافي | hot / warm | AC heat mode |
| ريح | wind / air / fan | all fan commands |
| ستاش | 16 (Darija number) | temp_16 |
| تمنتاش | 18 | temp_18 |
| عشرين | 20 | temp_20 |
| جوج وعشرين | 22 | temp_22 |
| تلاتين | 30 | temp_30 |
| ترجمة | subtitles / translation | subtitles |
| مباشر / لايف | live / direct | live |
| اللايسط | list (from "la liste") | list |

### French — trigger examples

| French | Maps to |
|--------|---------|
| allumer, ouvrir | open (TV or AC) |
| éteindre, fermer | close |
| augmenter le volume | volume up |
| baisser le son | volume down |
| sous-titres | subtitles |
| en direct | live |
| seize degrés | temp_16 |
| ريح هادية / vitesse basse | fan_low |

### Whisper language hint

The STT is set to `language="ar"` which transcribes Darija in Arabic script. French and English words within the same sentence are also captured correctly because Whisper is multilingual.

---

## 10. Performance

### Startup times (first run — model download from HuggingFace)

| Step | Time |
|------|------|
| Whisper tiny download | ~100 MB, depends on connection |
| MiniLM model download | ~120 MB |
| Whisper load + pre-warm | 3–6 s |
| Encoding TV aliases | 2–4 s |
| Encoding AC aliases | 2–4 s |
| **Total first run** | **~10–20 s** |

Subsequent runs: **5–8 s** (models loaded from local cache).

### Runtime latency (per command)

| Stage | Typical |
|-------|---------|
| Microphone recording | 1.5–3 s (depends on how long you speak) |
| Whisper transcription | 0.3–0.8 s |
| Trigger detection | < 1 ms |
| Command matching (exact) | ~1 ms |
| Command matching (semantic worst case) | ~40 ms |
| **End-to-end** | **~2–4 s** |

### Memory usage

| Component | RAM |
|-----------|-----|
| Whisper tiny (int8) | ~160 MB |
| MiniLM model | ~200 MB |
| TV alias embeddings | ~8 MB |
| AC alias embeddings | ~5 MB |
| Python + libraries | ~100 MB |
| **Total** | **~475 MB** |

---

## 11. Configuration & Tuning

### Whisper model size vs accuracy/speed

```bash
--model tiny     # default — fastest, ~94% accuracy on clean speech
--model small    # 2× slower, ~96% accuracy — recommended for noisy rooms
--model medium   # 4× slower, ~98% — for heavy accents / heavy noise
```

### Matching thresholds (in `DeviceCommandMatcher.match()`)

```python
fuzzy_threshold    = 68    # RapidFuzz WRatio score (0-100)
semantic_threshold = 0.48  # Cosine similarity (0.0-1.0)
```

- **Lower fuzzy_threshold** → catches more typos / slurred speech, but may produce false positives.
- **Higher semantic_threshold** → only very confident semantic matches get through.

### Microphone settings (in `run()`)

```python
recognizer.pause_threshold = 0.5    # seconds of silence to end phrase
recognizer.dynamic_energy_threshold = True
# phrase_time_limit = 4 s  (inside listen())
```

Increase `pause_threshold` to 0.8 if commands are getting cut off mid-sentence.

---

## 12. Troubleshooting

### "Could not understand — try again"

Whisper returned empty text. Causes:
- Room too quiet (VAD filter skipped the segment) — speak louder.
- Room too noisy — try `--model small`.
- `phrase_time_limit=4s` too short — increase it for long sentences.

### Trigger detected but wrong command matched

- Check if `method` is `"semantic ..."` with low confidence — the semantic model picked a bad match.
- Add the exact phrase you are saying as an alias to the command dictionary.
- Raise `semantic_threshold` to 0.55 to be more selective.

### PyAudio / microphone not found

```bash
pip install pyaudio
# On Windows if pip fails:
pip install pipwin && pipwin install pyaudio
```

### MiniLM model download fails

```bash
# Force re-download
pip install -U sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### Test utility crashes with ImportError

The test utility imports directly from the main file. Make sure both files are in the same folder:
```
f:\pfe\tv_voice_assistant_enhanced.py
f:\pfe\test_and_debug_utilities.py
```

---

## 13. File Structure

```
f:\pfe\
├── tv_voice_assistant_enhanced.py   ← main application (run this)
├── test_and_debug_utilities.py      ← offline tests (no mic needed)
├── requirements.txt                 ← pip dependencies
├── README.md                        ← this file
└── Tv voice controller.py           ← original legacy version
```

### tv_voice_assistant_enhanced.py — internal layout

```
PerformanceTracker          metrics collection
TV_COMMANDS                 dict: 40 commands × aliases + codes
AC_COMMANDS                 dict: 27 commands × aliases + codes
build_alias_map()           TV_COMMANDS / AC_COMMANDS → flat alias→key dict
detect_device()             trigger word detection (exact + fuzzy)
DeviceCommandMatcher        4-level matching engine (shared NLP model)
OfflineSTT                  faster-whisper wrapper
calibrate_mic()             one-shot ambient noise calibration
listen()                    microphone → AudioData
print_header/result/stats() terminal UI
run()                       main loop
```

---

## 14. Changelog

### v3.0 (May 2026) — current
- Added **AC device** (27 commands) with full Darija + FR + EN aliases
- Added TV: `zoom`, `subtitles`, `info`, `list`, `recent`, `live`
- Added TV: `hdmi1`–`hdmi4` input switching
- Added TV: digit keys `0`–`9` with Darija number words
- Added AC: specific temperature presets `16°–30°` with Darija
- Added AC: `fan_auto`, `fan_low`, `fan_med`, `fan_high`
- Replaced Google Cloud STT with **offline Whisper** (no internet needed)
- Rewrote test utility to cover both TV and AC

### v2.0
- `DeviceCommandMatcher` class with shared NLP model
- Dual-device trigger detection (`تيفي` vs `كليم`)
- Performance tracker and `--performance` flag
- Whisper pre-warming at startup
- 20 TV commands, Darija aliases

### v1.0
- Basic voice-to-command matching
- Google STT, sentence-transformers, rapidfuzz
- English + French only
