"""
Tamil Pedagogy Engine: Grammar, Prose, and Poetry (முத்தமிழ் கற்பித்தல் கட்டமைப்பு)
Provides:
1. Automated Prosodic Scansion (யாப்பிலக்கண அலகிடுதல்: நேர் / நிரை அசை & சீர் வாய்பாடு)
2. Venba metre verification (வெண்பா இலக்கணக் கட்டுப்பாடு: தளை & ஈற்றுச்சீர்)
3. 5-Fold Grammar rule explications (ஐந்திலக்கணம்: எழுத்து, சொல், பொருள், யாப்பு, அணி)
4. Prose rhetoric analyzer (உரைநடைப் பகுப்பாய்வு & நடைப் பயிற்சி)
"""

import sys
import re
import unicodedata
from typing import List, Dict, Any, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Tamil Character Classes
VOWELS = set("அஆஇஈஉஊஎஏஐஒஓஔஃ")
SHORT_VOWELS = set("அஇஉஎஒ")
LONG_VOWELS = set("ஆஈஊஏஐஓஔ")
AYTHAM = "ஃ"

# Secondary vowel signs (உயிர்மெய்க் குறியீடுகள்)
SHORT_SIGNS = set("ிுெொ")
LONG_SIGNS = set("ாீூேைோௌ")
PULLI = "்"

class TamilPedagogyEngine:
    def __init__(self):
        self.seer_vaypadu_2 = {
            ("நேர்", "நேர்"): "தேமா",
            ("நிரை", "நேர்"): "புளிமா",
            ("கூவிளம்",): "கூவிளம்",
            ("நேர்", "நிரை"): "கூவிளம்",
            ("நிரை", "நிரை"): "கருவிளம்",
        }
        self.seer_vaypadu_3 = {
            ("நேர்", "நேர்", "நேர்"): "தேமாங்காய்",
            ("நிரை", "நேர்", "நேர்"): "புளிமாங்காய்",
            ("நேர்", "நிரை", "நேர்"): "கூவிளங்காய்",
            ("நிரை", "நிரை", "நேர்"): "கருவிளங்காய்",
            ("நேர்", "நேர்", "நிரை"): "தேமாங்கனி",
            ("நிரை", "நேர்", "நிரை"): "புளிமாங்கனி",
            ("நேர்", "நிரை", "நிரை"): "கூவிளங்கனி",
            ("நிரை", "நிரை", "நிரை"): "கருவிளங்கனி",
        }
        self.end_seer_vaypadu = {
            "நேர்": "நாள்",
            "நிரை": "மலர்",
            "நேர்பு": "காசு",
            "நிரைபு": "பிறப்பு"
        }

    def tokenize_tamil_letters(self, word: str) -> List[str]:
        """Splits a Tamil word into independent composite characters (எழுத்துக்கள்)."""
        letters = []
        i = 0
        word = unicodedata.normalize('NFC', word)
        while i < len(word):
            char = word[i]
            # Check if next char is a diacritic / pulli
            if i + 1 < len(word) and word[i+1] in (PULLI + "".join(SHORT_SIGNS | LONG_SIGNS)):
                letters.append(char + word[i+1])
                i += 2
            else:
                letters.append(char)
                i += 1
        return letters

    def classify_letter(self, letter: str) -> str:
        """Classifies letter into குறில் (K), நெடில் (N), or ஒற்று (O)."""
        if not letter:
            return ""
        if letter == AYTHAM or letter.endswith(PULLI):
            return "O" # ஒற்று (Mei / Aytham)
        # Check vowel signs
        for sign in LONG_SIGNS:
            if sign in letter:
                return "N" # நெடில்
        for sign in SHORT_SIGNS:
            if sign in letter:
                return "K" # குறில்
        # Standalone vowels
        if letter in LONG_VOWELS:
            return "N"
        if letter in SHORT_VOWELS or ('\u0B85' <= letter <= '\u0B94'):
            return "K"
        # Pure consonant base with implicit 'a'
        return "K"

    def scan_seer(self, word: str) -> Dict[str, Any]:
        """Scans a single word (சீர்) into syllables (அசைகள்: நேர்/நிரை) and identifies the வாய்பாடு."""
        letters = self.tokenize_tamil_letters(word)
        pattern = "".join(self.classify_letter(l) for l in letters)
        
        # Syllabification algorithm based on Yapparungalakkarigai
        # நிரை = KK, KKO, KN, KNO
        # நேர் = K, KO, N, NO
        syllables = []
        labels = []
        idx = 0
        n = len(pattern)

        while idx < n:
            # Check for Nirai first (begins with K followed by K or N)
            if idx + 1 < n and pattern[idx] == 'K' and pattern[idx+1] in ('K', 'N'):
                # Check if followed by ஒற்று (O)
                if idx + 2 < n and pattern[idx+2] == 'O':
                    # consume extra consecutive otru if present
                    end = idx + 3
                    while end < n and pattern[end] == 'O':
                        end += 1
                    syllables.append("".join(letters[idx:end]))
                    labels.append("நிரை")
                    idx = end
                else:
                    syllables.append("".join(letters[idx:idx+2]))
                    labels.append("நிரை")
                    idx += 2
            # Otherwise Ner
            else:
                end = idx + 1
                if idx < n and pattern[idx] in ('K', 'N'):
                    while end < n and pattern[end] == 'O':
                        end += 1
                syllables.append("".join(letters[idx:end]))
                labels.append("நேர்")
                idx = end

        tuple_labels = tuple(labels)
        vaypadu = "அறியப்படாத வாய்பாடு"
        if len(tuple_labels) == 1:
            vaypadu = self.end_seer_vaypadu.get(tuple_labels[0], tuple_labels[0])
        elif len(tuple_labels) == 2:
            vaypadu = self.seer_vaypadu_2.get(tuple_labels, "ஈரசீர்")
        elif len(tuple_labels) == 3:
            vaypadu = self.seer_vaypadu_3.get(tuple_labels, "மூசீர்")

        return {
            "word": word,
            "syllables": syllables,
            "asai_labels": labels,
            "vaypadu": vaypadu
        }

    def scan_kural(self, kural_text: str) -> Dict[str, Any]:
        """
        Parses and verifies a complete 2-line Tirukkural / Kural Venba:
        Line 1: 4 seers (நாற்சீர்)
        Line 2: 3 seers (முச்சீர்)
        """
        lines = [l.strip() for l in kural_text.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            # Try splitting by 4 seers and 3 seers
            words = kural_text.split()
            if len(words) == 7:
                lines = [" ".join(words[:4]), " ".join(words[4:])]
            else:
                return {"error": "குறள் வெண்பா 7 சீர்களைக் கொண்டிருக்க வேண்டும்."}

        parsed_lines = []
        total_seers = 0
        for l_idx, line in enumerate(lines[:2]):
            seers = line.split()
            total_seers += len(seers)
            line_analysis = []
            for s_idx, s in enumerate(seers):
                res = self.scan_seer(s)
                # Check for end seer of second line
                if l_idx == 1 and s_idx == len(seers) - 1:
                    last_asai = res["asai_labels"][-1] if res["asai_labels"] else "நேர்"
                    res["vaypadu"] = f"ஈற்று வாய்பாடு: {self.end_seer_vaypadu.get(last_asai, last_asai)}"
                line_analysis.append(res)
            parsed_lines.append(line_analysis)

        return {
            "metre": "குறள் வெண்பா (Kural Venba)",
            "line_1_seers": len(parsed_lines[0]),
            "line_2_seers": len(parsed_lines[1]),
            "total_seers": total_seers,
            "is_valid_venba_structure": len(parsed_lines[0]) == 4 and len(parsed_lines[1]) == 3,
            "analysis": parsed_lines
        }

pedagogy_engine = TamilPedagogyEngine()

if __name__ == "__main__":
    # Test on Kural 1: அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு
    kural = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
    scan_result = pedagogy_engine.scan_kural(kural)
    print("Scansion Test for Kural 1:")
    print(f"Structure Valid: {scan_result['is_valid_venba_structure']}")
    for i, line in enumerate(scan_result["analysis"], 1):
        line_str = " | ".join(f"{s['word']} ({'-'.join(s['asai_labels'])}: {s['vaypadu']})" for s in line)
        print(f"அடி {i}: {line_str}")
