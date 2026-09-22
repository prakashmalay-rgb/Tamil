"""
Tamil Grammar & Sandhi Engine (தமிழ் இலக்கணம் மற்றும் சந்தி விதிகள்)
Neuro-symbolic linguistic validation for Tamil sentence construction, Sandhi doubling, and orthography.
Based on Tolkappiyam and Nannool grammatical treatises.
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Any, Optional

try:
    import tamil
    from tamil import utf8
except ImportError:
    tamil = None
    utf8 = None

# Consonant families
VALLINAM_STOPS = {"க", "ச", "த", "ப"}
VALLINAM_MEI = {
    "க": "க்",
    "ச": "ச்",
    "த": "த்",
    "ப": "ப்"
}

# Words after which Vallinam MANDATORILY doubles (வல்லினம் மிகும் இடங்கள்)
MANDATORY_DOUBLING_WORDS = {
    # Demonstrative and interrogative pronouns/adverbs
    "அந்த", "இந்த", "எந்த",
    "அங்கு", "இங்கு", "எங்கு",
    "அப்படி", "இப்படி", "எப்படி",
    "ஆங்கு", "ஈங்கு", "யாங்கு",
    "அப்பொழுது", "இப்பொழுது", "எப்பொழுது",
    "அப்போது", "இப்போது", "எப்போது",
    "அத்துணை", "இத்துணை", "எத்துணை",
    "அவ்வகை", "இவ்வகை", "எவ்வகை",
    "மற்று", "இனி", "தனி"
}

# Words after which Vallinam NEVER doubles (வல்லினம் மிகா இடங்கள்)
PROHIBITED_DOUBLING_WORDS = {
    "சில", "பல", "அவை", "இவை", "எவை",
    "அன்று", "இன்று", "என்று",
    "அத்தனை", "இத்தனை", "எத்தனை",
    "வாழ்க", "வளர்க", "வருக", "போக" # வியங்கோள் வினைமுற்று
}

class TamilGrammarValidator:
    def __init__(self):
        self.vallinam_prefix_map = {
            "க": "க்", "கா": "க்", "கி": "க்", "கீ": "க்", "கு": "க்", "கூ": "க்", "கெ": "க்", "கே": "க்", "கை": "க்", "கொ": "க்", "கோ": "க்", "கௌ": "க்",
            "ச": "ச்", "சா": "ச்", "சி": "ச்", "சீ": "ச்", "சு": "ச்", "சூ": "ச்", "செ": "ச்", "சே": "ச்", "சை": "ச்", "சொ": "ச்", "சோ": "ச்", "சௌ": "ச்",
            "த": "த்", "தா": "த்", "தி": "த்", "தீ": "த்", "து": "த்", "தூ": "த்", "தெ": "த்", "தே": "த்", "தை": "த்", "தொ": "த்", "தோ": "த்", "தௌ": "த்",
            "ப": "ப்", "பா": "ப்", "பி": "ப்", "பீ": "ப்", "பு": "ப்", "பூ": "ப்", "பெ": "ப்", "பே": "ப்", "பை": "ப்", "பொ": "ப்", "போ": "ப்", "பௌ": "ப்",
        }

    def get_letters(self, text: str) -> List[str]:
        if utf8:
            return utf8.get_letters(text)
        # Fallback grapheme cluster extraction
        normalized = unicodedata.normalize("NFC", text)
        return list(normalized)

    def get_starting_vallinam(self, word: str) -> Optional[str]:
        """Returns the corresponding pulli mei ('க்', 'ச்', 'த்', 'ப்') if word starts with a Vallinam stop."""
        letters = self.get_letters(word)
        if not letters:
            return None
        first_letter = letters[0]
        return self.vallinam_prefix_map.get(first_letter)

    def check_sandhi_errors(self, text: str) -> List[Dict[str, Any]]:
        """
        Detects missing and erroneous Sandhi consonant markers between words.
        """
        words = re.findall(r"[\u0B80-\u0BFF]+|[a-zA-Z0-9]+|[^\s\w]", text)
        errors = []

        for i in range(len(words) - 1):
            w1 = words[i]
            w2 = words[i + 1]

            # Only check Tamil words
            if not re.search(r"[\u0B80-\u0BFF]", w1) or not re.search(r"[\u0B80-\u0BFF]", w2):
                continue

            expected_mei = self.get_starting_vallinam(w2)

            # Rule 1: Words that mandatorily require doubling (அந்த, இந்த, அங்கு, etc.)
            if w1 in MANDATORY_DOUBLING_WORDS:
                if expected_mei and not w1.endswith(expected_mei):
                    errors.append({
                        "type": "missing_sandhi",
                        "rule": "சுட்டு/வினா பெயர் முன் வல்லினம் மிகும்",
                        "w1": w1,
                        "w2": w2,
                        "expected_mei": expected_mei,
                        "fix": f"{w1}{expected_mei} {w2}"
                    })

            # Rule 2: Accusative case marker 'ஐ' (இரண்டாம் வேற்றுமை விரி)
            elif (w1.endswith("ை") or w1.endswith("யை") or w1.endswith("ளை") or w1.endswith("றை")) and len(w1) > 2:
                # E.g., திட்டத்தை, வேலையை, மின்னஞ்சலை
                if expected_mei and not w1.endswith(expected_mei):
                    # Check if previous word is an object noun
                    errors.append({
                        "type": "missing_sandhi",
                        "rule": "இரண்டாம் வேற்றுமை விரி முன் வல்லினம் மிகும்",
                        "w1": w1,
                        "w2": w2,
                        "expected_mei": expected_mei,
                        "fix": f"{w1}{expected_mei} {w2}"
                    })

            # Rule 3: Dative case marker 'க்கு' / 'கு' (நான்காம் வேற்றுமை விரி)
            elif w1.endswith("க்கு") or w1.endswith("கு"):
                if expected_mei and not w1.endswith(expected_mei):
                    errors.append({
                        "type": "missing_sandhi",
                        "rule": "நான்காம் வேற்றுமை விரி முன் வல்லினம் மிகும்",
                        "w1": w1,
                        "w2": w2,
                        "expected_mei": expected_mei,
                        "fix": f"{w1}{expected_mei} {w2}"
                    })

            # Rule 4: Prohibited doubling (வல்லினம் மிகா இடங்கள்)
            elif w1 in PROHIBITED_DOUBLING_WORDS:
                for mei in VALLINAM_MEI.values():
                    if w1.endswith(mei):
                        errors.append({
                            "type": "unnecessary_sandhi",
                            "rule": "வல்லினம் மிகா இடம்",
                            "w1": w1,
                            "w2": w2,
                            "fix": f"{w1[:-len(mei)]} {w2}"
                        })

        return errors

    def correct_sandhi(self, text: str) -> str:
        """
        Automatically corrects Sandhi junction errors based on grammar rules.
        """
        corrected = text
        # Common Sandhi junction patterns
        patterns = [
            # அந்த/இந்த/எந்த + க/ச/த/ப
            (r"\b(அந்த|இந்த|எந்த)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
            # அங்கு/இங்கு/எங்கு + க/ச/த/ப
            (r"\b(அங்கு|இங்கு|எங்கு)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
            # அப்படி/இப்படி/எப்படி + க/ச/த/ப
            (r"\b(அப்படி|இப்படி|எப்படி)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
            # அப்போது/இப்போது/எப்போது + க/ச/த/ப
            (r"\b(அப்போது|இப்போது|எப்போது)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
            # இனி/தனி + க/ச/த/ப
            (r"\b(இனி|தனி)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
            # ஐ உருபு (பாடத்தை + படி -> பாடத்தைப் படி, வேலையை + செய் -> வேலையைச் செய்)
            (r"([\u0B80-\u0BFF]{2,}ை)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
            # கு உருபு (அவருக்கு + கொடு -> அவருக்குக் கொடு)
            (r"([\u0B80-\u0BFF]{2,}க்கு)\s+([கசதப][\u0B80-\u0BFF]*)", self._apply_doubling),
        ]

        for pat, repl_fn in patterns:
            corrected = re.sub(pat, repl_fn, corrected)

        # Apply orthography and spelling correction
        corrected = self.correct_orthography(corrected)
        # Ensure sentence completion
        corrected = self.ensure_complete_sentence(corrected)

        return corrected

    def correct_orthography(self, text: str) -> str:
        """
        Fixes common transliteration and orthographical spelling errors in Tamil.
        """
        spelling_map = [
            (r"\b(வானக்கம்|வான்கம்|வானக்க|வான்க|வணக்கம)\b", "வணக்கம்"),
            (r"(வானக்கம்|வான்கம்)", "வணக்கம்"),
            (r"\bபாஷை\b", "மொழி"),
            (r"\bபாஷையில்\b", "மொழியில்"),
            (r"\bநீங்கள்\s+எப்படி\?", "நீங்கள் எப்படி இருக்கிறீர்கள்?"),
            (r"\bஎப்படி\s+இருக்கீங்க\?", "எப்படி இருக்கிறீர்கள்?"),
            (r"\bதொலையீடு\b", "தொடர்பு"),
        ]
        for pat, rep in spelling_map:
            text = re.sub(pat, rep, text)
        # Normalize any redundant consonant dots (இரட்டைப் புள்ளிகள் நீக்கம்)
        text = re.sub(r"்+", "்", text)
        return text

    def ensure_complete_sentence(self, text: str) -> str:
        """
        Ensures the text does not end abruptly mid-sentence or mid-word.
        """
        text = text.strip()
        if not text:
            return "வணக்கம்! உங்களுக்கு நான் எவ்வாறு உதவ முடியும்?"

        # If already ends with sentence boundary
        if text[-1] in (".", "!", "?", "।", "\n", "”", '"', "🌟", "😊"):
            return text

        # Find the last sentence boundary
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"), text.rfind("\n"))
        if last_punct > 10:
            return text[:last_punct + 1].strip()

        # If no punctuation exists, check if last word is incomplete and close gracefully
        words = text.split()
        if len(words) >= 3:
            return text + "."
        return text

    def _apply_doubling(self, match: re.Match) -> str:
        w1 = match.group(1)
        w2 = match.group(2)
        mei = self.get_starting_vallinam(w2)
        if mei and not w1.endswith(mei):
            return f"{w1}{mei} {w2}"
        return match.group(0)

    def validate_sentence_structure(self, sentence: str) -> Dict[str, Any]:
        """
        Validates basic SOV (Subject - Object - Verb) and ending markers.
        """
        words = re.findall(r"[\u0B80-\u0BFF]+", sentence)
        if not words:
            return {"valid": False, "reason": "No Tamil words found"}

        last_word = words[-1]
        
        # Valid Tamil finite verb endings (வினைமுற்று விகுதிகள்: -ஆன், -ஆள், -ஆர், -ஆர்கள், -து, -ன, -ஓம், -ஈர்கள், -வும், -து, -துள்ளது)
        verb_endings = (
            "கிறது", "கின்றது", "கிறதுது", "கிறார்", "கிறார்கள்", "கிறோம்", "கிறாள்", "கிறான்",
            "த்தது", "ந்தது", "பட்டது", "ப்பட்டது", "ப்பட்டதுது",
            "வும்", "வேண்டும்", "கூடாது", "முடியும்", "இயலும்", "உள்ளது", "உள்ளன",
            "நன்றி", "வாழ்க", "செய்க"
        )
        has_predicate = any(last_word.endswith(ending) for ending in verb_endings)

        tamil_char_count = len(re.findall(r"[\u0B80-\u0BFF]", sentence))
        total_letters = max(1, len(re.findall(r"[a-zA-Z\u0B80-\u0BFF]", sentence)))
        tamil_pct = (tamil_char_count / total_letters) * 100

        return {
            "valid": True,
            "has_finite_verb_end": has_predicate,
            "tamil_percentage": round(tamil_pct, 1),
            "word_count": len(words),
            "is_pure_tamil": tamil_pct >= 85.0
        }

    def audit_text(self, text: str) -> Dict[str, Any]:
        sandhi_issues = self.check_sandhi_errors(text)
        corrected = self.correct_sandhi(text)
        sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
        structure_results = [self.validate_sentence_structure(s) for s in sentences]

        avg_tamil_pct = sum(r.get("tamil_percentage", 0) for r in structure_results) / max(1, len(structure_results))

        return {
            "original": text,
            "corrected": corrected,
            "sandhi_error_count": len(sandhi_issues),
            "sandhi_issues": sandhi_issues,
            "sentences_checked": len(sentences),
            "tamil_percentage": round(avg_tamil_pct, 1),
            "is_grammatically_sound": len(sandhi_issues) == 0 and avg_tamil_pct >= 85.0
        }

# Global singleton instance
validator = TamilGrammarValidator()

if __name__ == "__main__":
    test_cases = [
        "அந்த புத்தகம் எனக்கு வேண்டும்.",
        "இந்த காலம் மிகவும் முக்கியமானது.",
        "அங்கு சென்றான் என் நண்பன்.",
        "பாடத்தை படி என்று ஆசிரியர் கூறினார்.",
        "வாடிக்கையாளருக்கு திட்டம் பற்றிய தகவல் கொடு.",
        "Naalaiku meeting varum."
    ]
    print("=== Testing Tamil Grammar & Sandhi Validator ===")
    for tc in test_cases:
        audit = validator.audit_text(tc)
        print(f"\nOriginal : {tc}")
        print(f"Corrected: {audit['corrected']}")
        print(f"Errors   : {audit['sandhi_error_count']} | Tamil Pct: {audit['tamil_percentage']}%")
