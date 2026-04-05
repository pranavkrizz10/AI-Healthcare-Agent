"""
extractor.py
============
NLP engine that converts a raw user sentence into a binary symptom vector
suitable for the trained ML classifier.

Architecture
------------
                 User Text
                     │
            ┌────────▼────────┐
            │  Text Cleaner   │  lowercase, strip punctuation/stopwords
            └────────┬────────┘
                     │  cleaned tokens
            ┌────────▼────────┐
            │  Exact Matcher  │  direct token / bigram / trigram hit
            └────────┬────────┘
                     │  matched + unmatched tokens
            ┌────────▼────────┐
            │  Fuzzy Matcher  │  rapidfuzz (edit-distance) for typos
            └────────┬────────┘
                     │  all found symptoms
            ┌────────▼────────┐
            │  spaCy / NLTK   │  optional — entity & dependency enrichment
            │  Enricher       │  ("pain in my chest" → chest pain)
            └────────┬────────┘
                     │  final symptom set
            ┌────────▼────────┐
            │ Vector Builder  │  maps symptom names → binary numpy array
            └────────┬────────┘
                     │
              numpy array (0s and 1s)

Dependencies
------------
    pip install spacy rapidfuzz nltk joblib
    python -m spacy download en_core_web_sm
    python -m nltk.downloader stopwords punkt

Usage (standalone)
------------------
    from src.extractor import SymptomExtractor
    ext = SymptomExtractor()
    vec, found = ext.extract("I have a high fever and feel very nauseous")
    print(found)   # ['fever', 'nausea']
    print(vec)     # array([0, 1, 0, 0, 1, ...])
"""

import re
import json
import os
import numpy as np

# ── Optional heavy deps (graceful degradation) ────────────────────────────────
try:
    import spacy
    _SPACY_MODEL = spacy.load("en_core_web_sm")
    _HAS_SPACY   = True
except Exception:
    _SPACY_MODEL = None
    _HAS_SPACY   = False

try:
    from rapidfuzz import process as fuzz_process, fuzz
    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False

try:
    from nltk.corpus import stopwords as _sw
    _STOPWORDS = set(_sw.words("english"))
except Exception:
    _STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "am", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "a",
        "an", "the", "and", "but", "or", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "than", "so", "if",
        "very", "just", "also", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "same",
        "then", "there", "these", "they", "this", "those", "through",
    }

# ── Symptom synonym / colloquial alias map ────────────────────────────────────
# Maps what a user might say → canonical symptom name from the dataset
SYNONYM_MAP: dict[str, str] = {
    # Fever variants
    "high temperature":    "high fever",
    "running a temperature": "fever",
    "temperature":         "fever",
    "pyrexia":             "fever",
    "feverish":            "fever",
    # Pain / ache
    "headache":            "headache",
    "head pain":           "headache",
    "migraine":            "headache",
    "sore throat":         "throat irritation",
    "scratchy throat":     "throat irritation",
    "stomach ache":        "stomach pain",
    "abdominal pain":      "stomach pain",
    "belly pain":          "stomach pain",
    "tummy ache":          "stomach pain",
    "chest pain":          "chest pain",
    "chest tightness":     "chest pain",
    "chest discomfort":    "chest pain",
    "back ache":           "back pain",
    "backache":            "back pain",
    "joint ache":          "joint pain",
    "achy joints":         "joint pain",
    "muscle ache":         "muscle pain",
    "body ache":           "muscle pain",
    "body pain":           "muscle pain",
    # Respiratory
    "runny nose":          "runny nose",
    "blocked nose":        "congestion",
    "stuffy nose":         "congestion",
    "nasal congestion":    "congestion",
    "shortness of breath": "breathlessness",
    "difficulty breathing":"breathlessness",
    "cant breathe":        "breathlessness",
    "hard to breathe":     "breathlessness",
    # GI
    "throwing up":         "vomiting",
    "sick to stomach":     "nausea",
    "upset stomach":       "nausea",
    "loose stools":        "diarrhoea",
    "watery stool":        "diarrhoea",
    "loose motions":       "diarrhoea",
    "diarrhea":            "diarrhoea",
    # General
    "tired":               "fatigue",
    "exhausted":           "fatigue",
    "no energy":           "fatigue",
    "weak":                "weakness",
    "feeling weak":        "weakness",
    "feel dizzy":          "dizziness",
    "dizzy":               "dizziness",
    "spinning":            "dizziness",
    "lightheaded":         "dizziness",
    "rash":                "skin rash",
    "itchy skin":          "itching",
    "yellow eyes":         "yellowing of eyes",
    "yellow skin":         "yellowish skin",
    "jaundice":            "yellowing of eyes",
    "swollen lymph":       "swollen lymph nodes",
    "neck swelling":       "swollen lymph nodes",
    "loss of appetite":    "loss of appetite",
    "not hungry":          "loss of appetite",
    "no appetite":         "loss of appetite",
    "night sweats":        "sweating",
    "excessive sweating":  "sweating",
    "chills":              "chills",
    "shivering":           "chills",
    "stiff neck":          "stiff neck",
    "neck stiffness":      "stiff neck",
    "sensitive to light":  "photophobia",
    "light sensitivity":   "photophobia",
}

# Red-flag symptoms that should trigger an urgent alert
RED_FLAGS: set[str] = {
    "chest pain", "breathlessness", "fainting", "loss of consciousness",
    "severe headache", "sudden numbness", "paralysis", "coughing blood",
    "blood in urine", "blood in stool", "high fever", "stiff neck",
    "photophobia", "seizure", "confusion", "irregular heartbeat",
}


# ── Main Extractor Class ──────────────────────────────────────────────────────
class SymptomExtractor:
    """
    Converts a natural-language sentence into:
      • a list of matched canonical symptom names
      • a binary numpy vector aligned to the full symptom vocabulary
      • a boolean flag for any red-flag symptoms detected
    """

    def __init__(self, symptom_list_path: str = os.path.join("models", "symptom_list.json")):
        self.symptom_list: list[str] = self._load_symptom_list(symptom_list_path)
        self._sym_set: set[str]      = set(self.symptom_list)
        # Pre-build ngram lookup: {ngram_string: canonical_name}
        self._ngram_lookup: dict[str, str] = self._build_ngram_lookup()

    # ── Setup helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _load_symptom_list(path: str) -> list[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Symptom list not found at '{path}'.\n"
                "Run  python src/preprocess.py  first."
            )
        with open(path) as f:
            raw = json.load(f)
        # BUG FIX 1: Normalize all symptom names from the JSON to use spaces
        # instead of underscores so they match tokenized user input.
        return [s.strip().lower().replace("_", " ") for s in raw]

    def _build_ngram_lookup(self) -> dict[str, str]:
        """
        Build a flat dict mapping every ngram variant → canonical symptom name.
        Canonical names are already space-normalized at load time.
        Also builds a position index for O(1) vector lookups.
        """
        lookup: dict[str, str] = {}

        # Add every canonical symptom name directly (spaces, already normalized)
        for sym in self.symptom_list:
            lookup[sym] = sym

        # BUG FIX 2: _sym_set is now space-normalized, so synonym canonical
        # targets will actually match. Add all synonyms whose target is known.
        for alias, canonical in SYNONYM_MAP.items():
            canonical_norm = canonical.strip().lower().replace("_", " ")
            if canonical_norm in self._sym_set:
                lookup[alias] = canonical_norm   # alias → normalized canonical

        # BUG FIX 3: Pre-build an index dict for O(1) vector position lookups
        self._sym_index: dict[str, int] = {s: i for i, s in enumerate(self.symptom_list)}

        return lookup

    # ── Text pre-processing ───────────────────────────────────────────────────
    @staticmethod
    def _clean_text(text: str) -> str:
        """Lowercase, remove punctuation (keep spaces), collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _spacy_enrich(self, text: str) -> str:
        """
        Use spaCy dependency parsing to reconstruct compound symptom phrases.
        E.g. "pain in my chest" → prepend "chest pain" to the text.
        This is a best-effort enrichment; falls back silently if spaCy fails.
        """
        if not _HAS_SPACY or _SPACY_MODEL is None:
            return text
        try:
            doc = _SPACY_MODEL(text)
            additions = []
            for token in doc:
                # Pattern: noun + "in" + body-part noun
                if (token.dep_ in ("pobj", "nmod") and
                        token.head.text in ("pain", "ache", "discomfort", "swelling")):
                    phrase = f"{token.text} {token.head.text}"
                    additions.append(phrase)
            if additions:
                text = text + " " + " ".join(additions)
        except Exception:
            pass
        return text

    # ── Matching logic ────────────────────────────────────────────────────────
    def _extract_ngrams(self, tokens: list[str]) -> list[str]:
        """
        Slide a 3→2→1 word window over the token list and return all
        ngrams that hit the ngram lookup table (longest match wins).
        """
        found    : list[str] = []
        used_idx : set[int]  = set()
        n        = len(tokens)

        for size in (3, 2, 1):          # longest match first
            for i in range(n - size + 1):
                if any(j in used_idx for j in range(i, i + size)):
                    continue            # tokens already consumed
                ngram = " ".join(tokens[i: i + size])
                if ngram in self._ngram_lookup:
                    canonical = self._ngram_lookup[ngram]
                    if canonical not in found:
                        found.append(canonical)
                    used_idx.update(range(i, i + size))

        return found

    def _fuzzy_match(self, tokens: list[str], already_found: set[str],
                     threshold: int = 82) -> list[str]:
        """
        For every single token NOT yet matched, attempt a fuzzy lookup
        against the full symptom vocabulary.
        Returns additional canonical matches above `threshold` similarity.
        """
        if not _HAS_FUZZ:
            return []
        extra: list[str] = []
        candidates = [t for t in tokens if t not in _STOPWORDS and len(t) > 3]
        for token in candidates:
            result = fuzz_process.extractOne(
                token,
                self.symptom_list,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold
            )
            if result:
                match, score, _ = result
                if match not in already_found:
                    extra.append(match)
        return extra

    # ── Public API ────────────────────────────────────────────────────────────
    def extract(self, user_text: str) -> tuple[np.ndarray, list[str], bool]:
        """
        Parameters
        ----------
        user_text : str
            Raw sentence from the user, e.g. "I have a bad headache and fever"

        Returns
        -------
        vector : np.ndarray  shape (n_symptoms,)
            Binary 0/1 array aligned with self.symptom_list
        found  : list[str]
            Canonical symptom names that were detected
        red_flag : bool
            True if any red-flag symptom is present
        """
        # 1. Optional spaCy enrichment
        enriched = self._spacy_enrich(user_text)

        # 2. Clean & tokenise
        cleaned = self._clean_text(enriched)
        tokens  = [t for t in cleaned.split() if t not in _STOPWORDS]

        # 3. Ngram (exact + synonym) matching
        found = self._extract_ngrams(tokens)

        # 4. Fuzzy fallback for typos / near-misses
        if _HAS_FUZZ:
            fuzzy_extra = self._fuzzy_match(tokens, already_found=set(found))
            found.extend(fuzzy_extra)

        # 5. Build binary vector
        vector = self._build_vector(found)

        # 6. Red-flag check
        red_flag = bool(set(found) & RED_FLAGS)

        return vector, found, red_flag

    def _build_vector(self, found_symptoms: list[str]) -> np.ndarray:
        """Map a list of symptom names to a binary numpy array using O(1) index lookup."""
        vec = np.zeros(len(self.symptom_list), dtype=np.float32)
        for sym in found_symptoms:
            idx = self._sym_index.get(sym)   # O(1) dict lookup, never throws
            if idx is not None:
                vec[idx] = 1.0
        return vec

    # ── Convenience ───────────────────────────────────────────────────────────
    def list_symptoms(self) -> list[str]:
        """Return the full sorted symptom vocabulary."""
        return self.symptom_list.copy()

    def is_red_flag(self, symptoms: list[str]) -> bool:
        return bool(set(symptoms) & RED_FLAGS)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Adjust path if running from project root
    ext = SymptomExtractor(symptom_list_path=os.path.join("models", "symptom_list.json"))

    test_sentences = [
        "I've had a high fever and terrible headache since yesterday",
        "My stomach is hurting, I keep vomiting, and I feel very tired",
        "I have chest pain and difficulty breathing",          # red flag!
        "Feeling a bit dizzy with a runny nose and sore throat",
        "I hav fatuge and a slight coff",                     # typos — fuzzy test
    ]

    for sentence in test_sentences:
        vec, found, flag = ext.extract(sentence)
        print(f"\nInput   : {sentence}")
        print(f"Symptoms: {found}")
        print(f"Red Flag: {'⚠️  YES — seek urgent care!' if flag else 'No'}")
        print(f"Vector  : {vec.astype(int).tolist()[:20]} …")