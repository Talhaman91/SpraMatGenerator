"""
Sprachlern-Framework – Streamlit POC (Single File)
- Modus: Alpha 3/4/5/6 oder Ohne Alpha
- Textlänge (Wörter) nur bei Ohne Alpha (0 = unbegrenzt)
- Alpha-Parameter sind gesperrt, wenn Alpha-Modus aktiv ist
- Ohne Alpha: Parameter frei (0 = aus)
- User Prompt:
  - Alpha-Modus: nennt Alpha-Level + verweist auf Algorithmus als RAG-Kontext
  - Ohne Alpha: nennt keinen Alpha-Level, nur Parameter (nur aktive Constraints)
- Keine Validierung
- Zusatzparameter (sekundär) enthalten, ohne die bestehenden Alpha-Mechaniken zu ändern
- Zusatzparameter jetzt stufenbasiert (keine Vorgabe/niedrig/mittel/hoch/sehr hoch)
- Tooltips (help=...) an allen Parametern
- Copy-Funktion entfernt

Install:
  pip install streamlit google-genai

Run:
  python -m streamlit run app.py
"""

from dataclasses import dataclass
import streamlit as st
import re
import math
from collections import Counter

# =========================
# LLM Fix-Konfiguration
# =========================
# GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL = "gemini-2.5-flash"

TENSES_ALL = ["Präsens", "Präteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]

TEXT_TYPES = [
    "Erzählung",
    "Sachtext",
    "Dialog",
    "E-Mail",
    "Nachrichtenberichterstattung",
]

# Zusatzparameter: Nebensatz-Typen (vereinfachte Liste)
SUBCLAUSE_TYPES = [
    "Relativsatz",
    "Kausalsatz",
    "Temporalsatz",
    "Konditionalsatz",
    "Konzessivsatz",
    "Finalsatz",
    "Objektsatz",
    "Subjektsatz",
]

# Zusatzparameter: Stufen (vereinfachte Bedienung)
LEVELS_4 = ["keine Vorgabe", "niedrig", "mittel", "hoch", "sehr hoch"]

# Interne Richtwerte (nur für Prompt, nicht als harte Validierung)
MTUL_BANDS = {
    "niedrig": "≤ 8 Wörter pro T-Unit",
    "mittel": "9–12 Wörter pro T-Unit",
    "hoch": "13–18 Wörter pro T-Unit",
    "sehr hoch": "> 18 Wörter pro T-Unit",
}

ZIPF_BANDS = {
    "niedrig": "sehr häufige Wörter (Zipf grob ≥ 5.5)",
    "mittel": "alltagsnah (Zipf grob ≥ 5.0)",
    "hoch": "differenzierter Wortschatz erlaubt (Zipf grob ≥ 4.5)",
    "sehr hoch": "keine Einschränkung",
}

LEXVAR_BANDS = {
    "niedrig": "geringe lexikalische Vielfalt, mehr Wiederholung",
    "mittel": "ausgewogene Vielfalt",
    "hoch": "hohe Vielfalt erlaubt",
    "sehr hoch": "keine Einschränkung",
}

CONNECTOR_BANDS = {
    "niedrig": "wenige Konnektoren (z. B. 0–3)",
    "mittel": "moderate Konnektoren (z. B. 4–8)",
    "hoch": "viele Konnektoren erlaubt (z. B. 9–15)",
    "sehr hoch": "keine Einschränkung",
}


# =========================
# Helper
# =========================
def optional_float_or_none(value) -> float | None:
    """
    Streamlit number_input kann None/float/int liefern.
    Regel:
    - None -> None
    - 0.0 -> None (bedeutet: Regel deaktiviert)
    - sonst -> float(value)
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v == 0.0 else v


def optional_int_or_none(value) -> int | None:
    """
    Regel:
    - None -> None
    - 0 -> None (bedeutet: Regel deaktiviert)
    - sonst -> int(value)
    """
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return None if v == 0 else v


def clamp_level(value: str) -> str:
    if value in LEVELS_4:
        return value
    return "keine Vorgabe"




# =========================
# Datenmodelle
# =========================
@dataclass
class GeneralParams:
    topic: str
    text_type: str
    target_words: int | None  # nur bei Ohne Alpha; None = keine Begrenzung


@dataclass
class AlphaParams:
    mode: str  # "Alpha 3" | "Alpha 4" | "Alpha 5" | "Alpha 6" | "Ohne Alpha"

    max_sentences: int  # 0 = aus (nur Ohne Alpha)
    max_words_per_sentence: int  # 0 = aus (nur Ohne Alpha)
    max_syllables_per_token: int  # 0 = aus (nur Ohne Alpha)
    max_dep_clauses_per_sentence: float  # 0.0 = aus (nur Ohne Alpha)

    forbidden_tenses: list[str]
    max_perfekt_per_finite_verb: float | None
    min_lexical_coverage: float | None


@dataclass
class FineParams:
    """
    Zusatzparameter (sekundär): sollen das Alpha-Level NICHT ändern,
    sondern innerhalb eines Levels fein steuern.
    """
    enabled: bool

    mtul_level: str
    zipf_level: str
    lexvar_level: str
    connectors_level: str

    forbidden_subclause_types: list[str]
    konjunktiv_mode: str  # "keine Vorgabe" | "erlauben" | "vermeiden"
    coherence_hint: str  # "keine" | "hoch" | "mittel" | "niedrig"


@dataclass
class Params:
    general: GeneralParams
    alpha: AlphaParams
    fine: FineParams

# =========================
# Validation / Analyse (Heuristiken)
# =========================

_VOWELS = "aeiouyäöüAEIOUYÄÖÜ"


def _sent_split(text: str) -> list[str]:
    # einfache Satzsegmentierung (Heuristik)
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> list[str]:
    # einfache Tokenisierung: Wörter inkl. deutscher Umlaute
    return re.findall(r"[A-Za-zÄÖÜäöüß]+(?:'[A-Za-zÄÖÜäöüß]+)?", text or "")


def _count_syllables(word: str) -> int:
    # Heuristik: Zähle Vokalgruppen
    w = word.strip()
    if not w:
        return 0
    groups = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            groups += 1
        prev_vowel = is_v
    return max(1, groups)


def _ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    low = [t.lower() for t in tokens]
    return len(set(low)) / len(low)


def _mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """
    MTLD Heuristik (token-basiert).
    Quelle für Idee: McCarthy & Jarvis (2010). (In deiner BA sauber als Näherung kennzeichnen.)
    """
    low = [t.lower() for t in tokens if t.strip()]
    if len(low) < 10:
        return 0.0

    def mtld_one_pass(seq: list[str]) -> float:
        factors = 0
        types = set()
        tok_count = 0
        for tok in seq:
            tok_count += 1
            types.add(tok)
            cur_ttr = len(types) / tok_count
            if cur_ttr <= ttr_threshold:
                factors += 1
                types = set()
                tok_count = 0
        if tok_count == 0:
            return float(factors) if factors > 0 else 0.0
        # partial factor
        cur_ttr = len(types) / tok_count
        partial = (1 - cur_ttr) / (1 - ttr_threshold) if (1 - ttr_threshold) != 0 else 0.0
        return factors + partial

    fwd = mtld_one_pass(low)
    bwd = mtld_one_pass(list(reversed(low)))
    denom = (fwd + bwd) / 2.0
    if denom <= 0:
        return 0.0
    return len(low) / denom


def _mtul_approx(sentences: list[str]) -> float:
    """
    MTUL Approximation:
    - T-Unit Erkennung ist linguistisch schwer ohne Parser.
    - Heuristik: pro Satz splitte an ; und Koordinationen (und/aber/denn/sondern) grob.
    """
    if not sentences:
        return 0.0
    t_units = 0
    words_total = 0
    for s in sentences:
        toks = _tokens(s)
        words_total += len(toks)

        # grobe T-Unit Splits
        # +1 pro Satz
        units = 1
        units += len(re.findall(r";", s))
        units += len(re.findall(r"\b(und|aber|denn|sondern)\b", s, flags=re.IGNORECASE)) // 2
        t_units += max(1, units)
    return words_total / t_units if t_units > 0 else 0.0


def _zipf_approx(tokens: list[str]) -> float | None:
    """
    Optional: Zipf über wordfreq (wenn installiert).
    Ohne Paket -> None.
    """
    try:
        from wordfreq import zipf_frequency  # type: ignore
    except Exception:
        return None

    lows = [t.lower() for t in tokens if t.strip()]
    if not lows:
        return None
    vals = [zipf_frequency(w, "de") for w in lows]
    return sum(vals) / len(vals)


def analyze_text(text: str) -> dict:
    sents = _sent_split(text)
    toks = _tokens(text)

    word_count = len(toks)
    sent_count = len(sents)

    # words per sentence
    wps_list = [len(_tokens(s)) for s in sents] if sents else []
    max_wps = max(wps_list) if wps_list else 0
    avg_wps = (sum(wps_list) / len(wps_list)) if wps_list else 0.0

    # syllables per token
    syl_list = [_count_syllables(t) for t in toks] if toks else []
    max_syl = max(syl_list) if syl_list else 0
    avg_syl = (sum(syl_list) / len(syl_list)) if syl_list else 0.0

    # dep clauses per sentence: ohne Parser nur Näherung
    # Heuristik: zähle Subjunktoren/Relativpronomen grob
    dep_markers = re.findall(
        r"\b(dass|weil|obwohl|wenn|als|während|damit|ob|bevor|nachdem|bis|seit|sobald|der|die|das|welcher|welche|welches)\b",
        text,
        flags=re.IGNORECASE,
    )
    dep_per_sent = (len(dep_markers) / sent_count) if sent_count > 0 else 0.0

    # lexical
    ttr_val = _ttr(toks)
    mtld_val = _mtld(toks)
    mtul_val = _mtul_approx(sents)
    zipf_val = _zipf_approx(toks)

    return {
        "word_count": word_count,
        "sent_count": sent_count,
        "max_words_per_sentence": max_wps,
        "avg_words_per_sentence": avg_wps,
        "max_syllables_per_token": max_syl,
        "avg_syllables_per_token": avg_syl,
        "dep_clauses_per_sentence_approx": dep_per_sent,
        "ttr": ttr_val,
        "mtld": mtld_val,
        "mtul_approx": mtul_val,
        "zipf_avg": zipf_val,
    }


def validate_against_alpha(mode: str, metrics: dict) -> dict:
    """
    Validiert Kernmetriken gegen die Alpha-Presets (die im Tool geladen sind).
    Hinweis: Einige Originalregeln (Tempus-/Verb-Ratios) werden hier NICHT robust berechnet.
    """
    if mode not in ALPHA_PRESETS:
        return {"mode": mode, "overall": None, "checks": []}

    preset = ALPHA_PRESETS[mode]
    checks = []

    def add_check(name: str, value, op: str, target, ok: bool, note: str = ""):
        checks.append(
            {
                "name": name,
                "value": value,
                "op": op,
                "target": target,
                "ok": ok,
                "note": note,
            }
        )

    # words per sentence (max)
    add_check(
        "wordsPerSentence (max)",
        metrics["max_words_per_sentence"],
        "<=",
        preset["max_words_per_sentence"],
        metrics["max_words_per_sentence"] <= preset["max_words_per_sentence"],
    )

    # nSentences
    add_check(
        "nSentences",
        metrics["sent_count"],
        "<=",
        preset["max_sentences"],
        metrics["sent_count"] <= preset["max_sentences"],
    )

    # syllables per token (max)
    add_check(
        "syllablesPerToken (max)",
        metrics["max_syllables_per_token"],
        "<=",
        preset["max_syllables_per_token"],
        metrics["max_syllables_per_token"] <= preset["max_syllables_per_token"],
        note="Silbenzählung ist Heuristik.",
    )

    # dep clauses per sentence (approx)
    add_check(
        "depClausesPerSentence (approx)",
        round(metrics["dep_clauses_per_sentence_approx"], 3),
        "<=",
        preset["max_dep_clauses_per_sentence"],
        metrics["dep_clauses_per_sentence_approx"] <= preset["max_dep_clauses_per_sentence"],
        note="Ohne Parser nur Näherung über Marker.",
    )

    overall = all(c["ok"] for c in checks) if checks else None
    return {"mode": mode, "overall": overall, "checks": checks}


def build_validation_report(params: Params, text: str) -> str:
    m = analyze_text(text)
    a = params.alpha
    f = params.fine

    lines: list[str] = []
    lines.append("VALIDIERUNG (Heuristik, keine Gold-Standard-Auswertung)")
    lines.append("")

    # Basics
    lines.append(f"Wortanzahl: {m['word_count']}")
    lines.append(f"Satzanzahl: {m['sent_count']}")
    lines.append(f"Max Wörter/Satz: {m['max_words_per_sentence']} (Ø {m['avg_words_per_sentence']:.2f})")
    lines.append(f"Max Silben/Token: {m['max_syllables_per_token']} (Ø {m['avg_syllables_per_token']:.2f})")
    lines.append(f"Dep-Nebensätze/Satz (approx): {m['dep_clauses_per_sentence_approx']:.3f}")
    lines.append("")

    # Lexical / fine metrics (immer anzeigen)
    lines.append(f"MTUL (approx): {m['mtul_approx']:.2f}")
    lines.append(f"TTR: {m['ttr']:.3f}")
    lines.append(f"MTLD (approx): {m['mtld']:.2f}")
    if m["zipf_avg"] is None:
        lines.append("Zipf (avg): n/a (optional: pip install wordfreq)")
    else:
        lines.append(f"Zipf (avg): {m['zipf_avg']:.2f}")
    lines.append("")

    # Alpha validation
    if a.mode in ALPHA_PRESETS:
        res = validate_against_alpha(a.mode, m)
        status = "BESTANDEN ✅" if res["overall"] else "NICHT BESTANDEN ❌"
        lines.append(f"Alpha-Validierung: {a.mode} → {status}")
        for c in res["checks"]:
            ok = "ok" if c["ok"] else "FAIL"
            extra = f" ({c['note']})" if c.get("note") else ""
            lines.append(f"- {c['name']}: {c['value']} {c['op']} {c['target']} → {ok}{extra}")
    else:
        lines.append("Alpha-Validierung: n/a (Ohne Alpha)")
    lines.append("")

    # Parameter-Listing (immer anzeigen)
    lines.append("Parameter-Übersicht (aktuelle UI-Werte):")
    lines.append(f"- Mode: {a.mode}")
    lines.append(f"- Max Sätze: {a.max_sentences} (0 = aus)")
    lines.append(f"- Max Wörter/Satz: {a.max_words_per_sentence} (0 = aus)")
    lines.append(f"- Max Silben/Token: {a.max_syllables_per_token} (0 = aus)")
    lines.append(f"- Max Dep-Nebensätze/Satz: {a.max_dep_clauses_per_sentence} (0 = aus)")
    lines.append(f"- Verbotene Tempora: {', '.join(a.forbidden_tenses) if a.forbidden_tenses else 'keine'}")
    lines.append(f"- Max Perfekt/finite Verben: {a.max_perfekt_per_finite_verb if a.max_perfekt_per_finite_verb is not None else 'aus'}")
    lines.append(f"- Min lexikalische Abdeckung: {a.min_lexical_coverage if a.min_lexical_coverage is not None else 'aus'}")
    lines.append("")

    # Fine settings (immer anzeigen, auch wenn disabled)
    lines.append("Zusatzparameter (Stufen / Einstellungen):")
    lines.append(f"- Aktiviert: {f.enabled}")
    lines.append(f"- MTUL-Stufe: {f.mtul_level}")
    lines.append(f"- Zipf-Stufe: {f.zipf_level}")
    lines.append(f"- LexVar-Stufe: {f.lexvar_level}")
    lines.append(f"- Konnektoren-Stufe: {f.connectors_level}")
    lines.append(f"- Konjunktiv: {f.konjunktiv_mode}")
    lines.append(f"- Kohärenz: {f.coherence_hint}")
    lines.append(f"- Vermeidbare Nebensatzarten: {', '.join(f.forbidden_subclause_types) if f.forbidden_subclause_types else 'keine'}")

    return "\n".join(lines)


# =========================
# "RAG Kontext" (Algorithmus-Block als retrieved context)
# =========================
def rag_context_for_alpha(mode: str) -> str:
    """
    Liefert einen kurzen "retrieved context"-Block, der im Prompt als RAG-Kontext dient.
    Kein vollständiges Paper-Zitat, sondern eine knappe, zweckmäßige Zusammenfassung als Kontext.
    """
    if mode == "Alpha 3":
        return (
            "Retrieved Context (Weiß et al., Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 10\n"
            "- nSentences <= 5\n"
            "- syllablesPerToken <= 3\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
            "- future1sPerFiniteVerb == 0\n"
            "- future2sPerFiniteVerb == 0\n"
            "- depClausesPerSentence <= 0.5\n"
            "- presentPerfectsPerFiniteVerb <= 0.5\n"
            "- typesFoundInSubtlexPerLexicalType >= 0.95\n"
        )
    if mode == "Alpha 4":
        return (
            "Retrieved Context (Weiß et al., Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 10\n"
            "- nSentences <= 10\n"
            "- syllablesPerToken <= 5\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
            "- future1sPerFiniteVerb == 0\n"
            "- future2sPerFiniteVerb == 0\n"
        )
    if mode == "Alpha 5":
        return (
            "Retrieved Context (Weiß et al., Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 12\n"
            "- nSentences <= 15\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
        )
    if mode == "Alpha 6":
        return (
            "Retrieved Context (Weiß et al., Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 12\n"
            "- nSentences <= 20\n"
        )
    return ""


def rag_context_for_fine_params() -> str:
    """
    Kompaktes Glossar (Pseudo-RAG) für Zusatzparameter, damit das Modell Begriffe konsistent interpretiert.
    Bewusst kurz, um Prompt-Länge zu begrenzen.
    """
    return (
        "Retrieved Context (Zusatzparameter-Glossar):\n"
        "| Parameter | Bedeutung | Intention |\n"
        "|---|---|---|\n"
        "| MTUL | mittlere Wörter pro T-Unit | kürzere syntaktische Einheiten |\n"
        "| LexVar | lexikalische Vielfalt (TTR/MTLD zusammengefasst) | niedrig = mehr Wiederholung |\n"
        "| Zipf | Wortfrequenz (1–7) | höher = häufigere Wörter |\n"
        "| Konnektoren | z.B. weil, aber, obwohl | weniger explizite Verknüpfungen |\n"
        "| Konjunktiv | Modus (I/II) | vermeiden = morphologisch einfacher |\n"
        "| Nebensatzarten | Relativsatz, Kausalsatz, ... | bestimmte Strukturen vermeiden |\n"
        "| Kohärenz | logischer Zusammenhang | klar nachvollziehbare Bezüge |\n"
    )


# =========================
# Alpha Presets
# =========================
ALPHA_PRESETS = {
    "Alpha 3": dict(
        max_sentences=5,
        max_words_per_sentence=10,
        max_syllables_per_token=3,
        max_dep_clauses_per_sentence=0.5,
        forbidden_tenses=["Plusquamperfekt", "Futur I", "Futur II"],
        max_perfekt_per_finite_verb=0.5,
        min_lexical_coverage=0.95,
    ),
    "Alpha 4": dict(
        max_sentences=10,
        max_words_per_sentence=10,
        max_syllables_per_token=5,
        max_dep_clauses_per_sentence=1.0,
        forbidden_tenses=["Plusquamperfekt", "Futur I", "Futur II"],
        max_perfekt_per_finite_verb=None,
        min_lexical_coverage=None,
    ),
    "Alpha 5": dict(
        max_sentences=15,
        max_words_per_sentence=12,
        max_syllables_per_token=5,
        max_dep_clauses_per_sentence=1.5,
        forbidden_tenses=["Plusquamperfekt"],
        max_perfekt_per_finite_verb=None,
        min_lexical_coverage=None,
    ),
    "Alpha 6": dict(
        max_sentences=20,
        max_words_per_sentence=12,
        max_syllables_per_token=6,
        max_dep_clauses_per_sentence=2.0,
        forbidden_tenses=[],
        max_perfekt_per_finite_verb=None,
        min_lexical_coverage=None,
    ),
}

FREE_DEFAULTS = dict(
    max_sentences=8,
    max_words_per_sentence=18,
    max_syllables_per_token=6,
    max_dep_clauses_per_sentence=2.0,
    forbidden_tenses=[],
    max_perfekt_per_finite_verb=None,
    min_lexical_coverage=None,
)


# =========================
# Session State Init / Preset Anwendung
# =========================
def ensure_defaults_exist() -> None:
    st.session_state.setdefault("mode", "Alpha 4")

    st.session_state.setdefault("topic", "Alltag")
    st.session_state.setdefault("text_type", "Sachtext")
    st.session_state.setdefault("target_words", 140)  # nur genutzt bei Ohne Alpha (0 = unbegrenzt)

    # Alpha Parameter defaults (frei)
    st.session_state.setdefault("alpha_max_sentences", FREE_DEFAULTS["max_sentences"])
    st.session_state.setdefault("alpha_max_words_per_sentence", FREE_DEFAULTS["max_words_per_sentence"])
    st.session_state.setdefault("alpha_max_syllables_per_token", FREE_DEFAULTS["max_syllables_per_token"])
    st.session_state.setdefault("alpha_max_dep_clauses_per_sentence", FREE_DEFAULTS["max_dep_clauses_per_sentence"])
    st.session_state.setdefault("alpha_forbidden_tenses", FREE_DEFAULTS["forbidden_tenses"])

    # Optional: UI-seitig 0.0 = aus
    st.session_state.setdefault("alpha_max_perfekt_per_finite_verb", 0.0)
    st.session_state.setdefault("alpha_min_lexical_coverage", 0.0)

    # Zusatzparameter Defaults (stufenbasiert)
    st.session_state.setdefault("fine_enabled", False)
    st.session_state.setdefault("fine_mtul_level", "keine Vorgabe")
    st.session_state.setdefault("fine_zipf_level", "keine Vorgabe")
    st.session_state.setdefault("fine_lexvar_level", "keine Vorgabe")
    st.session_state.setdefault("fine_connectors_level", "keine Vorgabe")
    st.session_state.setdefault("fine_forbidden_subclause_types", [])
    st.session_state.setdefault("fine_konjunktiv_mode", "keine Vorgabe")
    st.session_state.setdefault("fine_coherence_hint", "keine")


def apply_preset_if_alpha(mode: str) -> None:
    st.session_state["mode"] = mode
    if mode in ALPHA_PRESETS:
        preset = ALPHA_PRESETS[mode]
        st.session_state["alpha_max_sentences"] = preset["max_sentences"]
        st.session_state["alpha_max_words_per_sentence"] = preset["max_words_per_sentence"]
        st.session_state["alpha_max_syllables_per_token"] = preset["max_syllables_per_token"]
        st.session_state["alpha_max_dep_clauses_per_sentence"] = preset["max_dep_clauses_per_sentence"]
        st.session_state["alpha_forbidden_tenses"] = preset["forbidden_tenses"]

        st.session_state["alpha_max_perfekt_per_finite_verb"] = (
            0.0 if preset["max_perfekt_per_finite_verb"] is None else float(preset["max_perfekt_per_finite_verb"])
        )
        st.session_state["alpha_min_lexical_coverage"] = (
            0.0 if preset["min_lexical_coverage"] is None else float(preset["min_lexical_coverage"])
        )


def on_mode_change() -> None:
    apply_preset_if_alpha(st.session_state["mode"])


# =========================
# Prompts
# =========================
def build_system_prompt() -> str:
    return """Du bist ein erfahrener L2-Lehrer und Sprachexperte für Deutsch. Du erstellst Sprachlernmaterialien in Form zusammenhängender Texte.

Regeln:
- Gib ausschließlich den fertigen Text aus.
- Keine Überschrift, keine Bulletpoints, keine Erklärungen, keine Metakommentare.
- Halte dich an die Vorgaben im User Prompt so gut wie möglich.
- Authentizität: Schreibe natürlich, alltagsnah und plausibel. Die Inhalte sollen realistisch wirken, so als wären sie von Muttersprachlern für Muttersprachler geschrieben.
- Vermeide erfundene, spezifische Fakten (z. B. konkrete Statistiken, Studien, offizielle Zahlen, reale Adressen), außer sie sind für die Aufgabe notwendig oder wurden vom Nutzer vorgegeben.
- Wenn Vorgaben widersprüchlich sind, löse sie sinnvoll auf: Priorisiere Verständlichkeit, Natürlichkeit und realistische Inhalte.
"""


def fine_params_to_prompt_lines(f: FineParams) -> list[str]:
    """
    Zusatzparameter als Prompt-Zeilen (stufenbasiert).
    Stufen sind Richtwerte und werden als weiche Ziele formuliert.
    """
    if not f.enabled:
        return []

    lines: list[str] = []

    # MTUL
    if f.mtul_level != "keine Vorgabe":
        band = MTUL_BANDS.get(f.mtul_level, f.mtul_level)
        lines.append(f"- MTUL: {f.mtul_level}. Richtwert: {band}. (bestmöglich)")

    # Nebensatzarten
    if f.forbidden_subclause_types:
        forbid = ", ".join(f.forbidden_subclause_types)
        lines.append(f"- Vermeide folgende Nebensatzarten: {forbid}.")

    # Zipf
    if f.zipf_level != "keine Vorgabe":
        band = ZIPF_BANDS.get(f.zipf_level, f.zipf_level)
        lines.append(f"- Lexik/Wortfrequenz (Zipf): {f.zipf_level}. Ziel: {band}. (bestmöglich)")

    # Lexikalische Vielfalt
    if f.lexvar_level != "keine Vorgabe":
        band = LEXVAR_BANDS.get(f.lexvar_level, f.lexvar_level)
        lines.append(f"- Lexikalische Vielfalt: {f.lexvar_level}. Ziel: {band}. (bestmöglich)")

    # Konjunktiv
    if f.konjunktiv_mode == "erlauben":
        lines.append("- Konjunktiv I/II: erlaubt (wenn passend).")
    elif f.konjunktiv_mode == "vermeiden":
        lines.append("- Konjunktiv I/II: vermeiden.")

    # Konnektoren
    if f.connectors_level != "keine Vorgabe":
        band = CONNECTOR_BANDS.get(f.connectors_level, f.connectors_level)
        lines.append(f"- Konnektoren: {f.connectors_level}. Richtwert: {band}. (bestmöglich)")

    # Kohärenz
    if f.coherence_hint and f.coherence_hint != "keine":
        lines.append(f"- Kohärenz: {f.coherence_hint} (logisch gut nachvollziehbar, klare Bezüge).")

    return lines


def build_user_prompt(p: Params) -> str:
    g = p.general
    a = p.alpha
    f = p.fine

    forbidden = ", ".join(a.forbidden_tenses) if a.forbidden_tenses else "keine"
    fine_lines = fine_params_to_prompt_lines(f)
    fine_ctx = rag_context_for_fine_params() if fine_lines else ""

    # Ohne Alpha: kein Alpha im Prompt, Textlänge ist optional (None = unbegrenzt)
    if a.mode == "Ohne Alpha":
        lines: list[str] = [
            f'Thema: "{g.topic}".',
            f"Textart: {g.text_type}.",
        ]

        if g.target_words is not None:
            lines.append(f"Textlänge: ca. {g.target_words} Wörter.")

        lines += [
            "",
            "Parameter (bitte so gut wie möglich einhalten):",
        ]

        # Nur aktive Constraints ausgeben (0 = aus)
        if a.max_sentences > 0:
            lines.append(f"- Anzahl Sätze: höchstens {a.max_sentences}.")
        if a.max_words_per_sentence > 0:
            lines.append(f"- Wörter pro Satz: höchstens {a.max_words_per_sentence}.")
        if a.max_syllables_per_token > 0:
            lines.append(f"- Silben pro Token: höchstens {a.max_syllables_per_token}.")
        if a.max_dep_clauses_per_sentence > 0.0:
            lines.append(f"- Dependenznebensätze pro Satz: höchstens {a.max_dep_clauses_per_sentence}.")
        lines.append(f"- Verbotene Tempora: {forbidden}.")

        if a.max_perfekt_per_finite_verb is not None:
            lines.append(f"- Perfekt pro finitem Verb: höchstens {a.max_perfekt_per_finite_verb}.")
        if a.min_lexical_coverage is not None:
            lines.append(f"- Lexik: nutze sehr häufige Wörter; Ziel-Abdeckung mindestens {a.min_lexical_coverage}.")

        if fine_ctx:
            lines += [
                "",
                "Beziehe dich für die folgenden Zusatzparameter auf das Glossar (Retrieved Context):",
                "",
                fine_ctx.strip(),
            ]
        if fine_lines:
            lines += [
                "",
                "Zusatzparameter (sekundär, ändern kein Alpha-Level):",
                *fine_lines,
            ]

        lines += [
            "",
            "Konsistenzregel:",
            "- Falls Textlänge und Satzanzahl nicht zusammenpassen, wähle eine sinnvolle Satzanzahl, "
            "die einen natürlichen Text ermöglicht, und bleibe dabei möglichst nahe an den Vorgaben.",
            "",
            "Stil:",
            "- Klar, alltagsnah, keine Schachtelsätze.",
            "- Keine Metakommentare.",
        ]
        return "\n".join(lines)

    # Alpha-Modus: unverändert (keine 0=aus-Logik nötig, weil Presets > 0)
    rag_ctx = rag_context_for_alpha(a.mode)

    lines = [
        f"ALPHA-LEVEL: {a.mode}",
        "Beziehe dich auf den folgenden Algorithmus-Kontext (RAG) und halte die Parameter so gut wie möglich ein.",
        "",
        rag_ctx.strip(),
        "",
        f'Thema: "{g.topic}".',
        f"Textart: {g.text_type}.",
        "",
        "Parameter (bitte so gut wie möglich einhalten):",
        f"- Anzahl Sätze: höchstens {a.max_sentences}.",
        f"- Wörter pro Satz: höchstens {a.max_words_per_sentence}.",
        f"- Silben pro Token: höchstens {a.max_syllables_per_token}.",
        f"- Dependenznebensätze pro Satz: höchstens {a.max_dep_clauses_per_sentence}.",
        f"- Verbotene Tempora: {forbidden}.",
    ]

    if a.max_perfekt_per_finite_verb is not None:
        lines.append(f"- Perfekt pro finitem Verb: höchstens {a.max_perfekt_per_finite_verb}.")
    if a.min_lexical_coverage is not None:
        lines.append(f"- Lexik: nutze sehr häufige Wörter; Ziel-Abdeckung mindestens {a.min_lexical_coverage}.")

    if fine_ctx:
        lines += [
            "",
            "Beziehe dich für die folgenden Zusatzparameter auf das Glossar (Retrieved Context):",
            "",
            fine_ctx.strip(),
        ]
    if fine_lines:
        lines += [
            "",
            "Zusatzparameter (sekundär, ändern dieses Alpha-Level nicht):",
            *fine_lines,
        ]

    lines += [
        "",
        "Stil:",
        "- Klar, alltagsnah, keine Schachtelsätze.",
        "- Keine Metakommentare.",
    ]
    return "\n".join(lines)


# =========================
# Gemini Call
# =========================
def gemini_generate(api_key: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    if not api_key.strip():
        raise RuntimeError("Kein Gemini API Key gesetzt.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(system_instruction=system_prompt, temperature=float(temperature))

    resp = client.models.generate_content(model=GEMINI_MODEL, contents=user_prompt, config=cfg)

    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Gemini hat keinen Text geliefert.")
    return text.strip()


# =========================
# UI
# =========================
st.set_page_config(page_title="Sprachlern POC", layout="wide")
st.title("Sprachlern Framework POC")
st.caption("Modus: Alpha 3–6 oder Ohne Alpha · Ohne Alpha: 0 = aus · Keine Validierung · Zusatzparameter verfügbar")

ensure_defaults_exist()

# Sidebar
with st.sidebar:
    st.header("LLM")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="API Key für Google Gemini.",
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        1.5,
        0.7,
        0.1,
        help="Steuert die Zufälligkeit der Generierung: niedrig = stabiler/regelkonformer, hoch = variantenreicher.",
    )

    st.divider()

    # ===== Allgemein oben =====
    with st.expander("Allgemein", expanded=True):
        st.text_input(
            "Thema",
            key="topic",
            help="Übergeordnetes Thema des Textes, z. B. Alltag, Arbeit, Freizeit oder Gesundheit.",
        )
        st.selectbox(
            "Textart",
            TEXT_TYPES,
            index=TEXT_TYPES.index(st.session_state["text_type"]) if st.session_state["text_type"] in TEXT_TYPES else 1,
            key="text_type",
            help="Kommunikative Textsorte, die Struktur, Stil und typische Formulierungen beeinflusst.",
        )

        if st.session_state["mode"] == "Ohne Alpha":
            st.number_input(
                "Textlänge (Wörter, 0 = unbegrenzt)",
                min_value=0,
                max_value=2000,
                key="target_words",
                step=10,
                help="Zielwortanzahl des Textes. 0 bedeutet: keine Vorgabe zur Textlänge.",
            )
        else:
            st.caption("Textlänge ist im Alpha-Modus deaktiviert (wird aus den Alpha-Constraints abgeleitet).")

    # ===== Alpha-Parameter darunter =====
    with st.expander("Alpha-Parameter", expanded=True):
        st.selectbox(
            "Modus",
            ["Alpha 3", "Alpha 4", "Alpha 5", "Alpha 6", "Ohne Alpha"],
            key="mode",
            on_change=on_mode_change,
            help="Alpha 3–6 lädt feste Presets (Parameter gesperrt). »Ohne Alpha« erlaubt freie Alpha-Parameter (0 = aus).",
        )

        mode = st.session_state["mode"]
        alpha_locked = (mode != "Ohne Alpha")

        st.number_input(
            "Max Sätze (0 = aus)",
            min_value=0,
            max_value=100,
            key="alpha_max_sentences",
            disabled=alpha_locked,
            help="Maximale Anzahl der Sätze im Text. In Alpha-Modi fest vorgegeben. In »Ohne Alpha« deaktiviert 0 die Begrenzung.",
        )
        st.number_input(
            "Max Wörter pro Satz (0 = aus)",
            min_value=0,
            max_value=60,
            key="alpha_max_words_per_sentence",
            disabled=alpha_locked,
            help="Maximale Satzlänge in Wörtern als Maß für syntaktische Einfachheit. 0 deaktiviert die Begrenzung (nur »Ohne Alpha«).",
        )
        st.number_input(
            "Max Silben pro Token (0 = aus)",
            min_value=0,
            max_value=12,
            key="alpha_max_syllables_per_token",
            disabled=alpha_locked,
            help="Begrenzung der Silben pro Wort (Token) als grobes Maß für Wortkomplexität. 0 deaktiviert die Begrenzung (nur »Ohne Alpha«).",
        )

        current_dep = float(st.session_state.get("alpha_max_dep_clauses_per_sentence", 1.0) or 0.0)
        dep_max = max(10.0, current_dep)
        st.slider(
            "Max Dep-Nebensätze pro Satz (0 = aus)",
            0.0,
            dep_max,
            key="alpha_max_dep_clauses_per_sentence",
            step=0.5,
            disabled=alpha_locked,
            help="Begrenzung der syntaktischen Subordination (Nebensätze) pro Satz. 0 deaktiviert die Begrenzung (nur »Ohne Alpha«).",
        )

        st.multiselect(
            "Verbotene Tempora",
            options=TENSES_ALL,
            key="alpha_forbidden_tenses",
            disabled=alpha_locked,
            help="Tempora, die im Text nicht verwendet werden sollen (z. B. zur Reduktion morphologischer Komplexität).",
        )

        st.number_input(
            "Max Perfekt/finite Verben (0 = aus)",
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            key="alpha_max_perfekt_per_finite_verb",
            disabled=alpha_locked,
            help="Maximales Verhältnis von Perfektformen zu finiten Verben. 0 deaktiviert die Vorgabe.",
        )
        st.number_input(
            "Min lexikalische Abdeckung (0 = aus)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="alpha_min_lexical_coverage",
            disabled=alpha_locked,
            help="Mindestanteil hochfrequenter Wörter (z. B. Subtlex-Abdeckung). 0 deaktiviert die Vorgabe.",
        )

    # ===== Zusatzparameter (sekundär) =====
    with st.expander("Zusatzparameter (sekundär, ändern Alpha nicht)", expanded=False):
        st.checkbox(
            "Zusatzparameter aktivieren",
            key="fine_enabled",
            help="Wenn deaktiviert, werden Zusatzparameter nicht in den Prompt aufgenommen.",
        )

        fine_disabled = not st.session_state.get("fine_enabled", False)

        st.selectbox(
            "MTUL-Komplexität",
            LEVELS_4,
            key="fine_mtul_level",
            disabled=fine_disabled,
            help="MTUL = mittlere Wörter pro T-Unit. Stufen sind Richtwerte (keine harte Validierung).",
        )

        st.multiselect(
            "Zu vermeidende Nebensatzarten",
            options=SUBCLAUSE_TYPES,
            key="fine_forbidden_subclause_types",
            disabled=fine_disabled,
            help="Bestimmte Typen von Nebensätzen, die im Text vermieden werden sollen.",
        )

        st.selectbox(
            "Wortfrequenz (Zipf) – Stufe",
            LEVELS_4,
            key="fine_zipf_level",
            disabled=fine_disabled,
            help="Zipf (1–7): höher = häufigere Wörter. Stufen sind Richtwerte.",
        )

        st.selectbox(
            "Lexikalische Vielfalt (TTR/MTLD) – Stufe",
            LEVELS_4,
            key="fine_lexvar_level",
            disabled=fine_disabled,
            help="Fasst TTR/MTLD als intuitive Stufe zusammen. Stufen sind Richtwerte.",
        )

        st.selectbox(
            "Konnektoren-Dichte – Stufe",
            LEVELS_4,
            key="fine_connectors_level",
            disabled=fine_disabled,
            help="Steuert, wie viele explizite Konnektoren (weil, aber, obwohl ...) genutzt werden sollen.",
        )

        st.selectbox(
            "Konjunktiv I/II",
            ["keine Vorgabe", "erlauben", "vermeiden"],
            key="fine_konjunktiv_mode",
            disabled=fine_disabled,
            help="Steuert, ob Konjunktivformen genutzt oder vermieden werden sollen.",
        )

        st.selectbox(
            "Kohärenz-Hinweis",
            ["keine", "hoch", "mittel", "niedrig"],
            key="fine_coherence_hint",
            disabled=fine_disabled,
            help="Abstrakter Hinweis auf die gewünschte inhaltliche Kohärenz (Zusammenhang) des Textes.",
        )

# Build Params
if st.session_state["mode"] == "Ohne Alpha":
    tw = int(st.session_state["target_words"])
    target_words = None if tw == 0 else tw
else:
    target_words = None

params = Params(
    general=GeneralParams(
        topic=st.session_state["topic"],
        text_type=st.session_state["text_type"],
        target_words=target_words,
    ),
    alpha=AlphaParams(
        mode=st.session_state["mode"],
        max_sentences=int(st.session_state["alpha_max_sentences"]),
        max_words_per_sentence=int(st.session_state["alpha_max_words_per_sentence"]),
        max_syllables_per_token=int(st.session_state["alpha_max_syllables_per_token"]),
        max_dep_clauses_per_sentence=float(st.session_state["alpha_max_dep_clauses_per_sentence"] or 0.0),
        forbidden_tenses=list(st.session_state["alpha_forbidden_tenses"]),
        max_perfekt_per_finite_verb=optional_float_or_none(st.session_state.get("alpha_max_perfekt_per_finite_verb")),
        min_lexical_coverage=optional_float_or_none(st.session_state.get("alpha_min_lexical_coverage")),
    ),
    fine=FineParams(
        enabled=bool(st.session_state.get("fine_enabled", False)),
        mtul_level=clamp_level(str(st.session_state.get("fine_mtul_level", "keine Vorgabe"))),
        zipf_level=clamp_level(str(st.session_state.get("fine_zipf_level", "keine Vorgabe"))),
        lexvar_level=clamp_level(str(st.session_state.get("fine_lexvar_level", "keine Vorgabe"))),
        connectors_level=clamp_level(str(st.session_state.get("fine_connectors_level", "keine Vorgabe"))),
        forbidden_subclause_types=list(st.session_state.get("fine_forbidden_subclause_types", [])),
        konjunktiv_mode=str(st.session_state.get("fine_konjunktiv_mode", "keine Vorgabe")),
        coherence_hint=str(st.session_state.get("fine_coherence_hint", "keine")),
    ),
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Prompts")
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(params)

    with st.expander("System Prompt"):
        st.code(system_prompt, language="text")

    with st.expander("User Prompt"):
        st.code(user_prompt, language="text")

with col2:
    st.subheader("Generierter Text")

    if st.button(
        "Generate",
        type="primary",
        use_container_width=True,
        help="Startet die Textgenerierung mit den aktuellen Einstellungen.",
    ):
        try:
            out = gemini_generate(api_key, system_prompt, user_prompt, temperature)
            st.session_state["last_text"] = out
        except Exception as e:
            st.error(str(e))

    if "last_text" in st.session_state:
        st.text_area(
            "Output",
            st.session_state["last_text"],
            height=520,
            help="Ausgabe des LLMs. Sollte nur den fertigen Text enthalten.",

        )
        report = build_validation_report(params, st.session_state["last_text"])
        st.text_area(
            "Validierung / Metriken (immer aktiv)",
            report,
            height=380,
            help="Automatische Heuristik-Auswertung: Wortzahl, Alpha-Kernchecks und Zusatzmetriken. Einige Werte sind Näherungen ohne Parser.",
        )

    else:
        st.info("Klick auf Generate.")
