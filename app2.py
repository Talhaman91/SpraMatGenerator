"""
Sprachlern-Framework – Streamlit POC (Single File)
- Modus: Alpha 3/4/5/6 oder Ohne Alpha
- Textlänge (Wörter) nur bei Ohne Alpha
- Alpha-Parameter sind gesperrt, wenn Alpha-Modus aktiv ist
- Ohne Alpha: Parameter frei
- User Prompt:
  - Alpha-Modus: nennt Alpha-Level + verweist auf Algorithmus als RAG-Kontext
  - Ohne Alpha: nennt keinen Alpha-Level, nur Parameter
- Keine Validierung
- Copy Buttons:
  - System+User Prompt zusammen kopierbar
  - Output kopierbar

Install:
  pip install streamlit google-genai

Run:
  streamlit run app.py
"""

from dataclasses import dataclass
import streamlit as st

# =========================
# LLM Fix-Konfiguration
# =========================
GEMINI_MODEL = "gemini-3-flash-preview"

TENSES_ALL = ["Präsens", "Präteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]

TEXT_TYPES = [
    "Erzählung",
    "Sachtext",
    "Dialog",
    "E-Mail",
    "Nachrichtenberichterstattung",
]


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


def streamlit_has_copy_to_clipboard() -> bool:
    """Kompatibilitätscheck für ältere Streamlit-Versionen."""
    return hasattr(st, "copy_to_clipboard")


# =========================
# Datenmodelle
# =========================
@dataclass
class GeneralParams:
    topic: str
    text_type: str
    target_words: int | None  # nur bei Ohne Alpha


@dataclass
class AlphaParams:
    mode: str  # "Alpha 3" | "Alpha 4" | "Alpha 5" | "Alpha 6" | "Ohne Alpha"

    max_sentences: int
    max_words_per_sentence: int
    max_syllables_per_token: int
    max_dep_clauses_per_sentence: float

    forbidden_tenses: list[str]
    max_perfekt_per_finite_verb: float | None
    min_lexical_coverage: float | None


@dataclass
class Params:
    general: GeneralParams
    alpha: AlphaParams


# =========================
# "RAG Kontext" (Algorithmus-Block als retrieved context)
# =========================
# POC: Wir implementieren keinen Retriever, aber prompten so,
# als wäre der folgende Block aus dem Paper/Algorithmus abgerufen worden.
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


# =========================
# Alpha Presets (ohne Above Alpha)
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
    st.session_state.setdefault("target_words", 140)  # nur genutzt bei Ohne Alpha

    # Parameter defaults (frei)
    st.session_state.setdefault("alpha_max_sentences", FREE_DEFAULTS["max_sentences"])
    st.session_state.setdefault("alpha_max_words_per_sentence", FREE_DEFAULTS["max_words_per_sentence"])
    st.session_state.setdefault("alpha_max_syllables_per_token", FREE_DEFAULTS["max_syllables_per_token"])
    st.session_state.setdefault("alpha_max_dep_clauses_per_sentence", FREE_DEFAULTS["max_dep_clauses_per_sentence"])
    st.session_state.setdefault("alpha_forbidden_tenses", FREE_DEFAULTS["forbidden_tenses"])

    # Optional: wir speichern hier UI-seitig 0.0 = aus
    st.session_state.setdefault("alpha_max_perfekt_per_finite_verb", 0.0)
    st.session_state.setdefault("alpha_min_lexical_coverage", 0.0)

    # Copy UI toggle
    st.session_state.setdefault("show_copy_buttons", True)


def apply_preset_if_alpha(mode: str) -> None:
    st.session_state["mode"] = mode
    if mode in ALPHA_PRESETS:
        preset = ALPHA_PRESETS[mode]
        st.session_state["alpha_max_sentences"] = preset["max_sentences"]
        st.session_state["alpha_max_words_per_sentence"] = preset["max_words_per_sentence"]
        st.session_state["alpha_max_syllables_per_token"] = preset["max_syllables_per_token"]
        st.session_state["alpha_max_dep_clauses_per_sentence"] = preset["max_dep_clauses_per_sentence"]
        st.session_state["alpha_forbidden_tenses"] = preset["forbidden_tenses"]

        # Optionalwerte: UI speichert 0.0 als "aus"
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
    return """Du bist ein Generator für deutsche Sprachlerntexte (L2).

Regeln:
- Gib ausschließlich den fertigen Text aus.
- Keine Überschrift, keine Bulletpoints, keine Erklärungen, keine Metakommentare.
- Halte dich an die Vorgaben im User Prompt so gut wie möglich.
- Wenn Vorgaben widersprüchlich sind, löse sie sinnvoll auf: Priorisiere Verständlichkeit und natürliche Sprache.
"""


def build_user_prompt(p: Params) -> str:
    g = p.general
    a = p.alpha

    forbidden = ", ".join(a.forbidden_tenses) if a.forbidden_tenses else "keine"

    # Ohne Alpha: kein Alpha im Prompt, Textlänge ist aktiv
    if a.mode == "Ohne Alpha":
        lines = [
            f'Thema: "{g.topic}".',
            f"Textart: {g.text_type}.",
            f"Textlänge: ca. {g.target_words} Wörter.",
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

    # Alpha-Modus: Alpha steht oben im Prompt + Bezug auf Algorithmus als RAG-Kontext
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
st.caption("Modus: Alpha 3–6 oder Ohne Alpha · Textlänge nur bei Ohne Alpha · keine Validierung · Copy Buttons")

ensure_defaults_exist()

# Sidebar
with st.sidebar:
    st.header("LLM")
    api_key = st.text_input("Gemini API Key", type="password")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    st.divider()

    # ===== Allgemein oben =====
    with st.expander("Allgemein", expanded=True):
        st.text_input("Thema", key="topic")
        st.selectbox(
            "Textart",
            TEXT_TYPES,
            index=TEXT_TYPES.index(st.session_state["text_type"]) if st.session_state["text_type"] in TEXT_TYPES else 1,
            key="text_type",
        )

        if st.session_state["mode"] == "Ohne Alpha":
            st.number_input("Textlänge (Wörter)", 30, 800, key="target_words", step=10)
        else:
            st.caption("Textlänge ist im Alpha-Modus deaktiviert (wird aus den Alpha-Constraints abgeleitet).")

    # ===== Alpha-Parameter darunter =====
    with st.expander("Alpha-Parameter", expanded=True):
        st.selectbox(
            "Modus",
            ["Alpha 3", "Alpha 4", "Alpha 5", "Alpha 6", "Ohne Alpha"],
            key="mode",
            on_change=on_mode_change,
        )

        mode = st.session_state["mode"]
        alpha_locked = (mode != "Ohne Alpha")

        st.number_input("Max Sätze", 1, 100, key="alpha_max_sentences", disabled=alpha_locked)
        st.number_input("Max Wörter pro Satz", 5, 60, key="alpha_max_words_per_sentence", disabled=alpha_locked)
        st.number_input("Max Silben pro Token", 1, 12, key="alpha_max_syllables_per_token", disabled=alpha_locked)

        current_dep = float(st.session_state.get("alpha_max_dep_clauses_per_sentence", 1.0) or 0.0)
        dep_max = max(10.0, current_dep)
        st.slider(
            "Max Dep-Nebensätze pro Satz",
            0.0,
            dep_max,
            key="alpha_max_dep_clauses_per_sentence",
            step=0.5,
            disabled=alpha_locked,
        )

        st.multiselect(
            "Verbotene Tempora",
            options=TENSES_ALL,
            key="alpha_forbidden_tenses",
            disabled=alpha_locked,
        )

        st.number_input(
            "Max Perfekt/finite Verben (0 = aus)",
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            key="alpha_max_perfekt_per_finite_verb",
            disabled=alpha_locked,
        )
        st.number_input(
            "Min lexikalische Abdeckung (0 = aus)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="alpha_min_lexical_coverage",
            disabled=alpha_locked,
        )

    st.divider()
    st.caption("Copy")
    st.toggle("Copy-Buttons anzeigen", value=True, key="show_copy_buttons")

# Build Params
target_words = int(st.session_state["target_words"]) if st.session_state["mode"] == "Ohne Alpha" else None

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

    combined_prompt = f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"

    if st.session_state.get("show_copy_buttons", True):
        if streamlit_has_copy_to_clipboard():
            st.copy_to_clipboard(combined_prompt)
            st.button("System + User Prompt kopieren", use_container_width=True, key="copy_prompts_btn")
            st.caption("Klick kopiert den kombinierten Prompt in die Zwischenablage.")
        else:
            st.warning("Deine Streamlit-Version unterstützt st.copy_to_clipboard() nicht. Fallback unten:")
            st.text_area("Kombi-Prompt (manuell kopieren)", combined_prompt, height=220)

with col2:
    st.subheader("Generierter Text")

    if st.button("Generate", type="primary", use_container_width=True):
        try:
            out = gemini_generate(api_key, system_prompt, user_prompt, temperature)
            st.session_state["last_text"] = out
        except Exception as e:
            st.error(str(e))

    if "last_text" in st.session_state:
        st.text_area("Output", st.session_state["last_text"], height=520)

        if st.session_state.get("show_copy_buttons", True):
            if streamlit_has_copy_to_clipboard():
                st.copy_to_clipboard(st.session_state["last_text"])
                st.button("Output kopieren", use_container_width=True, key="copy_output_btn")
            else:
                st.caption("Fallback: Output im Feld markieren und Strg+C drücken.")
    else:
        st.info("Klick auf Generate.")
