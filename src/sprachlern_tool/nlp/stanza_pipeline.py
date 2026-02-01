"""
Stanza-Pipeline Initialisierung.

Wird über Streamlit cache_resource gecached, damit das Modell nicht bei jedem Run neu lädt.
Enthält eine klare Fehlermeldung inkl. Hinweis auf stanza.download("de").
"""

import streamlit as st
import stanza


@st.cache_resource
def get_stanza_nlp():
    """
    Erstellt eine Stanza-Pipeline für Deutsch.

    Konfiguration:
    - tokenize: Segmentierung
    - pos: Part-of-Speech
    - lemma: Lemmatisierung
    - depparse: Dependency-Parsing (UD)

    Zweck:
    Die Analysemetriken im Tool basieren auf UD-Annotationen und Token/Satz-Infos.
    """
    try:
        return stanza.Pipeline(
            "de",
            processors="tokenize,pos,lemma,depparse",
            tokenize_no_ssplit=False,
            use_gpu=False,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            "Stanza Pipeline konnte nicht gestartet werden. "
            "Hast du vorher 'python -c \"import stanza; stanza.download(\\\"de\\\")\"' ausgeführt?\n"
            f"Originalfehler: {e}"
        )
