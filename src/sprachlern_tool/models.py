"""
Datenmodelle für die Parameterkonfiguration.

Die App hält die Einstellungen bewusst als Dataclasses, damit:
- UI-State: Parameterobjekt klar typisiert ist,
- Prompting/Validierung nur mit strukturierten Objekten arbeitet,
- spätere Erweiterungen (mehr Parameter) sauber möglich bleiben.
"""

from dataclasses import dataclass


@dataclass
class GeneralParams:
    """
    Allgemeine Parameter.

    Attributes:
        topic: Thema/Inhaltsschwerpunkt des Textes.
        text_type: Textsorte (z.B. Erzählung, Sachtext, Dialog).
        target_words: Ziel-Wortanzahl (nur relevant im Ohne-Alpha-Modus).
    """
    topic: str
    text_type: str
    target_words: int | None


@dataclass
class AlphaParams:
    """
    Alpha-Parameter (harte oder semi-harte Constraints).

    Diese Parameter repräsentieren das regelbasierte Alpha-Level-Regelwerk
    (z.B. maximale Satzanzahl, Silben pro Token, Tempus-Verbote).
    Je nach Mode (Alpha 3–6 vs. Ohne Alpha) werden sie unterschiedlich streng geprüft.
    """
    mode: str

    max_sentences: int
    max_words_per_sentence: int
    max_syllables_per_token: int
    max_dep_clauses_per_sentence: float

    forbidden_tenses: list[str]
    max_perfekt_per_finite_verb: float | None
    min_lexical_coverage: float | None


@dataclass
class FineParams:
    """
    Zusatzparameter zur Feinsteuerung innerhalb eines Alpha-Levels.

    Diese Werte sind als "weiche Ziele" gedacht, die die Textcharakteristik beeinflussen,
    ohne das Alpha-Level-Regelwerk zu ersetzen.

    Beispiele:
    - MTUL-Level: syntaktische Komplexität über T-Units
    - Zipf-Level: Wortfrequenz (Alltagsnähe vs. seltener Wortschatz)
    - LexVar-Level: lexikalische Vielfalt (TTR/MTLD)
    - Konnektoren-Level: Häufigkeit von Konnektoren
    - Tempus-Gewichtungen: gewünschte Verteilung/Vermeidung bestimmter Tempora
    - Nebensatz-Typen/Modi: verbieten/entmutigen bestimmter Strukturen (heuristisch)
    - Kohärenz-Hinweis: High-Level Ziel für thematische/entitätenbasierte Stringenz
    """
    enabled: bool
    mtul_level: str
    zipf_level: str
    lexvar_level: str
    connectors_level: str
    tense_weights: dict[str, str]
    forbidden_subclause_types: list[str]
    konjunktiv_mode: str
    coherence_hint: str


@dataclass
class Params:
    """
    Container für die vollständige Parameterkonfiguration.

    Bündelt:
    - general: Inhalt/Textsorte/Zielumfang
    - alpha: Alpha-Regelwerk-Parameter (Constraints)
    - fine: Zusatzparameter (weiche Ziele)
    """
    general: GeneralParams
    alpha: AlphaParams
    fine: FineParams
