from src.sprachlern_tool.config import LEVELS_4
"""
Kleine Hilfsfunktionen für robuste UI- und Parameterverarbeitung.

- Normalisierung von Eingaben (z.B. 0 → deaktiviert)
- Absicherung von Level-Strings gegen ungültige Werte
"""


def optional_float_or_none(value) -> float | None:
    """
    Normalisiert numerische UI-Eingaben zu optionalen Grenzwerten.

    Konvention:
    - None oder nicht parsebar → None
    - 0.0 → None (Regel gilt als deaktiviert)
    - sonst → float(value)

    Zweck:
    Viele Constraints sind optional. Streamlit liefert aber oft Zahlen,
    daher wird hier eine robuste Konvertierung zentral gelöst.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v == 0.0 else v


def clamp_level(value: str) -> str:
    """
    Schützt vor ungültigen Stufenwerten, z. B. bei Session-State Migrations.
    """
    return value if value in LEVELS_4 else "keine Vorgabe"
