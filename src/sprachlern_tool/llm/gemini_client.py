"""
Gemini-Client Wrapper.

Kapselt:
- API-Key Handling
- Model-Auswahl über zentrale Config
- Generierungs-Call inkl. Temperature
- optionales Anhängen eines lokalen PDF-Dokuments als Kontext

Ziel: Die Streamlit-App soll Gemini nur über eine Funktion ansprechen,
damit Provider-Wechsel oder API-Anpassungen lokal bleiben.
"""

import os
import sys

from src.sprachlern_tool.config import GEMINI_MODEL


# PDF wird immer implizit genutzt (kein Upload, keine Checkbox)
DEFAULT_PDF_PATH = os.path.join("resources", "pdfs", "rag.pdf")


def gemini_generate(api_key: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """
    Führt einen einzelnen Generierungsaufruf gegen Gemini aus.

    Parameter:
        api_key: Gemini API Key aus dem UI.
        system_prompt: Rollen-/Formatvorgaben (sollte stabil bleiben).
        user_prompt: konkrete Aufgabenbeschreibung inkl. Parameterkonfiguration.
        temperature: Kreativitätsparameter (0 = deterministischer, >0 variabler).

    Verhalten:
    - Wenn das PDF unter DEFAULT_PDF_PATH existiert, wird es hochgeladen und als Kontext mitgesendet.
    - Der Rückgabewert ist der reine Text (strip), oder es wird ein Fehler geworfen, falls leer.

    Raises:
        RuntimeError: wenn kein API-Key gesetzt ist oder Gemini keinen Text liefert.
    """

    if not api_key.strip():
        raise RuntimeError("Kein Gemini API Key gesetzt.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=float(temperature),
    )

    if os.path.isfile(DEFAULT_PDF_PATH):
        # Pfad zu einem lokalen PDF, das als Kontextmaterial an Gemini angehängt wird
        print(f"PDF gefunden: {DEFAULT_PDF_PATH}", file=sys.stderr)

        pdf_file = client.files.upload(file=DEFAULT_PDF_PATH)
        contents = [pdf_file, user_prompt]

        print("PDF wurde an Gemini angehängt", file=sys.stderr)
    else:
        contents = user_prompt
        print("Keine PDF gefunden –> Generierung ohne PDF", file=sys.stderr)

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=cfg,
    )

    text = getattr(resp, "text", None)
    if not text:
        raise RuntimeError("Gemini hat keinen Text geliefert.")

    return text.strip()
