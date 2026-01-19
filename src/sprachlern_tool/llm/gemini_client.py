import os
import sys

from src.sprachlern_tool.config import GEMINI_MODEL


# PDF wird immer implizit genutzt (kein Upload, keine Checkbox)
DEFAULT_PDF_PATH = os.path.join("resources", "pdfs", "rag.pdf")


def gemini_generate(api_key: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """
    Führt einen einzelnen Generierungs-Call gegen Gemini aus.
    Die PDF wird automatisch mitgesendet, falls vorhanden.
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
