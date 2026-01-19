def context_for_alpha(mode: str) -> str:
    """
    Kontextblock für Alpha-Level.

    Dieser Block wird im User Prompt als „Retrieved Context“ dargestellt,
    um die Regeln kompakt und wiederholbar einzubinden.
    """
    if mode == "Alpha 3":
        return (
            "Context (Alpha Readability Level Algorithmus, Regeln):\n"
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
            "Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 10\n"
            "- nSentences <= 10\n"
            "- syllablesPerToken <= 5\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
            "- future1sPerFiniteVerb == 0\n"
            "- future2sPerFiniteVerb == 0\n"
        )
    if mode == "Alpha 5":
        return (
            "Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 12\n"
            "- nSentences <= 15\n"
            "- pastPerfectsPerFiniteVerb == 0\n"
        )
    if mode == "Alpha 6":
        return (
            "Context (Alpha Readability Level Algorithmus, Regeln):\n"
            "- wordsPerSentence <= 12\n"
            "- nSentences <= 20\n"
        )
    return ""

