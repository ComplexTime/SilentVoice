from backend.llm_groq import (
    translate_with_groq,
    LANGUAGES
)


def translate_signs(
    signs: list[str],
    language: str = "en"
) -> str:

    if not signs:
        raise ValueError("No signs provided")

    if any(not sign.strip() for sign in signs):
        raise ValueError("Empty sign detected")

    if any(
        sign.strip().upper() == "UNKNOWN"
        for sign in signs
    ):
        raise ValueError("Remove unknown signs")

    if language not in LANGUAGES:
        raise ValueError("Unsupported language")

    return translate_with_groq(
        signs,
        language
    )