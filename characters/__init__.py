"""Character cards for AI companions."""

from .elara import ELARA
from .sofia import SOFIA
from .kai import KAI
from .marcus import MARCUS

CHARACTERS = {
    "elara": ELARA,
    "sofia": SOFIA,
    "kai": KAI,
    "marcus": MARCUS,
}

def get_character(name: str) -> dict:
    key = name.lower().strip()
    if key not in CHARACTERS:
        raise ValueError(f"Unknown character: {name}. Choose from {list(CHARACTERS)}")
    return CHARACTERS[key]

def list_characters() -> list[str]:
    return list(CHARACTERS.keys())
