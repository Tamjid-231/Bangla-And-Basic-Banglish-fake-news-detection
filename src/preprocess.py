import re

from banglish_normalizer import looks_banglish, normalize_banglish


# ======================= TEXT CLEANING: START =======================
def clean_text(text: str) -> str:
    """Clean English/Bangla mixed text without removing Bangla characters."""
    text = str(text).lower()

    # Banglish is converted only when the text has enough known Roman words.
    if looks_banglish(text):
        text = normalize_banglish(text)

    # Remove links, HTML and symbols that were not used during model training.
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\u0980-\u09FFa-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
# ======================== TEXT CLEANING: END ========================
