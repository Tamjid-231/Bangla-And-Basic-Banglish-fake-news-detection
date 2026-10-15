import re


# ====================== WORD DICTIONARY: START ======================
# These are common spellings seen in short Banglish news posts.
BANGLISH_WORDS = {
    "sorkar": "সরকার",
    "sarkar": "সরকার",
    "govt": "সরকার",
    "shikkha": "শিক্ষা",
    "sikkha": "শিক্ষা",
    "niti": "নীতি",
    "notun": "নতুন",
    "ghoshona": "ঘোষণা",
    "gushona": "ঘোষণা",
    "koreche": "করেছে",
    "korse": "করেছে",
    "korche": "করছে",
    "khobor": "খবর",
    "kono": "কোনো",
    "konu": "কোনো",
    "nei": "নেই",
    "nai": "নাই",
    "news": "নিউজ",
    "breaking": "ব্রেকিং",
    "obisshashyo": "অবিশ্বাস্য",
    "obishshashyo": "অবিশ্বাস্য",
    "proof": "প্রমাণ",
    "proman": "প্রমাণ",
    "chara": "ছাড়া",
    "charai": "ছাড়াই",
    "rate": "রাতে",
    "raterati": "রাতারাতি",
    "rog": "রোগ",
    "roger": "রোগের",
    "valo": "ভালো",
    "bhalo": "ভালো",
    "hobe": "হবে",
    "hoye": "হয়ে",
    "jabe": "যাবে",
    "sob": "সব",
    "share": "শেয়ার",
    "click": "ক্লিক",
    "ekhoni": "এখনই",
    "gopon": "গোপন",
    "gopone": "গোপনে",
    "voyonkor": "ভয়ংকর",
    "bhoyonkor": "ভয়ংকর",
    "decision": "সিদ্ধান্ত",
    "siddhanto": "সিদ্ধান্ত",
    "siddhanta": "সিদ্ধান্ত",
    "chikitsa": "চিকিৎসা",
    "cikitsa": "চিকিৎসা",
    "pawa": "পাওয়া",
    "paowa": "পাওয়া",
    "geche": "গেছে",
    "dhoni": "ধনী",
    "howar": "হওয়ার",
    "upay": "উপায়",
    "manush": "মানুষ",
    "bangladesh": "বাংলাদেশ",
    "bangla": "বাংলা",
    "corona": "করোনা",
    "cancer": "ক্যান্সার",
    "diabetes": "ডায়াবেটিস",
}
# ======================= WORD DICTIONARY: END =======================


# ===================== LANGUAGE CHECKS: START =======================
def has_bangla(text: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", str(text)))


def looks_banglish(text: str) -> bool:
    text = str(text).lower()
    if has_bangla(text):
        return False

    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return False

    # One matching word can be normal English, so at least two are required.
    matches = sum(1 for token in tokens if token in BANGLISH_WORDS)
    return matches >= 2
# ====================== LANGUAGE CHECKS: END ========================


# ======================= NORMALIZATION: START =======================
def normalize_banglish(text: str) -> str:
    """Map common Banglish words to Bangla script before vectorization."""
    text = str(text).lower()

    def replace_token(match: re.Match) -> str:
        token = match.group(0)
        return BANGLISH_WORDS.get(token, token)

    return re.sub(r"[a-z]+", replace_token, text)
# ======================== NORMALIZATION: END ========================
