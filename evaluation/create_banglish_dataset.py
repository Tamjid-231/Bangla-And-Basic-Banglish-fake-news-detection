"""Create a reproducible held-out Banglish evaluation dataset.

The fake/real labels are inherited from BanFakeNews-2.0. Bangla headlines are
selected only from the deterministic held-out test split and transliterated by
the rule-based mapping and lexical post-editing steps below. This is a
controlled derived benchmark, not
a naturally collected or human-labelled Banglish corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


# ===================== PROJECT SETTINGS: START ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = PROJECT_ROOT / "training"
DEFAULT_OUTPUT = Path(__file__).with_name("banglish") / "banglish_controlled_250.csv"
DEFAULT_REPORT = Path(__file__).with_name("banglish") / "dataset_report.json"
RANDOM_STATE = 42
# ====================== PROJECT SETTINGS: END =======================

# ===================== LOCAL MODULES: START =========================
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TRAINING_DIR))
from banglish_normalizer import BANGLISH_WORDS  # noqa: E402
from train_model import read_dataset, split_samples  # noqa: E402
# ====================== LOCAL MODULES: END ===========================


# ================= TRANSLITERATION MAPS: START ======================
INDEPENDENT_VOWELS = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
}

CONSONANTS = {
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "chh",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "n",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "f",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "rh",
    "য়": "y",
}

VOWEL_SIGNS = {
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
}

OTHER_SIGNS = {"ং": "ng", "ঃ": "h", "ঁ": "n"}
VIRAMA = "্"
BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
# ================== TRANSLITERATION MAPS: END =======================


# ============== DATASET CONSTRUCTION HELPERS: START =================
def preferred_dictionary_spellings() -> dict[str, str]:
    """Reverse the app dictionary using its first spelling as the canonical form."""
    spellings: dict[str, str] = {}
    for roman, bangla in BANGLISH_WORDS.items():
        spellings.setdefault(bangla, roman)
    return spellings


PREFERRED_WORDS = preferred_dictionary_spellings()


def transliterate_word(word: str) -> str:
    if word in PREFERRED_WORDS:
        return PREFERRED_WORDS[word]

    output: list[str] = []
    index = 0
    while index < len(word):
        char = word[index]
        if char in INDEPENDENT_VOWELS:
            output.append(INDEPENDENT_VOWELS[char])
        elif char in CONSONANTS:
            # য is normally written as y inside a conjunct (for example, ত্য).
            consonant = "y" if char == "য" and index > 0 and word[index - 1] == VIRAMA else CONSONANTS[char]
            output.append(consonant)
            next_char = word[index + 1] if index + 1 < len(word) else ""
            if next_char in VOWEL_SIGNS:
                output.append(VOWEL_SIGNS[next_char])
                index += 1
            elif next_char == VIRAMA:
                index += 1
            else:
                output.append("o")
        elif char in OTHER_SIGNS:
            output.append(OTHER_SIGNS[char])
        elif char in VOWEL_SIGNS:
            output.append(VOWEL_SIGNS[char])
        elif char in "০১২৩৪৫৬৭৮৯":
            output.append(char.translate(BANGLA_DIGITS))
        elif char.isascii() and char.isalnum():
            output.append(char.lower())
        else:
            output.append(" ")
        index += 1
    roman = "".join(output)
    # Final inherent অ is usually not pronounced in informal Banglish writing.
    if len(roman) > 3 and roman.endswith("o"):
        roman = roman[:-1]
    return roman


def transliterate_text(text: str) -> str:
    parts = re.findall(r"[\u0980-\u09FF]+|[A-Za-z0-9]+", str(text))
    transliterated = [transliterate_word(part) for part in parts]
    return re.sub(r"\s+", " ", " ".join(transliterated)).strip()


# Raw character conversion can produce spellings that a Banglish writer would
# almost never type. These lexical revisions keep the process
# reproducible while making common verbs, names, conjuncts, and y-phala forms
# read like everyday Roman Bangla.
NATURAL_PHRASE_REVISIONS = {
    "egijo e gijo e": "egiye giye",
    "dhorijo e dijo echhe": "dhoriye diyeche",
    "chhodo ijo e podo ar": "choriye porar",
    "chhodo ijo e": "choriye",
    "pejo echhe": "peyeche",
    "hojo echhe": "hoyeche",
    "jojo debopur": "joydebpur",
    "brajo an uilijo amos": "brian williams",
    "sirijo ar": "syria-r",
    "elijo enoder": "alien-der",
    "elijo enogul": "alien-gulo",
    "elijo en": "alien",
    "hoyatosoo yaper": "whatsapp-er",
    "hoyatosoo yape": "whatsapp-e",
    "hoyatosoo yap": "whatsapp",
    "bado ir": "barir",
    "podo a": "pora",
    "u padon": "utpadon",
    "u sah": "utshaho",
    "o yambulens": "ambulance",
    "o yadominoder": "admin-der",
    "bilijo n": "billion",
    "nojo a": "noya",
    "dejo al": "deyal",
    "jojo ar": "joar",
    "ajo kor": "aykor",
    "fajo korer": "faykor-er",
    "dajo e rajo": "daye ray",
    "seks o yakter": "sex act-er",
    "haojo aijo er": "hawai-er",
    "miloojo aki": "milwaukee",
    "niye e": "niye",
    "lukiye e": "lukiye",
    "diye e": "diye",
    "poro e": "pore",
    "bojo frend": "boyfriend",
    "bhado a": "bhara",
    "neye ar": "newar",
    "o yap": "app",
    "o yaleks": "alex",
    "o yanti": "anti",
    "o yakaunter": "account-er",
    "o yatakos": "attacks",
    "sobochejo e": "shobcheye",
    "nojo akhalir": "noakhalir",
    "jamajo at": "jamayat",
    "jijo a": "zia",
    "khaojo ar": "khawar",
    "shehonajo ke": "shehnaz-ke",
    "meye eder": "meyeder",
    "kaojo ardos": "cowards",
    "projo ojon": "proyojon",
    "unnojo n": "unnoyon",
    "obajo dul": "obaydul",
    "elijabeth ojo aren": "elizabeth warren",
    "fes buk": "facebook",
}

NATURAL_WORD_REVISIONS = {
    "achchhe": "ache",
    "achhe": "ache",
    "hochchhe": "hocche",
    "hochchhen": "hocchen",
    "hoyechhe": "hoyeche",
    "korechhilen": "korechilen",
    "korechhen": "korechen",
    "gechhen": "gechen",
    "pathiyechhe": "pathiyeche",
    "bolo chhi": "bolchi",
    "bolochhi": "bolchi",
    "chhilo": "chilo",
    "chhil": "chilo",
    "korote": "korte",
    "koroben": "korben",
    "korobe": "korbe",
    "korol": "korlo",
    "porobe": "porbe",
    "parob": "parbo",
    "asote": "aste",
    "hasote": "hashte",
    "janote": "jante",
    "bolenoni": "bolenni",
    "bololen": "bollen",
    "milol": "millo",
    "harol": "harlo",
    "tanol": "tanlo",
    "porol": "porlo",
    "pathal": "pathalo",
    "ekoti": "ekti",
    "ekota": "ekta",
    "konoti": "konti",
    "choloti": "cholti",
    "somprochare": "shomprochare",
    "somporke": "shomporke",
    "somporker": "shomporker",
    "songbad": "shongbad",
    "songbader": "shongbader",
    "songkromon": "shongkromon",
    "songlape": "shonglape",
    "songghorshe": "shongghorshe",
    "songge": "shonge",
    "songka": "shonka",
    "shongka": "shonka",
    "sbasthyokor": "shasthokor",
    "sbasthyomontri": "shasthomontri",
    "sbabhabiker": "shabhabiker",
    "sbakshor": "shakkhor",
    "sborashtromontri": "shorastromontri",
    "sborn": "shorno",
    "osbikar": "oshikar",
    "osbosti": "oshosti",
    "dbitijo": "ditiyo",
    "dbara": "dara",
    "dbar": "dar",
    "dbigun": "digun",
    "protidbondbitajo": "protidwondwitay",
    "jatijo": "jatiyo",
    "bharotijo": "bharotiyo",
    "kshomotajo": "khomotay",
    "ghotonajo": "ghotonay",
    "porikshajo": "porikkhay",
    "protikrijo": "protikriya",
    "poryajo": "porjay",
    "rastajo": "rastay",
    "tulajo": "tolay",
    "namijo": "namiye",
    "lukijo": "lukiye",
    "shorijo": "shoriye",
    "jalijo": "jaliye",
    "egijo": "egiye",
    "dijo": "diye",
    "pejo": "peye",
    "hojo": "hoy",
    "nijo": "niye",
    "jajo": "jay",
    "khejo": "kheye",
    "khajo": "khay",
    "chajo": "chay",
    "mejo": "meye",
    "nejo": "neye",
    "ajo": "ay",
    "jojo": "joy",
    "bhujo": "bhuyo",
    "bhu y": "bhuyo",
    "bhuy": "bhuyo",
    "chhodo": "chora",
    "podo": "poro",
    "bado": "baro",
    "rijobhi": "rizvi",
    "fokhorul": "fakhrul",
    "imoran": "imran",
    "mashorafi": "mashrafi",
    "mushofik": "mushfiq",
    "dijital": "digital",
    "dijitaler": "digitaler",
    "bhidi": "video",
    "bhidioti": "videoti",
    "bhiu": "view",
    "iutiub": "youtube",
    "chyanel": "channel",
    "chyaneler": "channel-er",
    "kyampeiner": "campaign-er",
    "ofisijo": "official",
    "motijo": "motiyo",
    "midijo": "media",
    "niuijo": "newyork",
    "sotyo": "shotto",
    "soty": "shotto",
    "jony": "jonno",
    "priy": "priyo",
    "jibone": "jibone",
    "shbeti": "sheti",
    "sposht": "sposto",
    "unnoyonoshil": "unnoyonshil",
    "septembor": "september",
    "disemborer": "december-er",
    "oktobor": "october",
    "somojo": "shomoy",
    "osohajo": "oshohay",
    "boktritajo": "boktritay",
    "papijo": "papiya",
    "hridojo": "hridoy",
    "bodo": "boro",
    "bolo chhen": "bolchen",
    "bolochhen": "bolchen",
    "dekhote": "dekhte",
    "gunote": "gunte",
    "pabolik": "public",
    "madhyomogul": "madhyomgulo",
    "byoktii": "bektiti",
    "koron": "karan",
    "jeahorer": "johor-er",
    "fesobuker": "facebook-er",
    "fichar": "feature",
    "rajo": "ray",
    "dajo": "daye",
}


def naturalize_banglish(text: str) -> str:
    """Polish deterministic output into readable, informal Roman Bangla."""
    padded = f" {text.lower()} "
    for source, replacement in sorted(
        NATURAL_PHRASE_REVISIONS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        padded = padded.replace(f" {source} ", f" {replacement} ")

    words = padded.split()
    polished = [NATURAL_WORD_REVISIONS.get(word, word) for word in words]
    return re.sub(r"\s+", " ", " ".join(polished)).strip()


def stable_sample_id(label: str, headline: str) -> str:
    digest = hashlib.sha256(f"{label}\n{headline}".encode("utf-8")).hexdigest()[:12]
    return f"BNG-{digest}"


def choose_balanced_test_samples(
    train: list[dict],
    validation: list[dict],
    test: list[dict],
    total: int,
) -> list[dict]:
    if total < 2 or total % 2:
        raise ValueError("The dataset size must be an even integer of at least 2.")

    per_class = total // 2
    # The model is trained on both article text and headline examples. Exclude
    # every test headline whose exact cleaned headline occurs in train or
    # validation, even when the full article text is different.
    excluded_headlines = {
        sample["headline"] for sample in [*train, *validation]
    }
    by_label = {
        label: [
            sample
            for sample in test
            if sample["label"] == label
            and len(sample["headline"]) >= 20
            and re.search(r"[\u0980-\u09FF]", sample["headline"])
            and sample["headline"] not in excluded_headlines
        ]
        for label in ("fake", "real")
    }
    rng = random.Random(RANDOM_STATE)
    chosen: list[dict] = []
    for label, candidates in by_label.items():
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(f"Not enough held-out {label} headlines for {per_class} cases.")
        chosen.extend(candidates[:per_class])
    rng.shuffle(chosen)
    return chosen


def create_rows(samples: list[dict]) -> list[dict]:
    rows = []
    seen_banglish: set[str] = set()
    for sample in samples:
        banglish = naturalize_banglish(transliterate_text(sample["headline"]))
        if not banglish or re.search(r"[\u0980-\u09FF]", banglish):
            raise ValueError("Transliteration left Bangla characters in a benchmark row.")
        if banglish in seen_banglish:
            raise ValueError("Duplicate Banglish text was generated.")
        seen_banglish.add(banglish)
        rows.append(
            {
                "sample_id": stable_sample_id(sample["label"], sample["headline"]),
                "source_headline_bangla": sample["headline"],
                "banglish_text": banglish,
                "label": sample["label"],
                "label_source": "inherited_from_banfakenews_2_0",
            }
        )
    return rows


def write_dataset(rows: list[dict], output: Path, report_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["label"] for row in rows)
    report = {
        "dataset_name": "Controlled Held-out Banglish Evaluation Set",
        "total_cases": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "source_dataset": "BanFakeNews-2.0",
        "source_split": "deterministic held-out 15% test split",
        "leakage_control": (
            "source headlines are excluded when the same cleaned headline "
            "appears in training or validation"
        ),
        "train_headline_overlap": 0,
        "validation_headline_overlap": 0,
        "random_state": RANDOM_STATE,
        "label_method": "labels inherited from source headlines",
        "transliteration_method": (
            "deterministic Bangla-to-Roman transliteration with rule-based "
            "mapping and lexical post-editing"
        ),
        "human_validation": False,
        "scope_note": (
            "This is a controlled derived benchmark. It is not a naturally collected "
            "Banglish corpus and must not be described as human-labelled Banglish data."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============== DATASET CONSTRUCTION HELPERS: END ==================


# ===================== DATASET RUNNER: START ========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Create the held-out Banglish benchmark.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "All60Kdataset.csv",
    )
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    samples, _ = read_dataset(args.dataset.resolve())
    train, validation, test = split_samples(samples)
    selected = choose_balanced_test_samples(train, validation, test, args.size)
    rows = create_rows(selected)
    write_dataset(rows, args.output.resolve(), args.report.resolve())
    print(f"Created {len(rows)} held-out Banglish cases: {args.output.resolve()}")
    print(f"Dataset report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
# ====================== DATASET RUNNER: END =========================
