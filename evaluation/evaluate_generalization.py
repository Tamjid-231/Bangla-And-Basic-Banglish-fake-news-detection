"""Run a basic controlled distribution-shift generalization check (RQ5)."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score


# ===================== PROJECT SETTINGS: START ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = PROJECT_ROOT / "training"
CASES_PATH = Path(__file__).with_name("evidence_logic_cases.csv")
MODEL_PATH = PROJECT_ROOT / "models" / "banfakenews_tfidf_svm.joblib"
OUTPUT_CSV = Path(__file__).with_name("generalization_results.csv")
OUTPUT_JSON = Path(__file__).with_name("generalization_report.json")
OUTPUT_MD = PROJECT_ROOT / "Documentation" / "evaluation" / "GENERALIZATION_CHECK.md"
THRESHOLD = 0.50
# ====================== PROJECT SETTINGS: END =======================

# ===================== LOCAL MODULES: START =========================
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TRAINING_DIR))
from preprocess import clean_text  # noqa: E402
from train_model import read_dataset, split_samples  # noqa: E402
# ====================== LOCAL MODULES: END ===========================


# ==================== STATUS MAPPING: START =========================
STATUS_LABEL = {
    "fact_checked_false": "fake",
    "fact_checked_true": "real",
    "supported_by_sources": "real",
}
# ===================== STATUS MAPPING: END ==========================


# ================= METRIC CALCULATION: START ========================
def metric_summary(labels: list[str], predictions: list[str]) -> dict:
    report = classification_report(
        labels,
        predictions,
        labels=["fake", "real"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "fake_f1": float(report["fake"]["f1-score"]),
        "real_f1": float(report["real"]["f1-score"]),
    }
# ================== METRIC CALCULATION: END =========================


# ================== EVALUATION RUNNER: START ========================
def main() -> None:
    with CASES_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        all_cases = list(csv.DictReader(csv_file))
    cases = [case for case in all_cases if case["expected_status"] in STATUS_LABEL]

    samples, _ = read_dataset(PROJECT_ROOT / "data" / "All60Kdataset.csv")
    train, validation, _ = split_samples(samples)
    development_texts = {
        value
        for sample in [*train, *validation]
        for value in (sample["text"], sample["headline"])
    }

    model = joblib.load(MODEL_PATH)
    results = []
    for case in cases:
        cleaned = clean_text(case["claim"])
        if cleaned in development_texts:
            raise ValueError(
                f"Controlled generalization claim overlaps development data: {case['id']}"
            )
        score = float(model.decision_function([cleaned])[0])
        prediction = "real" if score >= THRESHOLD else "fake"
        expected = STATUS_LABEL[case["expected_status"]]
        results.append(
            {
                "id": case["id"],
                "language": case.get("language", ""),
                "claim": case["claim"],
                "expected_label": expected,
                "prediction": prediction,
                "decision_score": f"{score:.6f}",
                "correct": prediction == expected,
                "development_overlap": False,
            }
        )

    labels = [row["expected_label"] for row in results]
    predictions = [row["prediction"] for row in results]
    shifted_metrics = metric_summary(labels, predictions)
    by_language: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row["language"]].append(row)
    for language, language_rows in sorted(grouped.items()):
        language_labels = [row["expected_label"] for row in language_rows]
        language_predictions = [row["prediction"] for row in language_rows]
        by_language[language] = {
            "cases": len(language_rows),
            **metric_summary(language_labels, language_predictions),
        }

    metadata = getattr(model, "capstone_metadata_", {})
    in_sample = metadata["comparison_results"]["Hybrid Linear SVM"]["test"]
    gap = float(in_sample["accuracy"]) - shifted_metrics["accuracy"]
    report = {
        "evaluation_type": "controlled differently-distributed short-claim check",
        "in_sample_test": {
            "cases": sum(sum(row) for row in in_sample["confusion_matrix"]),
            "accuracy": float(in_sample["accuracy"]),
            "macro_f1": float(in_sample["macro_f1"]),
        },
        "distribution_shift_test": {
            "cases": len(results),
            **shifted_metrics,
            "by_language": by_language,
            "development_text_overlap": 0,
        },
        "accuracy_drop": gap,
        "scope_note": (
            "This is a small controlled short-claim distribution shift, not a "
            "publisher-based, time-based, or independently collected external dataset."
        ),
    }

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Controlled Generalization Check (RQ5)",
        "",
        "## Result",
        "",
        f"- In-sample held-out article accuracy: {in_sample['accuracy'] * 100:.2f}%",
        f"- Controlled short-claim accuracy: {shifted_metrics['accuracy'] * 100:.2f}%",
        f"- Accuracy drop: {gap * 100:.2f} percentage points",
        f"- Controlled cases: {len(results)}",
        "- Exact overlap with train/validation text: 0",
        "",
        "## Interpretation",
        "",
        "The large drop shows that strong random-split article performance does not",
        "transfer reliably to short, differently written claims. This supports cautious",
        "website wording and separate evidence display.",
        "",
        "## Scope",
        "",
        report["scope_note"],
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"RQ5 accuracy: {shifted_metrics['accuracy'] * 100:.2f}% "
        f"(drop {gap * 100:.2f} points from in-sample test)"
    )


if __name__ == "__main__":
    main()
# =================== EVALUATION RUNNER: END =========================
