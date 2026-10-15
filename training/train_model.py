"""Retrain and compare the capstone fake-news models.

The default output is a candidate model, so the website model is not replaced
accidentally. Use --replace-model only after checking the new evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


# ===================== PROJECT SETTINGS: START ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "All60Kdataset.csv"
ACTIVE_MODEL = PROJECT_ROOT / "models" / "banfakenews_tfidf_svm.joblib"
REPORT_DIR = PROJECT_ROOT / "evaluation" / "retraining"
RANDOM_STATE = 42
WEBSITE_THRESHOLD = 0.50
# ====================== PROJECT SETTINGS: END =======================

# ===================== LOCAL MODULES: START =========================
sys.path.insert(0, str(SRC_DIR))
from preprocess import clean_text  # noqa: E402
# ====================== LOCAL MODULES: END ===========================


# ================= DATA AND MODEL HELPERS: START ====================
def read_dataset(path: Path) -> tuple[list[dict], dict]:
    """Read, clean, and deduplicate the three required CSV columns."""
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    samples: list[dict] = []
    seen_text: set[str] = set()
    source_rows = 0
    duplicates_removed = 0

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Headline", "Content", "Label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

        for row in reader:
            source_rows += 1
            label_value = str(row.get("Label", "")).strip()
            if label_value not in {"0", "1"}:
                continue

            headline = str(row.get("Headline") or "").strip()
            content = str(row.get("Content") or "").strip()
            text = clean_text(f"{headline} {content[:3000]}")
            if len(text) < 20:
                continue
            if text in seen_text:
                duplicates_removed += 1
                continue

            seen_text.add(text)
            headline_text = clean_text(headline)
            if len(headline_text) < 10:
                headline_text = text
            samples.append(
                {
                    "text": text,
                    "headline": headline_text,
                    "label": "fake" if label_value == "0" else "real",
                }
            )

    statistics = {
        "source_rows": source_rows,
        "usable_rows": len(samples),
        "duplicates_removed": duplicates_removed,
        "fake_rows": sum(item["label"] == "fake" for item in samples),
        "real_rows": sum(item["label"] == "real" for item in samples),
    }
    return samples, statistics


def select_stratified_sample(samples: list[dict], max_rows: int | None) -> list[dict]:
    """Create a smaller reproducible subset for a quick code check."""
    if not max_rows or max_rows >= len(samples):
        return samples
    labels = [item["label"] for item in samples]
    selected, _ = train_test_split(
        samples,
        train_size=max_rows,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return list(selected)


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [item["label"] for item in samples]
    train, test = train_test_split(
        samples,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_labels = [item["label"] for item in train]
    train, validation = train_test_split(
        train,
        test_size=0.1764705882,
        random_state=RANDOM_STATE,
        stratify=train_labels,
    )
    return list(train), list(validation), list(test)


def metric_summary(labels: list[str], predictions: np.ndarray) -> dict:
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
        "fake_precision": float(report["fake"]["precision"]),
        "fake_recall": float(report["fake"]["recall"]),
        "fake_f1": float(report["fake"]["f1-score"]),
        "real_f1": float(report["real"]["f1-score"]),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=["fake", "real"],
        ).tolist(),
    }


def train_baselines(
    train: list[dict],
    validation: list[dict],
    test: list[dict],
) -> dict:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_features = vectorizer.fit_transform(item["text"] for item in train)
    validation_features = vectorizer.transform(item["text"] for item in validation)
    test_features = vectorizer.transform(item["text"] for item in test)
    train_labels = [item["label"] for item in train]
    validation_labels = [item["label"] for item in validation]
    test_labels = [item["label"] for item in test]

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=300,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
    }
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_features, train_labels)
        results[name] = {
            "validation": metric_summary(
                validation_labels,
                model.predict(validation_features),
            ),
            "test": metric_summary(test_labels, model.predict(test_features)),
        }
    return results


def choose_threshold(labels: list[str], scores: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.02
    best_macro_f1 = -1.0
    for threshold in np.linspace(0.02, 0.50, 97):
        predictions = np.where(scores >= threshold, "real", "fake")
        score = f1_score(labels, predictions, average="macro")
        if score > best_macro_f1:
            best_threshold = float(threshold)
            best_macro_f1 = float(score)
    return best_threshold, best_macro_f1


def make_svm_pipeline(feature_kind: str) -> Pipeline:
    word_features = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    character_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        min_df=4,
        max_features=150_000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    if feature_kind == "word":
        features = word_features
    elif feature_kind == "character":
        features = character_features
    elif feature_kind == "hybrid":
        features = FeatureUnion(
            [("word", word_features), ("character", character_features)],
            transformer_weights={"word": 1.0, "character": 0.7},
        )
    else:
        raise ValueError(f"Unknown SVM feature kind: {feature_kind}")

    classifier = LinearSVC(
        C=1.0,
        class_weight="balanced",
        fit_intercept=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", features), ("classifier", classifier)])


def train_svm_variants(
    train: list[dict],
    validation: list[dict],
    test: list[dict],
) -> tuple[Pipeline, dict]:
    """Compare word-only, character-only, and hybrid Linear SVM models."""
    # Headline examples are repeated so the website handles short input better.
    training_text = [item["text"] for item in train]
    training_text.extend(item["headline"] for item in train)
    training_labels = [item["label"] for item in train] * 2
    validation_text = [item["text"] for item in validation]
    validation_labels = [item["label"] for item in validation]
    test_text = [item["text"] for item in test]
    test_labels = [item["label"] for item in test]
    headline_text = [item["headline"] for item in test]

    variants = {
        "Word-only Linear SVM": "word",
        "Character-only Linear SVM": "character",
        "Hybrid Linear SVM": "hybrid",
    }
    results: dict[str, dict] = {}
    deployment_model: Pipeline | None = None

    for model_name, feature_kind in variants.items():
        print(f"Training {model_name}...")
        model = make_svm_pipeline(feature_kind)
        model.fit(training_text, training_labels)

        validation_scores = model.decision_function(validation_text)
        threshold, validation_macro_f1 = choose_threshold(
            validation_labels,
            validation_scores,
        )
        test_predictions = np.where(
            model.decision_function(test_text) >= threshold,
            "real",
            "fake",
        )
        headline_predictions = np.where(
            model.decision_function(headline_text) >= threshold,
            "real",
            "fake",
        )
        results[model_name] = {
            "feature_kind": feature_kind,
            "selected_threshold_before_alignment": threshold,
            "validation_macro_f1_at_threshold": validation_macro_f1,
            "validation": metric_summary(
                validation_labels,
                np.where(validation_scores >= threshold, "real", "fake"),
            ),
            "test": metric_summary(test_labels, test_predictions),
            "headline_test": metric_summary(test_labels, headline_predictions),
        }

        if feature_kind == "hybrid":
            # Shift the fitted score so the website can consistently use 0.50.
            model.named_steps["classifier"].intercept_ += WEBSITE_THRESHOLD - threshold
            deployment_model = model

    if deployment_model is None:
        raise RuntimeError("The Hybrid Linear SVM was not trained.")
    return deployment_model, results


def write_reports(metadata: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "model_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (report_dir / "model_comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "validation_macro_f1",
                "test_accuracy",
                "test_macro_f1",
                "test_fake_f1",
                "headline_accuracy",
                "headline_macro_f1",
                "headline_fake_f1",
            ]
        )
        for name, result in metadata["comparison_results"].items():
            test = result["test"]
            headline = result.get("headline_test", {})
            writer.writerow(
                [
                    name,
                    f"{result['validation']['macro_f1']:.6f}",
                    f"{test['accuracy']:.6f}",
                    f"{test['macro_f1']:.6f}",
                    f"{test['fake_f1']:.6f}",
                    f"{headline['accuracy']:.6f}" if headline else "",
                    f"{headline['macro_f1']:.6f}" if headline else "",
                    f"{headline['fake_f1']:.6f}" if headline else "",
                ]
            )

    matrix = metadata["comparison_results"]["Hybrid Linear SVM"]["test"][
        "confusion_matrix"
    ]
    with (report_dir / "confusion_matrix.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual/predicted", "fake", "real"])
        writer.writerow(["fake", *matrix[0]])
        writer.writerow(["real", *matrix[1]])


# ================== DATA AND MODEL HELPERS: END =====================


# ===================== TRAINING RUNNER: START =======================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain and compare the capstone fake-news models."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Use a stratified subset for a quick test; omit for the full experiment.",
    )
    parser.add_argument(
        "--replace-model",
        action="store_true",
        help="Replace the active website model after training.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("Reading and cleaning the dataset...")
    samples, statistics = read_dataset(dataset_path)
    samples = select_stratified_sample(samples, args.max_rows)
    if len(samples) < 100:
        raise ValueError("At least 100 usable rows are required for this experiment.")

    train, validation, test = split_samples(samples)
    print(
        f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}"
    )

    results = train_baselines(train, validation, test)
    final_model, svm_results = train_svm_variants(train, validation, test)
    results.update(svm_results)

    selected_model = max(
        results,
        key=lambda name: results[name]["validation"]["macro_f1"],
    )
    metadata = {
        "model_version": "BanFakeNews-2.0 hybrid TF-IDF LinearSVC",
        "dataset": statistics,
        "rows_used_for_this_run": len(samples),
        "dataset_doi": "10.17632/kjh887ct4j.1",
        "split": "stratified 70/15/15",
        "random_state": RANDOM_STATE,
        "comparison_results": results,
        "selected_model": selected_model,
        "selection_metric": "validation macro F1",
        "deployed_model": "Hybrid Linear SVM",
        "rq1_comparison": "word-only vs character-only vs hybrid Linear SVM",
        "deployed_threshold": WEBSITE_THRESHOLD,
        "limitations": [
            "The model classifies language patterns; it does not verify facts.",
            "Banglish evaluation uses a controlled transliterated set, not a naturally collected or human-reviewed Banglish dataset.",
            "A random split may not represent future news or completely new sources.",
        ],
    }
    final_model.capstone_metadata_ = metadata

    output_model = (
        ACTIVE_MODEL
        if args.replace_model
        else PROJECT_ROOT / "models" / "retrained_candidate.joblib"
    )
    joblib.dump(final_model, output_model, compress=3)
    write_reports(metadata, REPORT_DIR)

    print(f"Selected model: {selected_model}")
    print(f"Saved model: {output_model}")
    print(f"Saved reports: {REPORT_DIR}")
    if not args.replace_model:
        print("The active website model was not replaced.")


if __name__ == "__main__":
    main()
# ====================== TRAINING RUNNER: END ========================
