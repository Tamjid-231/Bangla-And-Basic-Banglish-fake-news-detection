"""Evaluate raw Banglish, rule-normalized Banglish, and optional XLM-R zero-shot.

The transformer condition is deliberately optional because it requires a large
external checkpoint.  Classical conditions always run from the self-contained
project files. Results are written as JSON, CSV, and a concise evaluation report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ===================== PROJECT SETTINGS: START ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET = Path(__file__).with_name("banglish") / "banglish_controlled_250.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "banfakenews_tfidf_svm.joblib"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("banglish")
DEFAULT_TRANSFORMER = "AyoubChLin/xlm-roberta-base-zeroshot-nli"
THRESHOLD = 0.50
# ====================== PROJECT SETTINGS: END =======================


# ===================== LOCAL MODULES: START =========================
sys.path.insert(0, str(SRC_DIR))
from banglish_normalizer import looks_banglish  # noqa: E402
from preprocess import clean_text  # noqa: E402
# ====================== LOCAL MODULES: END ===========================


# ==================== BENCHMARK HELPERS: START ======================
def transformer_condition_name(model_name: str) -> str:
    """Keep the report label accurate for either an exact or derived model."""
    lowered = model_name.lower()
    if "xlm-roberta" in lowered and "minilm" not in lowered:
        return "C. XLM-R zero-shot transformer"
    if "bert-base-multilingual" in lowered or "mbert" in lowered:
        return "C. mBERT zero-shot transformer"
    return "C. XLM-R-derived multilingual transformer"


def clean_without_banglish_normalization(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^\u0980-\u09FFa-z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def timed_svm_predictions(model, texts: list[str]) -> tuple[list[str], float]:
    start = time.perf_counter()
    scores = model.decision_function(texts)
    elapsed = time.perf_counter() - start
    predictions = np.where(scores >= THRESHOLD, "real", "fake").tolist()
    return predictions, elapsed


def run_transformer_zero_shot(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> tuple[list[str], dict]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Transformer benchmark requires torch, transformers, and sentencepiece."
        ) from exc

    start_load = time.perf_counter()
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
    )
    load_seconds = time.perf_counter() - start_load

    start_inference = time.perf_counter()
    outputs = classifier(
        texts,
        candidate_labels=["fake news", "authentic news"],
        hypothesis_template="This text is {}.",
        multi_label=False,
        batch_size=batch_size,
        truncation=True,
    )
    inference_seconds = time.perf_counter() - start_inference
    if isinstance(outputs, dict):
        outputs = [outputs]
    predictions = [
        "fake" if str(output["labels"][0]).lower().startswith("fake") else "real"
        for output in outputs
    ]

    model_object = classifier.model
    parameter_count = sum(parameter.numel() for parameter in model_object.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model_object.parameters()
    )
    details = {
        "model_name": model_name,
        "task": "zero-shot classification using a multilingual XNLI checkpoint",
        "device": "cpu",
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_example": inference_seconds * 1000 / len(texts),
        "parameter_count": int(parameter_count),
        "parameter_memory_mb": parameter_bytes / (1024 * 1024),
        "torch_version": torch.__version__,
    }
    return predictions, details


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Banglish Benchmark",
        "",
        "## Dataset",
        "",
        f"- Controlled held-out cases: {report['dataset']['total_cases']}",
        f"- Fake: {report['dataset']['fake_cases']}",
        f"- Real: {report['dataset']['real_cases']}",
        "- Labels are inherited from BanFakeNews-2.0.",
        "- Banglish text uses curated natural-style Roman Bangla spellings.",
        "",
        "## Results",
        "",
        "| Condition | Accuracy | Macro F1 | Fake F1 | ms/example | Model MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, result in report["conditions"].items():
        if result.get("status") != "completed":
            lines.append(f"| {condition} | not run | not run | not run | not run | not run |")
            continue
        metrics = result["metrics"]
        model_mb = result.get("model_file_mb", result.get("parameter_memory_mb", 0))
        lines.append(
            f"| {condition} | {metrics['accuracy'] * 100:.2f}% | "
            f"{metrics['macro_f1'] * 100:.2f}% | {metrics['fake_f1'] * 100:.2f}% | "
            f"{result['milliseconds_per_example']:.2f} | {model_mb:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Limitation",
            "",
            "This benchmark measures a controlled transliterated subset. It must not be "
            "presented as performance on naturally written Banglish social-media data.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===================== BENCHMARK HELPERS: END =======================


# ==================== BENCHMARK RUNNER: START =======================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the held-out Banglish benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-transformer", action="store_true")
    parser.add_argument("--transformer-model", default=DEFAULT_TRANSFORMER)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    with args.dataset.resolve().open("r", encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("The Banglish benchmark dataset is empty.")

    labels = [row["label"] for row in rows]
    raw_texts = [clean_without_banglish_normalization(row["banglish_text"]) for row in rows]
    normalized_texts = [clean_text(row["banglish_text"]) for row in rows]
    normalizer_triggered = sum(looks_banglish(row["banglish_text"]) for row in rows)

    model_path = args.model.resolve()
    model = joblib.load(model_path)
    raw_predictions, raw_seconds = timed_svm_predictions(model, raw_texts)
    normalized_predictions, normalized_seconds = timed_svm_predictions(model, normalized_texts)

    transformer_condition = transformer_condition_name(args.transformer_model)
    conditions = {
        "A. Raw Banglish + Hybrid SVM": {
            "status": "completed",
            "metrics": metric_summary(labels, raw_predictions),
            "inference_seconds": raw_seconds,
            "milliseconds_per_example": raw_seconds * 1000 / len(rows),
            "model_file_mb": model_path.stat().st_size / (1024 * 1024),
        },
        "B. Rule-normalized Banglish + Hybrid SVM": {
            "status": "completed",
            "metrics": metric_summary(labels, normalized_predictions),
            "inference_seconds": normalized_seconds,
            "milliseconds_per_example": normalized_seconds * 1000 / len(rows),
            "model_file_mb": model_path.stat().st_size / (1024 * 1024),
            "normalizer_triggered_cases": normalizer_triggered,
        },
    }

    transformer_predictions: list[str] | None = None
    if args.run_transformer:
        transformer_predictions, details = run_transformer_zero_shot(
            raw_texts,
            args.transformer_model,
            args.batch_size,
        )
        conditions[transformer_condition] = {
            "status": "completed",
            "metrics": metric_summary(labels, transformer_predictions),
            **details,
        }
    else:
        conditions[transformer_condition] = {
            "status": "not_run",
            "reason": "Run again with --run-transformer after installing optional dependencies.",
            "model_name": args.transformer_model,
        }

    raw_macro = conditions["A. Raw Banglish + Hybrid SVM"]["metrics"]["macro_f1"]
    normalized_result = conditions["B. Rule-normalized Banglish + Hybrid SVM"]
    normalized_macro = normalized_result["metrics"]["macro_f1"]
    difference = normalized_macro - raw_macro
    interpretation = (
        f"Rule normalization changed macro F1 by {difference * 100:+.2f} percentage points "
        f"and triggered for {normalizer_triggered}/{len(rows)} controlled cases. "
        "This is an experimental comparison, not proof of performance on all Banglish writing."
    )
    transformer_result = conditions[transformer_condition]
    if transformer_result.get("status") == "completed":
        transformer_macro = transformer_result["metrics"]["macro_f1"]
        normalized_ms = conditions["B. Rule-normalized Banglish + Hybrid SVM"][
            "milliseconds_per_example"
        ]
        transformer_ms = transformer_result["milliseconds_per_example"]
        speed_ratio = transformer_ms / max(normalized_ms, 0.000001)
        model_description = (
            "The exact XLM-R zero-shot transformer"
            if transformer_condition == "C. XLM-R zero-shot transformer"
            else "The XLM-R-derived zero-shot transformer"
        )
        interpretation += (
            f" {model_description} reached "
            f"{transformer_macro * 100:.2f}% macro F1, used approximately "
            f"{transformer_result['parameter_memory_mb']:.2f} MB of parameter memory, "
            f"and was about {speed_ratio:.0f} times slower per example than the normalized SVM."
        )

    report = {
        "evaluation_type": "held-out controlled Banglish benchmark",
        "dataset": {
            "path": str(args.dataset.resolve().relative_to(PROJECT_ROOT)),
            "total_cases": len(rows),
            "fake_cases": labels.count("fake"),
            "real_cases": labels.count("real"),
        },
        "conditions": conditions,
        "interpretation": interpretation,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "banglish_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "banglish_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        fieldnames = [
            "sample_id",
            "label",
            "raw_svm_prediction",
            "normalized_svm_prediction",
            "transformer_prediction",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "raw_svm_prediction": raw_predictions[index],
                    "normalized_svm_prediction": normalized_predictions[index],
                    "transformer_prediction": (
                        transformer_predictions[index] if transformer_predictions else "not_run"
                    ),
                }
            )
    documentation_report = (
        PROJECT_ROOT / "Documentation" / "evaluation" / "banglish" / "BANGGLISH_BENCHMARK.md"
    )
    documentation_report.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, documentation_report)
    print(f"Banglish benchmark report: {output_dir / 'banglish_benchmark.json'}")
    print(interpretation)


if __name__ == "__main__":
    main()
# ===================== BENCHMARK RUNNER: END ========================
