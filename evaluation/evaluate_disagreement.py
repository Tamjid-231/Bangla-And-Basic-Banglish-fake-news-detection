"""Evaluate classifier/evidence disagreement and presentation impact (RQ4/RQ6)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib


# ===================== PROJECT SETTINGS: START ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CASES = Path(__file__).with_name("evidence_logic_cases.csv")
DEFAULT_MODEL = PROJECT_ROOT / "models" / "banfakenews_tfidf_svm.joblib"
OUTPUT_CSV = Path(__file__).with_name("disagreement_results.csv")
OUTPUT_JSON = Path(__file__).with_name("disagreement_report.json")
OUTPUT_MD = PROJECT_ROOT / "Documentation" / "evaluation" / "DISAGREEMENT_ANALYSIS.md"
THRESHOLD = 0.50
# ====================== PROJECT SETTINGS: END =======================

# ===================== LOCAL MODULES: START =========================
sys.path.insert(0, str(SRC_DIR))
from preprocess import clean_text  # noqa: E402
from web_evidence import analyze_evidence  # noqa: E402
# ====================== LOCAL MODULES: END ===========================


# ==================== STATUS MAPPING: START =========================
DECISIVE_EVIDENCE_LABEL = {
    "fact_checked_false": "fake",
    "fact_checked_true": "real",
    "supported_by_sources": "real",
}
# ===================== STATUS MAPPING: END ==========================


# ================= EVALUATION HELPERS: START ========================
def make_result(source: str, title: str, case_id: str, number: int) -> dict | None:
    if not source.strip() or not title.strip():
        return None
    return {
        "source": source.strip(),
        "source_url": "",
        "title": title.strip(),
        "snippet": "",
        "url": f"https://example.test/{case_id.lower()}/{number}",
    }


def presentation_action(status: str, disagreement: bool | None) -> str:
    if disagreement is True:
        return "manual_review"
    if status in {"conflicting_evidence", "limited_evidence", "not_enough_evidence"}:
        return "manual_review_or_unverified"
    if status in {"fact_checked_false", "fact_checked_true"}:
        return "fact_check_supported_result"
    if status == "supported_by_sources" and disagreement is False:
        return "cautious_supported_result"
    return "manual_review_or_unverified"


# ================== EVALUATION HELPERS: END =========================


# ================== EVALUATION RUNNER: START ========================
def main() -> None:
    with DEFAULT_CASES.open("r", encoding="utf-8-sig", newline="") as csv_file:
        cases = list(csv.DictReader(csv_file))
    model = joblib.load(DEFAULT_MODEL)

    results = []
    for case in cases:
        search_results = []
        for number in (1, 2):
            result = make_result(
                case[f"result_{number}_source"],
                case[f"result_{number}_title"],
                case["id"],
                number,
            )
            if result:
                search_results.append(result)

        score = float(model.decision_function([clean_text(case["claim"])])[0])
        model_prediction = "real" if score >= THRESHOLD else "fake"
        evidence = analyze_evidence(case["claim"], search_results, model_prediction)
        evidence_label = DECISIVE_EVIDENCE_LABEL.get(evidence["status"])
        disagreement = (
            model_prediction != evidence_label if evidence_label is not None else None
        )
        action = presentation_action(evidence["status"], disagreement)
        results.append(
            {
                "id": case["id"],
                "language": case.get("language", ""),
                "claim": case["claim"],
                "model_prediction": model_prediction,
                "decision_score": f"{score:.6f}",
                "evidence_status": evidence["status"],
                "evidence_label": evidence_label or "indeterminate",
                "comparable": evidence_label is not None,
                "disagreement": disagreement if disagreement is not None else "not_applicable",
                "presentation_action": action,
            }
        )

    comparable = [row for row in results if row["comparable"]]
    disagreements = [row for row in comparable if row["disagreement"] is True]
    cautious_cases = [
        row
        for row in results
        if row["presentation_action"] in {"manual_review", "manual_review_or_unverified"}
    ]
    report = {
        "evaluation_type": "controlled classifier/evidence disagreement analysis",
        "total_cases": len(results),
        "comparable_decisive_cases": len(comparable),
        "indeterminate_evidence_cases": len(results) - len(comparable),
        "disagreement_cases": len(disagreements),
        "disagreement_rate_on_comparable_cases": (
            len(disagreements) / len(comparable) if comparable else None
        ),
        "disagreement_rate_on_all_cases": len(disagreements) / len(results),
        "rq6_presentation_impact": {
            "binary_only_definitive_outputs": len(results),
            "separated_output_manual_or_unverified_cases": len(cautious_cases),
            "separated_output_supported_cases": len(results) - len(cautious_cases),
            "interpretation": (
                "Separating evidence prevents the interface from presenting every model "
                "prediction as an equally strong final conclusion."
            ),
        },
        "scope_note": (
            "This analysis uses controlled evidence fixtures and the actual saved "
            "classifier. It does not measure changing live-search accuracy or user trust."
        ),
    }

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    disagreement_rate = report["disagreement_rate_on_comparable_cases"] or 0
    lines = [
        "# Classifier-Evidence Disagreement Analysis (RQ4 and RQ6)",
        "",
        "## RQ4 Result",
        "",
        f"- Controlled cases: {report['total_cases']}",
        f"- Cases with decisive evidence labels: {report['comparable_decisive_cases']}",
        "- Cases with limited/conflicting/insufficient evidence: "
        f"{report['indeterminate_evidence_cases']}",
        f"- Classifier/evidence disagreements: {report['disagreement_cases']}",
        f"- Disagreement rate among comparable cases: {disagreement_rate * 100:.2f}%",
        "",
        "A disagreement means the saved classifier's real/fake prediction conflicts with",
        "a fact-check verdict or with support from at least two trusted sources.",
        "",
        "## RQ6 Presentation Impact",
        "",
        f"A model-only interface would show {len(results)} binary outputs. The separated",
        f"policy marks {len(cautious_cases)} cases as manual-review or unverified instead",
        "of presenting every output with the same confidence. This is a system-policy",
        "evaluation, not a human-user study.",
        "",
        "## Scope",
        "",
        report["scope_note"],
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"RQ4 disagreement: {len(disagreements)}/{len(comparable)} "
        f"({disagreement_rate * 100:.2f}%) comparable cases"
    )
    print(f"RQ6 cautious presentation cases: {len(cautious_cases)}/{len(results)}")


if __name__ == "__main__":
    main()
# =================== EVALUATION RUNNER: END =========================
