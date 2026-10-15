"""Evaluate the deterministic decision rules in the web-evidence layer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


# ===================== PROJECT SETTINGS: START ======================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CASES = Path(__file__).with_name("evidence_logic_cases.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("evidence_logic_results.csv")
DEFAULT_REPORT = Path(__file__).with_name("evidence_logic_report.json")
# ====================== PROJECT SETTINGS: END =======================

# ===================== LOCAL MODULES: START =========================
sys.path.insert(0, str(SRC_DIR))
from web_evidence import analyze_evidence  # noqa: E402
# ====================== LOCAL MODULES: END ===========================


# ==================== STATUS MAPPING: START =========================
STATUS_CATEGORY = {
    "fact_checked_false": "conflicting",
    "fact_checked_true": "supporting",
    "supported_by_sources": "supporting",
    "limited_evidence": "limited",
    "conflicting_evidence": "conflicting",
    "not_enough_evidence": "insufficient",
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


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        cases = list(csv.DictReader(csv_file))
    required = {
        "id",
        "claim",
        "model_prediction",
        "expected_status",
        "result_1_source",
        "result_1_title",
        "result_2_source",
        "result_2_title",
    }
    if not cases:
        raise ValueError("The evidence case file is empty.")
    missing = required.difference(cases[0])
    if missing:
        raise ValueError(f"Evidence case file is missing columns: {sorted(missing)}")
    return cases


def evaluate_cases(cases: list[dict]) -> list[dict]:
    evaluated = []
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

        evidence = analyze_evidence(
            case["claim"],
            search_results,
            case["model_prediction"],
        )
        actual_status = evidence["status"]
        expected_category = STATUS_CATEGORY[case["expected_status"]]
        evaluated.append(
            {
                "id": case["id"],
                "language": case.get("language", ""),
                "claim": case["claim"],
                "expected_status": case["expected_status"],
                "actual_status": actual_status,
                "correct": actual_status == case["expected_status"],
                "expected_category": expected_category,
                "actual_category": evidence["category"],
                "category_correct": evidence["category"] == expected_category,
                "trusted_source_count": evidence["trusted_source_count"],
                "note": evidence["note"],
            }
        )
    return evaluated


def save_results(results: list[dict], output: Path, report_path: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    correct = sum(item["correct"] for item in results)
    expected_counts = Counter(item["expected_status"] for item in results)
    correct_counts = Counter(
        item["expected_status"] for item in results if item["correct"]
    )
    report = {
        "evaluation_type": "controlled evidence decision-logic evaluation",
        "total_cases": len(results),
        "correct_cases": correct,
        "accuracy": correct / len(results),
        "status_breakdown": {
            status: {
                "cases": count,
                "correct": correct_counts[status],
            }
            for status, count in sorted(expected_counts.items())
        },
        "failed_case_ids": [item["id"] for item in results if not item["correct"]],
        "four_way_category_accuracy": (
            sum(item["category_correct"] for item in results) / len(results)
        ),
        "scope_note": (
            "This controlled test measures source classification, text matching, "
            "fact-check priority, and status decisions. It does not measure the "
            "changing accuracy or availability of live Google News search results."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


# ================== EVALUATION HELPERS: END =========================


# ================== EVALUATION RUNNER: START ========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the labelled web-evidence decision-logic evaluation."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    cases = load_cases(args.cases.resolve())
    results = evaluate_cases(cases)
    report = save_results(results, args.output.resolve(), args.report.resolve())
    print(
        f"Evidence logic accuracy: {report['correct_cases']}/"
        f"{report['total_cases']} ({report['accuracy'] * 100:.2f}%)"
    )
    print(f"Results: {args.output.resolve()}")
    print(f"Report: {args.report.resolve()}")
    if report["failed_case_ids"]:
        print(f"Failed cases: {', '.join(report['failed_case_ids'])}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# =================== EVALUATION RUNNER: END =========================
