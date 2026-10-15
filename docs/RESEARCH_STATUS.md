# Research status

| Requirement | Status | Evidence |
|---|---|---|
| Dataset cleaning and stratified split | Complete | `training/train_model.py`, retraining report |
| Naive Bayes and Logistic Regression baselines | Complete | `evaluation/retraining/model_metrics.json` |
| Word, character, and Hybrid SVM comparison | Complete | retraining metrics and model comparison CSV |
| Class-wise metrics and confusion matrices | Complete | `evaluation/retraining/` |
| 250 balanced Banglish cases | Complete as a controlled transliterated set | `evaluation/banglish/` |
| Raw and normalized comparison | Complete; optional XLM-R condition not run | `banglish_benchmark.json` |
| 15 Bangla + 15 English evidence cases | Complete | `evidence_logic_cases.csv` |
| Evidence decision-logic evaluation | Complete, 30/30 | `evidence_logic_report.json` |
| Model/evidence disagreement analysis | Complete | `disagreement_report.json` and case CSV |
| Controlled distribution-shift check | Complete with limited scope | `generalization_report.json` |
| Local website functional tests | Complete | `evaluation/test_functional.py` |

The controlled Banglish and distribution-shift studies have explicit limitations and should not be presented as naturally collected Banglish or a large independent external dataset.
