# Bangla and Basic Banglish Fake News Detection with Web Evidence

**Capstone 2 Final Project — Presidency University, Bangladesh**

We built this project to explore whether a lightweight machine-learning model can help users review Bangla and basic Banglish news text while a separate web search checks for related reporting from trusted sources.

The application always keeps two results separate:

1. a local model prediction based on language patterns; and
2. a web-evidence result based on fact-check and news-source matches.

A model prediction is not proof that a real-world claim is true or false, so the interface shows a cautious combined message instead of hiding uncertainty.

## Team

**University:** Presidency University, Bangladesh  
**Academic project:** Capstone 2 Final Project


## What we completed

- reproducible BanFakeNews-2.0 cleaning and stratified 70/15/15 split
- Multinomial Naive Bayes and Logistic Regression baselines
- word-only, character-only, and Hybrid TF-IDF Linear SVM comparison
- validation-selected Hybrid model with a deployed 0.50 threshold
- basic dictionary-based Banglish recognition and normalization
- 250-case controlled Banglish comparison; the optional XLM-R condition is prepared but not run in the saved benchmark
- separate Google News RSS web-evidence layer
- 30 controlled evidence-decision cases
- classifier/evidence disagreement analysis
- controlled distribution-shift check
- local responsive website, offline packages, and Windows launchers

## Resources and attribution

| Resource | Use | Source / licence |
|---|---|---|
| BanFakeNews-2.0 | Bangla news records and inherited labels | [Mendeley Data](https://doi.org/10.17632/kjh887ct4j.1), CC BY 4.0 |
| scikit-learn | TF-IDF, classical models, splitting, and metrics | BSD-3-Clause |
| NumPy and SciPy | Numerical and sparse-matrix operations | BSD licences |
| joblib | Saving and loading trained models | BSD-3-Clause |
| XLM-R zero-shot checkpoint | Optional Banglish research comparison | Original model terms apply |
| Google News RSS | Optional live search for related reports | External service terms apply |

## Project overview

| Item | Details |
|---|---|
| Input | Bangla or basic Banglish news text |
| Deployed model | Hybrid word + character TF-IDF with balanced Linear SVM |
| Model output | `real` or `fake`, decision score, and threshold |
| Evidence output | `supporting`, `limited`, `conflicting`, `insufficient`, or `unavailable` |
| Evidence provider | Google News RSS |
| Interface | Local responsive browser application |
| Database | None |
| Internet | Needed only for live web evidence |

## Project structure

```text
|-- data/                         BanFakeNews-2.0 CSV
|-- docs/                         Method, architecture, status, and screenshot
|-- evaluation/                   Banglish, evidence, disagreement, and shift checks
|-- models/                       Deployed, backup, and retrained model files
|-- offline_packages/             Offline Python wheels
|-- src/                          Preprocessing, decision, and evidence modules
|-- training/                     Reproducible model training
|-- web/                          Local interface and HTTP server
|-- Check_Project.bat             Functional project checks
|-- Start_Project.bat             One-click website launcher
|-- requirements.txt              Core runtime packages
`-- requirements-transformer.txt  Optional XLM-R research packages
```

## Dataset and model structure

The project starts with 61,581 source records. After cleaning and duplicate removal, 57,908 usable records remain: 9,393 fake and 48,515 real.

The stratified split uses random state `42`:

- 40,534 training records
- 8,687 validation records
- 8,687 test records

The Hybrid model combines:

- word TF-IDF unigrams and bigrams;
- character-within-word TF-IDF trigrams; and
- a class-balanced Linear SVM.

Validation macro-F1 is used for model selection. The selected raw decision threshold is aligned to `0.50` in the deployed model.

## Verified evaluation results

| Evaluation | Accuracy | Macro F1 | Fake-class F1 |
|---|---:|---:|---:|
| Held-out article test | 94.08% | 89.10% | 81.73% |
| Held-out headline-only test | 87.94% | 79.13% | 65.57% |
| Controlled Banglish: raw Hybrid SVM | 50.40% | 34.22% | 66.84% |
| Controlled Banglish: normalized Hybrid SVM | 50.40% | 34.22% | 66.84% |
| Controlled Banglish: XLM-R zero-shot | Not run | Not run | Not run |
| Differently distributed short-claim check | 46.67% | 46.43% | 42.86% |

The evidence decision logic passed all 30 controlled cases. In the disagreement study, 15 cases had decisive evidence and the classifier disagreed with that evidence in 8 cases. Full metrics, confusion matrices, timing, model size, case-level outputs, and limitations are stored in `evaluation/`.

## How the model works

1. The user enters a Bangla or basic Banglish headline, claim, or article text.
2. Runtime preprocessing lowercases the text and removes URLs, HTML, unsupported symbols, and extra spaces.
3. Roman text is treated as Banglish only when at least two words match the curated dictionary and no Bangla script is present.
4. Known Banglish words are converted to Bangla before vectorization.
5. The Hybrid TF-IDF model creates word and character features.
6. The Linear SVM produces a decision score.
7. A score at or above `0.50` gives `real`; a lower score gives `fake`.

The decision score is not a probability and does not prove a claim.

## How web evidence works

The web-evidence branch uses the original claim and runs separately from the classifier.

1. It creates useful full, sentence, tail, compact, and Banglish-normalized query variants.
2. It searches Bangladesh-focused Bangla and English Google News RSS feeds.
3. It classifies results as fact-check, trusted-news, or other sources.
4. Relevance uses 80% token overlap and 20% sequence similarity.
5. Clear fact-check verdict wording receives priority.
6. It returns supporting, limited, conflicting, insufficient, or unavailable evidence.

If the model and strong evidence point in different directions, the interface requests manual review.

## Result presentation

After a successful prediction, the interface opens the result popup immediately with the local model output. The live web-evidence search then runs in the background and updates the same popup when it finishes. The popup keeps the model label, decision score, deployed threshold, evidence status, trusted-source count, overall guidance, explanatory notes, and matching source links separate.

## Quick start

1. Install 64-bit Python 3.12 or 3.14.
2. Double-click `Start_Project.bat`.
3. Wait while the local environment is prepared from `offline_packages`.
4. Open `http://127.0.0.1:8502` if the browser does not open automatically.
5. Use **Close Application** and confirm the popup when you want to stop the local server.

Local model prediction works without internet. Only live evidence search needs internet.

## Privacy and safe use

We designed the website as a local classroom project. It listens only on `127.0.0.1`, accepts requests only for its local address, does not save submitted text, and loads the bundled model from the project folder. When live web evidence is available, the claim is used to search Google News RSS, so private or sensitive information should not be entered. The server limits request and feed sizes, validates JSON and page-session input, prevents browser caching, and disables unused browser permissions.

## Manual setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python web\server.py
```

## Research commands

```powershell
# Re-train and compare the classical models
.venv\Scripts\python training\train_model.py

# Banglish comparison
.venv\Scripts\python evaluation\benchmark_banglish.py

# Evidence, disagreement, generalisation, and functional checks
.venv\Scripts\python evaluation\evaluate_evidence_logic.py
.venv\Scripts\python evaluation\evaluate_disagreement.py
.venv\Scripts\python evaluation\evaluate_generalization.py
.venv\Scripts\python evaluation\test_functional.py
```

## Limitations

- The classifier learns writing patterns; it does not verify facts.
- Banglish support uses a small dictionary and misses many natural spellings.
- The controlled Banglish set is transliterated rather than naturally collected and human-reviewed.
- The distribution-shift check contains only 15 controlled short claims and is not a publisher- or time-based external dataset.
- Live evidence depends on current Google News results, source availability, wording, and internet access.
- A matching article is supporting evidence, not automatic proof.
- Important claims should be checked manually using multiple independent sources.

## Licence

Project source code is available under the [MIT Licence](LICENSE). Dataset records, models, packages, and external services retain their original terms.

---

<p align="center">
<strong>Author: Md. Tamjid Hossain</strong>
</p>
# Bangla-And-Banglish-Fake-News-Dectection-With-Web-Evidence
