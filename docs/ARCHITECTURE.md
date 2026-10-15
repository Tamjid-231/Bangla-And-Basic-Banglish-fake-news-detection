# System architecture

```text
User text
   |
   +--> Local classifier
   |      Banglish check -> optional normalization -> cleaning
   |      -> Hybrid word/character TF-IDF -> Linear SVM -> 0.50 decision
   |      -> result popup opens immediately
   |
   +--> Web evidence
          original claim -> query variants -> Google News RSS
          -> source classification -> similarity/verdict rules
          -> evidence category and source links update the open popup
   |
   +--> Interface
          separate model result + evidence result + cautious combined status
```

| Component | Responsibility |
|---|---|
| `web/server.py` | Local HTTP server, background model warm-up, separate prediction/evidence endpoints, and confirmed shutdown |
| `src/preprocess.py` | Runtime text cleaning |
| `src/banglish_normalizer.py` | Curated Banglish dictionary and trigger |
| `src/model_decision.py` | Shared deployed threshold |
| `src/web_evidence.py` | Search, matching, source priority, and evidence status |
| `training/train_model.py` | Reproducible classical-model training and comparison |
| `evaluation/` | Banglish, evidence, disagreement, shift, and functional evaluation |

The local server runs on `127.0.0.1`. It has no login, database, or persistent user-text storage. It accepts only the expected local Host address and rejects malformed page-session identifiers. Prediction and evidence requests accept JSON only, limit request size, and return generic public errors while detailed failures stay in the local server log. The close control requires the same page origin, a dedicated local-action header, and a random token created for that server session before shutdown is accepted. Browser responses are not cached, unused browser permissions are disabled, and the remaining security headers restrict framing and resource loading. Internet access is used only for live Google News RSS evidence retrieval, and the downloaded feed size is limited before XML parsing.
