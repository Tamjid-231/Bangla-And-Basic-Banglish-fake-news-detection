# ===================== DECISION SETTINGS: START =====================
# The updated model score is aligned to this threshold during training.
DEFAULT_DECISION_THRESHOLD = 0.5
# ====================== DECISION SETTINGS: END ======================


# ======================= LABEL DECISION: START ======================
def label_from_score(score: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> str:
    """LinearSVC score is positive toward real and negative toward fake."""
    return "real" if score >= threshold else "fake"
# ======================== LABEL DECISION: END =======================
