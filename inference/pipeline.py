from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.rule_engine import RuleEngine
from core.store import ArtifactBundle


@dataclass
class FlowResult:
    flow_id: str
    consensus: str           # SAFE | KNOWN_ATTACK | NOVEL_ATTACK
    severity: str            # low | medium | high | critical
    predicted_label: int | None
    classifier_confidence: float | None
    ae_score: float
    ae_threshold: float
    top_features: list[dict] = field(default_factory=list)  # [{name, contribution}]


def _derive_severity(
    consensus: str,
    classifier_confidence: float | None,
    ae_score: float,
    ae_threshold: float,
    novel_critical_threshold: float,
) -> str:
    """Derive alert severity per SPEC §3.1 severity table."""
    if consensus == "SAFE":
        return "low"
    if consensus == "KNOWN_ATTACK":
        # Stage-1 rules have no classifier_confidence; treat as high certainty.
        if classifier_confidence is None:
            return "high"
        return "high" if classifier_confidence >= 0.8 else "medium"
    # NOVEL_ATTACK: critical when ae_score exceeds the 75th-percentile of
    # val-benign reconstruction errors (top quartile → highest novelty).
    return "critical" if ae_score > novel_critical_threshold else "high"


def _top_features_by_threshold(
    contributions: np.ndarray,
    feature_names: list[str],
    threshold: float,
    max_n: int,
) -> list[dict]:
    """Return features whose contribution share ≥ threshold, capped at max_n, min 1."""
    ranked_idx = np.argsort(contributions)[::-1]
    total = contributions.sum()
    if total > 0:
        selected = [int(j) for j in ranked_idx if contributions[j] / total >= threshold]
    else:
        selected = []
    if not selected:
        selected = [int(ranked_idx[0])]
    selected = selected[:max_n]
    return [
        {"name": feature_names[j], "contribution": round(float(contributions[j]), 6)}
        for j in selected
    ]


def run_batch_inference(
    flows: list[dict[str, float]],
    bundle: ArtifactBundle,
    clf_confidence_threshold: float = 0.5,
    feature_threshold: float = 0.05,
    max_top_features: int = 10,
    rule_engine: RuleEngine | None = None,
) -> list[FlowResult]:
    """
    Score a batch of flows through the 3-stage pipeline (SPEC §5.2).

    Stage 1 — Rule Engine   : deterministic rules; exits as KNOWN_ATTACK or SAFE.
    Stage 2 — XGBoost       : classifies known attack patterns; exits as KNOWN_ATTACK
                               when confidence ≥ clf_confidence_threshold.
    Stage 3 — MLP-AE        : always runs for ae_score; finalises NOVEL_ATTACK or SAFE
                               when Stage 2 did not fire KNOWN_ATTACK.

    XAI attributions include every feature whose contribution share ≥ feature_threshold
    (default 5 %), capped at max_top_features (default 10), with a minimum of 1.
    A focused anomaly (one dominant feature) shows 1–2 features; a diffuse one shows up
    to 10 — more honest than a fixed-N slice.

    Args:
        flows                    : list of {feature_name: value} dicts (post I3 validation)
        bundle                   : loaded ArtifactBundle (models + scalers)
        clf_confidence_threshold : CLF_CONFIDENCE_THRESHOLD config value
        feature_threshold        : minimum contribution share (0–1) to include a feature
        max_top_features         : hard cap on number of features returned per alert
        rule_engine              : override with a custom RuleEngine; defaults to
                                   RuleEngine.with_defaults()
    """
    if rule_engine is None:
        rule_engine = RuleEngine.with_defaults()

    feature_names = bundle.feature_names
    results: list[FlowResult | None] = [None] * len(flows)
    ambiguous: list[int] = []

    # ── Stage 1: rule engine ───────────────────────────────────────────────────
    for i, flow in enumerate(flows):
        rule_verdict = rule_engine.check(flow)
        if rule_verdict is not None:
            results[i] = FlowResult(
                flow_id=str(i),
                consensus=rule_verdict,
                severity=_derive_severity(
                    rule_verdict, None, 0.0,
                    bundle.ae_threshold, bundle.novel_critical_threshold,
                ),
                predicted_label=None,
                classifier_confidence=None,
                ae_score=0.0,
                ae_threshold=bundle.ae_threshold,
                top_features=[],
            )
        else:
            ambiguous.append(i)

    if not ambiguous:
        return results  # type: ignore[return-value]

    # ── Build feature matrix for ML stages ────────────────────────────────────
    # feature_names already reflects the final selected set (post all filters).
    # csv_adapter guarantees every key in feature_names is present in each flow.
    X_raw = np.array(
        [[flows[i][name] for name in feature_names] for i in ambiguous],
        dtype=np.float32,
    )

    # ── Stage 2: XGBoost classifier ───────────────────────────────────────────
    X_clf = bundle.classifier_scaler.transform(X_raw)
    labels, confidences = bundle.clf.predict(X_clf)

    # Stage 2 explainability: |feature_importances × scaled_input| per sample.
    importances = bundle.clf.feature_importances  # shape (n_features,)

    # ── Stage 3: MLP-AE anomaly detector ──────────────────────────────────────
    X_det = bundle.detector_scaler.transform(X_raw)
    ae_scores = bundle.ae.score(X_det)                        # (n_ambiguous,)
    ae_per_feature = bundle.ae.score_per_feature(X_det)       # (n_ambiguous, n_features)

    # ── Assemble results ───────────────────────────────────────────────────────
    for batch_idx, flow_idx in enumerate(ambiguous):
        label_int = int(labels[batch_idx])
        conf = float(confidences[batch_idx])
        ae_score = float(ae_scores[batch_idx])

        is_known_attack = (label_int != 0) and (conf >= clf_confidence_threshold)
        ae_verdict = "NOVEL_ATTACK" if ae_score > bundle.ae_threshold else "SAFE"
        consensus = "KNOWN_ATTACK" if is_known_attack else ae_verdict

        # Explainability: Stage 2 attribution when KNOWN_ATTACK; Stage 3 otherwise.
        if is_known_attack and importances is not None:
            # Local attribution proxy: |importances[j] × scaled_input[j]|
            contributions = np.abs(importances * X_clf[batch_idx])
        else:
            # Per-feature reconstruction error: (x_i − x̂_i)²
            contributions = ae_per_feature[batch_idx]
        top_features = _top_features_by_threshold(
            contributions, feature_names, feature_threshold, max_top_features
        )

        severity = _derive_severity(
            consensus, conf, ae_score,
            bundle.ae_threshold, bundle.novel_critical_threshold,
        )

        results[flow_idx] = FlowResult(
            flow_id=str(flow_idx),
            consensus=consensus,
            severity=severity,
            predicted_label=label_int if label_int != 0 else None,
            classifier_confidence=conf,
            ae_score=ae_score,
            ae_threshold=bundle.ae_threshold,
            top_features=top_features,
        )

    return results  # type: ignore[return-value]
