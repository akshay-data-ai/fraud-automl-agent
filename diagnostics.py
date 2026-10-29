"""Explainable, lightweight diagnostics for the experimental Fraud AutoML Agent.

No raw training records are retained. Drift uses the fitted model's training
split as the reference, never the validation/test data. The thresholds are
diagnostic heuristics, not financial-industry safety certifications.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

EPS = 1e-5
_MISSING = "<MISSING>"
_OTHER = "<OTHER>"


def data_quality_report(frame: pd.DataFrame, target: str | None = None) -> dict:
    """Provide an auditable, *descriptive* report on an uploaded CSV."""
    n = len(frame)
    if n == 0:
        raise ValueError("Cannot profile an empty dataset.")
    columns = []
    for name in frame.columns:
        series = frame[name]
        missing = int(series.isna().sum())
        columns.append({
            "Column": name,
            "Type": str(series.dtype),
            "Missing": missing,
            "Missing %": round(100 * missing / n, 2),
            "Unique non-null": int(series.nunique(dropna=True)),
        })
    duplicate_rows = int(frame.duplicated().sum())
    warnings = []
    if duplicate_rows:
        warnings.append(f"{duplicate_rows:,} duplicate rows detected. Review whether these are valid repeated transactions.")
    high_missing = [r["Column"] for r in columns if r["Missing %"] >= 20]
    if high_missing:
        warnings.append("Columns with ≥20% missing values: " + ", ".join(high_missing))
    if target is not None and target in frame:
        if frame[target].isna().any():
            warnings.append("Rows with missing target labels cannot be used to train.")
        label_counts = frame[target].astype("string").fillna(_MISSING).value_counts().rename_axis("Label").reset_index(name="Count")
        label_counts["Share %"] = (100 * label_counts["Count"] / n).round(2)
    else:
        label_counts = pd.DataFrame(columns=["Label", "Count", "Share %"])
    return {
        "rows": n,
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "column_report": pd.DataFrame(columns),
        "label_counts": label_counts,
        "warnings": warnings,
    }


def _numeric_bins(values: pd.Series) -> tuple[list[float], list[float]]:
    observed = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if not len(observed):
        return [float("-inf"), float("inf")], [1.0]
    # Deciles, deduplicated, with open tails so new observations are included.
    interior = np.unique(np.quantile(observed, np.linspace(.1, .9, 9)))
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    probs = np.histogram(observed, bins=edges)[0].astype(float)
    probs /= probs.sum()
    return edges.tolist(), probs.tolist()


def _categorical_distribution(values: pd.Series, known: list[str] | None = None) -> dict[str, float]:
    text = values.astype("string").fillna(_MISSING).astype(str)
    if known is None:
        counts = text.value_counts()
        known = [str(x) for x in counts.index[:20] if x != _MISSING]
    counts = text.where(text.isin(known + [_MISSING]), _OTHER).value_counts(normalize=True)
    return {key: float(counts.get(key, 0.0)) for key in known + [_MISSING, _OTHER]}


def training_reference(frame: pd.DataFrame, features: list[str]) -> dict:
    """Keep only aggregate distribution summaries from *training* features."""
    output = {}
    for col in features:
        s = frame[col]
        missing = float(s.isna().mean())
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            bins, proportions = _numeric_bins(s)
            output[col] = {
                "type": "numeric", "missing": missing,
                "bins": bins, "proportions": proportions,
            }
        else:
            proportions = _categorical_distribution(s)
            output[col] = {
                "type": "categorical", "missing": missing,
                "proportions": proportions,
            }
    return output


def compare_drift(reference: dict, incoming: pd.DataFrame) -> pd.DataFrame:
    """Compute numeric PSI and categorical total-variation distance.

    Numeric PSI: sum((incoming%-reference%) * ln(incoming%/reference%)).
    Categorical TVD: half the sum of absolute category-share differences.
    Neither value is a statistical proof of drift. Differences in the positive
    class mix require separately labeled data.
    """
    rows = []
    for col, ref in reference.items():
        if col not in incoming.columns:
            raise ValueError(f"New dataset is missing training feature: {col}")
        s = incoming[col]
        new_missing = float(s.isna().mean())
        old_missing = float(ref["missing"])
        if ref["type"] == "numeric":
            vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            counts = np.histogram(vals, bins=ref["bins"])[0].astype(float)
            if counts.sum():
                q = counts / counts.sum()
                p = np.asarray(ref["proportions"], dtype=float)
                p = np.clip(p, EPS, 1.0)
                q = np.clip(q, EPS, 1.0)
                distance = float(np.sum((q-p)*np.log(q/p)))
            else:
                distance = float("nan")
            metric = "PSI"
            # Heuristic warning levels only.
            shift = "Investigate" if distance >= .25 else "Review" if distance >= .10 else "Low"
        else:
            keys = list(ref["proportions"].keys())
            known = [c for c in keys if c not in {_MISSING, _OTHER}]
            new_dist = _categorical_distribution(s, known=known)
            distance = .5 * sum(abs(ref["proportions"][k] - new_dist[k]) for k in keys)
            metric = "TVD"
            shift = "Investigate" if distance >= .20 else "Review" if distance >= .10 else "Low"
        delta = new_missing - old_missing
        if abs(delta) >= .10:
            shift = "Investigate"
        elif abs(delta) >= .05 and shift == "Low":
            shift = "Review"
        rows.append({
            "Feature": col,
            "Type": ref["type"],
            "Metric": metric,
            "Shift score": round(distance, 4) if np.isfinite(distance) else np.nan,
            "Train missing %": round(old_missing*100, 2),
            "New missing %": round(new_missing*100, 2),
            "Status": shift,
        })
    order = {"Investigate": 0, "Review": 1, "Low": 2}
    return pd.DataFrame(rows).sort_values(
        "Status", key=lambda col: col.map(order)
    ).reset_index(drop=True)


def evaluate_scored_labels(result: dict, scored: pd.DataFrame, *, label_col: str) -> dict:
    """Evaluate new *independently labeled* records using the fixed threshold.

    Never use this report to retune an already-published test/holdout.
    """
    if label_col not in scored:
        raise ValueError(f"Label column {label_col!r} not found.")
    labels = scored[label_col].astype("string")
    valid = labels.notna()
    allowed = {result["positive_label"], result["negative_label"]}
    unknown = set(labels.loc[valid].astype(str).unique()) - allowed
    if unknown:
        raise ValueError(f"Unrecognized labels: {sorted(unknown)[:5]}. Expected {sorted(allowed)}")
    if valid.sum() == 0:
        raise ValueError("There are no nonmissing labels for independent evaluation.")
    truth = (labels.loc[valid] == result["positive_label"]).to_numpy(dtype=np.int8)
    prediction = (scored.loc[valid, "predicted_label"].astype(str) == result["positive_label"]).to_numpy(dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    output = {
        "Evaluated rows": int(valid.sum()),
        "Excluded unlabeled rows": int((~valid).sum()),
        "True negatives": int(tn),
        "False positives": int(fp),
        "False negatives": int(fn),
        "True positives": int(tp),
        "Precision": float(precision_score(truth, prediction, zero_division=0)),
        "Recall": float(recall_score(truth, prediction, zero_division=0)),
        "F1": float(f1_score(truth, prediction, zero_division=0)),
        "Observed positive rate": float(truth.mean()),
    }
    if len(np.unique(truth)) == 2:
        scores = scored.loc[valid, "automl_score_uncalibrated"].astype(float).to_numpy()
        output["Average Precision"] = float(average_precision_score(truth, scores))
        output["ROC AUC"] = float(roc_auc_score(truth, scores))
    else:
        output["Average Precision"] = None
        output["ROC AUC"] = None
    return output


def reliability_warnings(
    result: dict, *,
    scored: pd.DataFrame | None = None,
    evaluation: dict | None = None,
    drift: pd.DataFrame | None = None,
) -> list[str]:
    """Generate nonblocking alerts. Never convert these into model selection."""
    warnings = []
    baseline = result.get("dummy_validation_ap")
    validation_ap = float(result["table"].iloc[0]["Average Precision"])
    if baseline is not None and validation_ap <= 1.25 * baseline:
        warnings.append("Validation average precision is close to the dummy-model baseline; discrimination may be weak.")
    t = result.get("test_metrics")
    if t:
        if int(t["True Positives"]) == 0:
            warnings.append("The held-out test detected no fraud cases.")
        if int(t["Missed Frauds"]) > 0 and float(t["Recall"]) < .50:
            warnings.append("The held-out test missed over half of labeled frauds. Investigate before considering this model for any decisions.")
    if scored is not None:
        n_flags = int((scored["predicted_label"].astype(str) == result["positive_label"]).sum())
        if n_flags == 0:
            warnings.append("No transactions were flagged. This can indicate dataset shift, excessive threshold, or a genuine low-risk sample.")
        if n_flags == len(scored):
            warnings.append("Every transaction was flagged. Review the inputs and the decision threshold.")
    if drift is not None:
        important = drift.loc[drift["Status"] == "Investigate", "Feature"].tolist()
        if important:
            warnings.append("Distribution-shift indicators need investigation for: " + ", ".join(important[:8]))
    if evaluation is not None:
        if evaluation["True positives"] == 0 and evaluation["False negatives"] > 0:
            warnings.append("Labeled upload: zero fraudulent transactions detected.")
        if evaluation["False negatives"] and evaluation["Recall"] < .50:
            warnings.append("Labeled upload: fewer than half of known fraud cases were detected.")
        if evaluation["Average Precision"] is not None and evaluation["Average Precision"] <= 1.25 * evaluation["Observed positive rate"]:
            warnings.append("Labeled upload: average precision is close to observed positive prevalence.")
    return warnings
