import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

SEED = 42
ROOT = Path(__file__).resolve().parent


def load_student():
    df = pd.read_csv(ROOT / "Student_Performance.csv")
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    X = df.drop(columns=["Performance Index"]).to_numpy(dtype=float)
    y = df["Performance Index"].to_numpy(dtype=float)
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def load_german():
    df = pd.read_csv(ROOT / "german.csv", sep=";")
    y = df["Creditability"].to_numpy(dtype=int)
    X_df = df.drop(columns=["Creditability"]).copy()

    for col in X_df.columns:
        if X_df[col].dtype == object:
            try:
                X_df[col] = pd.to_numeric(X_df[col])
            except (ValueError, TypeError):
                X_df[col] = X_df[col].astype("category").cat.codes

    X = X_df.to_numpy(dtype=float)
    return train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )


def main():
    Xtr, Xte, ytr, yte = load_student()
    reg = train_linear_regression(
        RegressionInput(
            train_features=Xtr, train_target=ytr,
            test_features=Xte, test_target=yte,
            learning_rate=0.05, epochs=800,
        )
    )
    test_mse = float(mean_squared_error(yte, reg.predictions))

    Xtr_c, Xte_c, ytr_c, yte_c = load_german()
    clf_input = ClassificationInput(
        train_features=Xtr_c, train_target=ytr_c,
        test_features=Xte_c, test_target=yte_c,
        learning_rate=0.1, epochs=1000,
    )
    own = train_logistic_classifier(clf_input)
    baseline = compare_classical_models(clf_input)

    def _m(y_true, y_pred, y_proba):
        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    own_m = _m(yte_c, own.predictions, own.probabilities)

    def _as_dict(m):
        return {
            "roc_auc": m.roc_auc,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
        }

    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_m,
            "logistic_regression": _as_dict(baseline.logistic_regression),
            "decision_tree": _as_dict(baseline.decision_tree),
            "knn": _as_dict(baseline.knn),
        },
    }

    out = ROOT / "experiment_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
