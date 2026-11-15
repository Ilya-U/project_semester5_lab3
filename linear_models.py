from __future__ import annotations

import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Z-масштабирование признаков по статистикам train (защита от σ=0)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Численно устойчивая логистическая функция."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out



def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    Xtr, Xte = _standardize(X_train, X_test)
    n, d = Xtr.shape

    w = np.zeros(d, dtype=float)
    b = 0.0
    lr = float(data.learning_rate)
    epochs = int(data.epochs)
    loss_history: list[float] = []

    for _ in range(epochs):
        y_pred = Xtr @ w + b
        err = y_pred - y_train
        loss_history.append(float(np.mean(err ** 2)))

        grad_w = (2.0 / n) * (Xtr.T @ err)
        grad_b = (2.0 / n) * float(err.sum())
        w -= lr * grad_w
        b -= lr * grad_b

    predictions = Xte @ w + b
    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    Xtr, Xte = _standardize(X_train, X_test)
    n, d = Xtr.shape

    w = np.zeros(d, dtype=float)
    b = 0.0
    lr = float(data.learning_rate)
    epochs = int(data.epochs)
    loss_history: list[float] = []
    eps = 1e-12

    for _ in range(epochs):
        z = Xtr @ w + b
        p = _sigmoid(z)
        bce = -np.mean(
            y_train * np.log(p + eps) + (1.0 - y_train) * np.log(1.0 - p + eps)
        )
        loss_history.append(float(bce))

        err = p - y_train
        grad_w = (1.0 / n) * (Xtr.T @ err)
        grad_b = (1.0 / n) * float(err.sum())
        w -= lr * grad_w
        b -= lr * grad_b

    proba = _sigmoid(Xte @ w + b)
    predictions = (proba >= 0.5).astype(int)
    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        probabilities=proba,
        predictions=predictions,
    )


def _metrics(y_true, y_pred, y_proba) -> ClassificationMetrics:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, random_state=0)
    lr_model.fit(Xtr, y_train)
    lr_metrics = _metrics(
        y_test, lr_model.predict(Xte), lr_model.predict_proba(Xte)[:, 1]
    )

    dt_model = DecisionTreeClassifier(random_state=0)
    dt_model.fit(Xtr, y_train)
    dt_metrics = _metrics(
        y_test, dt_model.predict(Xte), dt_model.predict_proba(Xte)[:, 1]
    )

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(Xtr, y_train)
    knn_metrics = _metrics(
        y_test, knn_model.predict(Xte), knn_model.predict_proba(Xte)[:, 1]
    )

    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )