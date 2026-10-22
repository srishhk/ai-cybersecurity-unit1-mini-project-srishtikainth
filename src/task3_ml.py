"""
task3_ml.py  (Task 3 - Basic ML threat classification)
------------------------------------------------------
Name    : Srishti Kainth
Roll No : 2301730175

Trains two classic ML models to label a login as normal (0) or
suspicious (1):
    - Logistic Regression  (simple, easy to explain)
    - Decision Tree        (captures rules like "failed > 5 AND night")

Evaluated with accuracy, precision, recall and F1. In security, recall
(how many attacks we catch) and precision (how many alerts are real)
matter far more than plain accuracy, because the data is imbalanced.
"""

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import PLOT_DIR, RESULT_DIR, ROOT, SEED, build_preprocessor, load_data, split


def evaluate(name, model, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "train_time_s": round(train_time, 4),
        "predict_time_ms_per_1k": round(predict_time / len(X_test) * 1000 * 1000, 3),
    }
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, target_names=["Normal", "Suspicious"], digits=3))
    return metrics, pred


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    models = {
        "Logistic Regression": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        # max_depth limits the tree so it doesn't just memorise the training data
        "Decision Tree": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=SEED)),
        ]),
    }

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, model) in zip(axes, models.items()):
        metrics, pred = evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Normal", "Suspicious"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name)
        joblib.dump(model, ROOT / "outputs" / f"{name.lower().replace(' ', '_')}.joblib")
    fig.suptitle("Confusion matrices - classic ML models")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_ml_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Which features does the decision tree rely on most?
    tree = models["Decision Tree"]
    names = tree.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(tree.named_steps["clf"].feature_importances_, index=names)
    importance = importance[importance > 0].sort_values()
    importance.index = [i.split("__")[1] for i in importance.index]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importance.plot.barh(ax=ax, color="#55A868")
    ax.set_title("Decision Tree - feature importance")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "08_feature_importance.png", dpi=150)
    plt.close(fig)
    print("\nTop features (Decision Tree):")
    print(importance.sort_values(ascending=False).head(6).round(3))

    table = pd.DataFrame(results)
    table.to_csv(RESULT_DIR / "ml_results.csv", index=False)
    with open(RESULT_DIR / "ml_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n", table.to_string(index=False))


if __name__ == "__main__":
    main()
