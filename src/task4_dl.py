"""
task4_dl.py  (Task 4 - Deep learning prototype + ML vs DL comparison)
---------------------------------------------------------------------
Name    : Srishti Kainth
Roll No : 2301730175

Trains a small neural network (scikit-learn MLPClassifier) on the exact
same split and preprocessing as Task 3, then compares it with the
classic models on:
    - detection accuracy (accuracy / precision / recall / F1)
    - training time and prediction speed
    - practical feasibility (discussed in the report)

Each model is trained 3 times and the training time is averaged, because
a single timing on a laptop is noisy.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import PLOT_DIR, RESULT_DIR, SEED, build_preprocessor, load_data, split

RUNS = 3


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=SEED),
        # 2 hidden layers (32 -> 16 neurons), ReLU, Adam optimiser, early stopping
        # so it stops once the validation score stops improving.
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                                              solver="adam", alpha=1e-3, max_iter=300,
                                              early_stopping=True, random_state=SEED),
    }


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    rows = []
    mlp_model = None
    for name, clf in make_models().items():
        times = []
        for _ in range(RUNS):
            model = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
            start = time.perf_counter()
            model.fit(X_train, y_train)
            times.append(time.perf_counter() - start)

        start = time.perf_counter()
        pred = model.predict(X_test)
        predict_time = time.perf_counter() - start

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "train_time_s": np.mean(times),
            "predict_time_ms_per_1k": predict_time / len(X_test) * 1e6,
        })
        if name.startswith("Neural"):
            mlp_model = model

    table = pd.DataFrame(rows).round(4)
    table.to_csv(RESULT_DIR / "ml_vs_dl_comparison.csv", index=False)
    print(table.to_string(index=False))

    mlp = mlp_model.named_steps["clf"]
    print(f"\nMLP stopped after {mlp.n_iter_} epochs (early stopping).")

    # ---- Plot: metric comparison -------------------------------------------
    melted = table.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
                        var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (model_name, grp) in enumerate(melted.groupby("model", sort=False)):
        x = np.arange(4) + (i - 1) * 0.27
        bars = ax.bar(x, grp["score"], width=0.27, label=model_name)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(min(0.8, melted["score"].min() - 0.03), 1.02)
    ax.set_title("Detection performance: classic ML vs neural network")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "09_ml_vs_dl_metrics.png", dpi=150)
    plt.close(fig)

    # ---- Plot: training time ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(table["model"], table["train_time_s"], color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(table["train_time_s"]):
        ax.text(v, i, f"  {v:.3f}s", va="center")
    ax.set_xlabel(f"Average training time over {RUNS} runs (seconds)")
    ax.set_title("Training cost")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "10_training_time.png", dpi=150)
    plt.close(fig)

    # ---- Plot: MLP learning curve ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(mlp.loss_curve_, label="training loss")
    ax2 = ax.twinx()
    ax2.plot(mlp.validation_scores_, color="#C44E52", label="validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Validation accuracy")
    ax.set_title("Neural network training curve")
    fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "11_mlp_training_curve.png", dpi=150)
    plt.close(fig)
    print("Plots saved.")


if __name__ == "__main__":
    main()
