"""
task5_ethics.py  (Task 5 - Ethical reflection and AI challenges)
----------------------------------------------------------------
Name    : Srishti Kainth
Roll No : 2301730175

Two experiments:

A) False positives block legitimate users
   We move the decision threshold of the Logistic Regression model from
   0.05 to 0.95 and count, at each step, how many genuine users would be
   blocked and how many attacks would slip through.

B) Biased data leads to unfair decisions
   In our training data almost every "foreign" login is an attack, because
   genuine staff rarely log in from abroad. We test the model on 300
   genuine employees working from abroad and measure how many get blocked.
   Then we retrain WITHOUT the location feature and compare.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from common import (CATEGORICAL, DATA_DIR, NUMERIC, PLOT_DIR, RESULT_DIR, SEED,
                    build_preprocessor, load_data, split)


def lr_model(categorical=CATEGORICAL):
    return Pipeline([
        ("prep", build_preprocessor(categorical=categorical)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def threshold_experiment(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({
            "threshold": t,
            "legit_users_blocked": fp,
            "false_positive_rate": fp / (fp + tn),
            "attacks_missed": fn,
            "recall": tp / (tp + fn),
            "f1": f1_score(y_test, pred),
        })
    table = pd.DataFrame(rows).round(4)
    table.to_csv(RESULT_DIR / "threshold_tradeoff.csv", index=False)
    print("\n--- Experiment A: threshold trade-off ---")
    print(table[table["threshold"].isin([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table["threshold"], table["legit_users_blocked"], marker="o",
            color="#C44E52", label="Genuine users blocked (false positives)")
    ax.plot(table["threshold"], table["attacks_missed"], marker="s",
            color="#4C72B0", label="Attacks missed (false negatives)")
    ax.axvline(0.5, ls="--", color="grey")
    ax.text(0.51, ax.get_ylim()[1] * 0.9, "default 0.5", color="grey")
    ax.set_xlabel("Decision threshold (probability needed to flag a login)")
    ax.set_ylabel(f"Count in test set (n={len(y_test)})")
    ax.set_title("Strict vs relaxed detection: who pays the price?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "12_threshold_tradeoff.png", dpi=150)
    plt.close(fig)
    return table


def bias_experiment(df, X_train, X_test, y_train, y_test):
    travellers = pd.read_csv(DATA_DIR / "foreign_travellers.csv")
    X_trav = travellers[CATEGORICAL + NUMERIC]

    # How (un)balanced is the training data for "foreign"?
    train = X_train.assign(label=y_train.values)
    foreign_train = train[train["location"] == "foreign"]
    print("\n--- Experiment B: bias from skewed data ---")
    print(f"'foreign' rows in training data: {len(foreign_train)} "
          f"-> {foreign_train['label'].mean():.1%} of them are attacks")

    results = []
    variants = {
        "With location feature": CATEGORICAL,
        "Location feature removed": [c for c in CATEGORICAL if c != "location"],
    }
    for name, cats in variants.items():
        model = lr_model(cats).fit(X_train, y_train)
        test_pred = model.predict(X_test)
        trav_pred = model.predict(X_trav)
        # false positive rate for normal users in the main test set, split by location
        test = X_test.assign(label=y_test.values, pred=test_pred)
        normal = test[test["label"] == 0]
        results.append({
            "model": name,
            "overall_recall": recall_score(y_test, test_pred),
            "overall_f1": f1_score(y_test, test_pred),
            "fpr_office_home_users": normal[normal["location"] != "foreign"]["pred"].mean(),
            "genuine_travellers_blocked_pct": trav_pred.mean(),
            "genuine_travellers_blocked": int(trav_pred.sum()),
        })
    table = pd.DataFrame(results).round(4)
    table.to_csv(RESULT_DIR / "bias_experiment.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["fpr_office_home_users"] * 100, width=0.4,
           label="Office/home genuine users blocked (%)", color="#4C72B0")
    ax.bar(x + 0.2, table["genuine_travellers_blocked_pct"] * 100, width=0.4,
           label="Genuine foreign travellers blocked (%)", color="#C44E52")
    for i, r in table.iterrows():
        ax.text(i - 0.2, r["fpr_office_home_users"] * 100 + 0.5,
                f"{r['fpr_office_home_users']*100:.1f}%", ha="center")
        ax.text(i + 0.2, r["genuine_travellers_blocked_pct"] * 100 + 0.5,
                f"{r['genuine_travellers_blocked_pct']*100:.1f}%", ha="center")
    ax.set_ylim(0, table["genuine_travellers_blocked_pct"].max() * 100 * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"])
    ax.set_ylabel("% of genuine users wrongly blocked")
    ax.set_title("Same behaviour, different treatment: effect of biased training data")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "13_bias_experiment.png", dpi=150)
    plt.close(fig)
    return table


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    model = lr_model().fit(X_train, y_train)
    threshold_experiment(model, X_test, y_test)
    bias_experiment(df, X_train, X_test, y_train, y_test)
    print("\nPlots saved.")


if __name__ == "__main__":
    main()
