"""
task2_eda.py  (Task 2 - Threat landscape analysis)
--------------------------------------------------
Name    : Srishti Kainth
Roll No : 2301730175

Exploratory data analysis on the login events. The goal is to find
which signals actually separate normal logins from suspicious ones,
before any model is trained.
"""

import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import CATEGORICAL, NUMERIC, PLOT_DIR, RESULT_DIR, load_data

sns.set_theme(style="whitegrid")
PALETTE = {"Normal": "#4C72B0", "Suspicious": "#C44E52"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  saved plots/{name}")


def main():
    df = load_data()
    df["status"] = df["label"].map({0: "Normal", 1: "Suspicious"})

    # ---- 1. Basic overview -------------------------------------------------
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nClass balance:")
    print(df["status"].value_counts(normalize=True).round(3))

    # ---- 2. Threat indicators: mean of each feature by class --------------
    summary = df.groupby("status")[NUMERIC].mean().T.round(2)
    summary["ratio_susp_vs_normal"] = (summary["Suspicious"] / summary["Normal"]).round(2)
    summary = summary.sort_values("ratio_susp_vs_normal", ascending=False)
    summary.to_csv(RESULT_DIR / "threat_indicators.csv")
    print("\nAverage value by class (top threat indicators first):")
    print(summary)

    # ---- Plot 1: normal vs suspicious count --------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="status", hue="status", palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Normal vs Suspicious login events")
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    save(fig, "01_class_balance.png")

    # ---- Plot 2: frequency of each type of suspicious activity -------------
    attacks = df[df["label"] == 1]["attack_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=attacks.values, y=attacks.index, color="#C44E52", ax=ax)
    ax.set_title("Frequency of suspicious activity types")
    ax.set_xlabel("Number of events")
    ax.set_ylabel("")
    save(fig, "02_attack_type_frequency.png")

    # ---- Plot 3: distribution of attack-related features -------------------
    features = ["failed_attempts_1h", "distinct_ips_24h", "data_downloaded_mb",
                "session_duration_min", "hour_of_day"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, features):
        data = df.copy()
        log_scale = col == "data_downloaded_mb"   # downloads are heavily skewed
        sns.histplot(data=data, x=col, hue="status", palette=PALETTE, element="step",
                     stat="density", common_norm=False, bins=30, log_scale=log_scale, ax=ax)
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of attack-related features: Normal vs Suspicious", y=1.01)
    save(fig, "03_feature_distributions.png")

    # ---- Plot 4: login hour (when do attacks happen?) ----------------------
    hourly = df.groupby(["hour_of_day", "status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    hourly.plot(ax=ax, color=[PALETTE["Normal"], PALETTE["Suspicious"]], marker="o")
    ax.set_title("Login events by hour of day")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Events")
    save(fig, "04_hourly_pattern.png")

    # ---- Plot 5: suspicious rate by category -------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, col in zip(axes, CATEGORICAL):
        rate = df.groupby(col)["label"].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.index, y=rate.values, color="#DD8452", ax=ax)
        ax.set_title(f"% suspicious by {col}")
        ax.set_ylabel("% suspicious")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    save(fig, "05_suspicious_rate_by_category.png")

    # ---- Plot 6: correlation heatmap ---------------------------------------
    corr = df[NUMERIC + ["label"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation between numeric features and label")
    save(fig, "06_correlation_heatmap.png")

    # numbers that go into the report
    foreign = df[df["location"] == "foreign"]
    print(f"\nShare of 'foreign' logins that are suspicious: {foreign['label'].mean():.1%}")
    print(f"Share of all suspicious events from 'foreign':  "
          f"{(df[df.label == 1]['location'] == 'foreign').mean():.1%}")


if __name__ == "__main__":
    main()
