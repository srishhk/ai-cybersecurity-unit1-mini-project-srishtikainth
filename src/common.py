"""
common.py
---------
Name    : Srishti Kainth
Roll No : 2301730175

Shared helpers used by every task script: paths, loading the data,
and the preprocessing pipeline (missing values, encoding, scaling).
Keeping this in one place means every model sees the data in exactly
the same way, so the ML vs DL comparison is fair.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "outputs" / "plots"
RESULT_DIR = ROOT / "outputs" / "results"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

CATEGORICAL = ["user_role", "device_type", "location", "auth_method"]
NUMERIC = ["hour_of_day", "failed_attempts_1h", "session_duration_min",
           "data_downloaded_mb", "distinct_ips_24h", "is_vpn"]
TARGET = "label"


def load_data():
    path = DATA_DIR / "login_events.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python src/generate_data.py` first."
        )
    return pd.read_csv(path)


def build_preprocessor(categorical=CATEGORICAL, numeric=NUMERIC):
    """Missing values -> encoding -> scaling, all inside one sklearn object."""
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # median is safe with outliers
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])


def split(df):
    """Stratified 75/25 split so both sets keep the same attack ratio."""
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
