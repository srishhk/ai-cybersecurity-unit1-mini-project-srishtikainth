"""
check_setup.py  (Task 1 - validate the environment)
---------------------------------------------------
Name    : Srishti Kainth
Roll No : 2301730175

Quick sanity check: imports every required library, prints its version,
and trains a tiny model so we know the setup really works.
"""

import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

print(f"Python       : {sys.version.split()[0]}")
print(f"numpy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"scikit-learn : {sklearn.__version__}")
print(f"matplotlib   : {matplotlib.__version__}")
print(f"seaborn      : {sns.__version__}")

X, y = make_classification(n_samples=200, n_features=5, random_state=0)
score = LogisticRegression().fit(X, y).score(X, y)
print(f"\nSmoke test: tiny logistic regression trained, accuracy = {score:.2f}")
print("Environment is ready.")
