"""
run_all.py
----------
Name    : Srishti Kainth
Roll No : 2301730175

Runs the whole project in order with one command:
    python src/run_all.py
"""

import runpy
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ("Task 1 - environment check", "check_setup.py"),
    ("Dataset generation", "generate_data.py"),
    ("Task 2 - threat landscape EDA", "task2_eda.py"),
    ("Task 3 - ML threat classification", "task3_ml.py"),
    ("Task 4 - neural network + ML vs DL", "task4_dl.py"),
    ("Task 5 - ethics experiments", "task5_ethics.py"),
]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SRC))
    for title, script in STEPS:
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
        runpy.run_path(str(SRC / script), run_name="__main__")
    print("\nAll done. Plots are in outputs/plots, tables in outputs/results.")
