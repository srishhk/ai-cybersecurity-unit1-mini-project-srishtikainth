# AI-Driven Cyber Threat Awareness and Detection Prototype

**Name:** Srishti Kainth
**Roll No:** 2301730175
**Course:** AI in Cyber Security Lab (ENSP355) — Lab Assignment 01 (Unit 1)
**Faculty:** Dr. Monika Khatkar
**University:** K.R. Mangalam University

---

## What this project does

I play the role of an entry-level security analyst. The system looks at login events and decides whether each one is **normal** or **suspicious**.

- **Task 1** – set up the environment and check it works
- **Task 2** – explore the data and find which signals point to an attack
- **Task 3** – train Logistic Regression and a Decision Tree to classify logins
- **Task 4** – train a small neural network (MLPClassifier) and compare it with the classic models
- **Task 5** – show how false positives and biased data hurt real users

## Dataset

Real company login logs are private, so I wrote a simulator (`src/generate_data.py`) that creates 5,000 login events (20% suspicious) based on three well-known attack patterns:

| Attack type | What it looks like |
|---|---|
| Brute force | many failed logins, late at night, password only |
| Credential stuffing | many different IP addresses, often via VPN |
| Insider data theft | valid user downloading very large amounts of data |

The data includes ~3% missing values and some overlap between classes on purpose, so it is not unrealistically clean. A second file, `foreign_travellers.csv`, holds 300 genuine employees logging in from abroad, used for the bias test in Task 5.
The attack patterns follow the MITRE ATT&CK descriptions of T1110 (Brute Force) and T1078/T1567 (Valid Accounts / Exfiltration).

Columns: `user_role, device_type, location, auth_method, hour_of_day, failed_attempts_1h, session_duration_min, data_downloaded_mb, distinct_ips_24h, is_vpn, attack_type, label`

## Folder structure

```
ai-cybersecurity-unit1-mini-project-srishtikainth/
├── data/                  # generated CSV datasets
├── src/
│   ├── common.py          # shared paths + preprocessing pipeline
│   ├── check_setup.py     # Task 1
│   ├── generate_data.py   # creates the dataset
│   ├── task2_eda.py       # Task 2
│   ├── task3_ml.py        # Task 3
│   ├── task4_dl.py        # Task 4
│   ├── task5_ethics.py    # Task 5
│   ├── run_all.py         # runs everything in order
│   └── AICS_Lab1_Srishti_Kainth_2301730175.ipynb   # full project as one notebook, with outputs
├── outputs/
│   ├── plots/             # all graphs (PNG)
│   └── results/           # metric tables (CSV/JSON)
├── report/                # technical report (PDF)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. clone the repo
git clone <repo-link>
cd ai-cybersecurity-unit1-mini-project-srishtikainth

# 2. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

# 3. install packages
pip install -r requirements.txt
```

## How to run

Run everything at once:

```bash
python src/run_all.py
```

Or run one task at a time from inside `src/`:

```bash
cd src
python check_setup.py
python generate_data.py
python task2_eda.py
python task3_ml.py
python task4_dl.py
python task5_ethics.py
```

Or open `src/AICS_Lab1_Srishti_Kainth_2301730175.ipynb` in Jupyter / VS Code and click **Run All**.

All plots are saved in `outputs/plots/` and all result tables in `outputs/results/`.

## Key results

| Model | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| Logistic Regression | 0.978 | 0.955 | 0.932 | 0.943 | ~0.03 s |
| Decision Tree | 0.977 | 0.987 | 0.896 | 0.939 | ~0.03 s |
| Neural Network (MLP) | 0.989 | 0.984 | 0.960 | 0.972 | ~0.37 s |

- The neural network catches the most attacks but takes about 14x longer to train and is much harder to explain.
- Lowering the threshold from 0.5 to 0.1 catches 10 more attacks but blocks 5x more genuine users (11 → 55).
- Because most "foreign" logins in history were attacks, the model blocked **10% of genuine employees working abroad**. Removing the location feature cut this to **0.7%**, at a small cost in recall.

Timings change slightly from machine to machine.

## References

- scikit-learn documentation — https://scikit-learn.org/stable/
- NIST Cybersecurity Framework 2.0 — https://www.nist.gov/cyberframework
- MITRE ATT&CK — T1110 Brute Force, T1078 Valid Accounts, T1567 Exfiltration Over Web Service — https://attack.mitre.org/
