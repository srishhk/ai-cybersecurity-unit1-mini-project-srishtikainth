"""
generate_data.py
----------------
Name    : Srishti Kainth
Roll No : 2301730175
Course  : AI in Cyber Security Lab (ENSP355)

Creates the login-attempt dataset used in this project.

Why a generated dataset?
Real login logs from companies are private and almost never shared.
So I built a small simulator that copies how real login logs look:
most logins are normal, and a smaller share come from known attack
patterns (brute force, credential stuffing, insider data theft).

I also add a few messy things on purpose (missing values, noise,
overlap between classes) so the data is not "too clean" and the
models have to actually learn something.

Output:
    data/login_events.csv          -> main dataset (train + test)
    data/foreign_travellers.csv    -> extra set of GENUINE users logging in
                                      from abroad, used in Task 5 (bias test)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def normal_logins(n, location_probs=(0.62, 0.36, 0.02)):
    """Everyday logins by real employees."""
    return pd.DataFrame({
        "user_role": rng.choice(["employee", "admin", "contractor", "guest"], n,
                                p=[0.70, 0.08, 0.15, 0.07]),
        "device_type": rng.choice(["desktop", "mobile", "unknown"], n, p=[0.60, 0.37, 0.03]),
        "location": rng.choice(["office", "home", "foreign"], n, p=location_probs),
        "auth_method": rng.choice(["sso", "password", "otp"], n, p=[0.50, 0.30, 0.20]),
        # people mostly log in during the working day
        "hour_of_day": np.clip(rng.normal(13, 3.5, n).round(), 0, 23).astype(int),
        "failed_attempts_1h": rng.poisson(0.4, n),
        "session_duration_min": np.clip(rng.normal(45, 20, n), 1, None).round(1),
        "data_downloaded_mb": np.clip(rng.lognormal(3.0, 0.9, n), 0, None).round(1),
        "distinct_ips_24h": rng.poisson(1.2, n) + 1,
        "is_vpn": rng.choice([0, 1], n, p=[0.80, 0.20]),
        "attack_type": "none",
        "label": 0,
    })


def brute_force(n):
    """Attacker guessing passwords again and again."""
    df = normal_logins(n, location_probs=(0.05, 0.25, 0.70))
    df["auth_method"] = "password"
    df["device_type"] = rng.choice(["desktop", "unknown"], n, p=[0.4, 0.6])
    df["hour_of_day"] = rng.choice(list(range(0, 6)) + list(range(20, 24)), n)
    df["failed_attempts_1h"] = rng.poisson(9, n)
    df["session_duration_min"] = np.clip(rng.normal(6, 5, n), 1, None).round(1)
    df["attack_type"] = "brute_force"
    df["label"] = 1
    return df


def credential_stuffing(n):
    """Leaked passwords tried from many different IPs."""
    df = normal_logins(n, location_probs=(0.02, 0.28, 0.70))
    df["failed_attempts_1h"] = rng.poisson(3, n)
    df["distinct_ips_24h"] = rng.poisson(7, n) + 2
    df["is_vpn"] = rng.choice([0, 1], n, p=[0.35, 0.65])
    df["session_duration_min"] = np.clip(rng.normal(12, 8, n), 1, None).round(1)
    df["attack_type"] = "credential_stuffing"
    df["label"] = 1
    return df


def insider_exfiltration(n):
    """Valid user (often a contractor) pulling out large amounts of data."""
    df = normal_logins(n, location_probs=(0.45, 0.45, 0.10))
    df["user_role"] = rng.choice(["contractor", "employee", "admin"], n, p=[0.55, 0.30, 0.15])
    df["hour_of_day"] = np.clip(rng.normal(21, 3, n).round(), 0, 23).astype(int)
    df["data_downloaded_mb"] = np.clip(rng.lognormal(6.2, 0.7, n), 0, None).round(1)
    df["session_duration_min"] = np.clip(rng.normal(95, 35, n), 1, None).round(1)
    df["attack_type"] = "insider_exfiltration"
    df["label"] = 1
    return df


def add_missing_values(df, cols, frac=0.03):
    """Real logs have gaps (agent crashed, field not captured, etc.)."""
    df = df.copy()
    for col in cols:
        idx = rng.choice(df.index, int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


def main():
    # Sampling bias on purpose: very few normal logins come from "foreign",
    # while most attacks do. This is what a real company's history often
    # looks like, and it is exactly what Task 5 studies.
    parts = [
        normal_logins(4000),
        brute_force(420),
        credential_stuffing(380),
        insider_exfiltration(200),
    ]
    df = pd.concat(parts, ignore_index=True)

    # A small amount of "grey zone" noise: some normal users who forgot
    # their password at night, some attacks that look calm.
    noisy = rng.choice(df.index, int(len(df) * 0.04), replace=False)
    df.loc[noisy, "failed_attempts_1h"] = rng.poisson(4, len(noisy))

    df = add_missing_values(df, ["device_type", "session_duration_min",
                                 "data_downloaded_mb", "distinct_ips_24h"])

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "event_id", [f"EVT{100000 + i}" for i in range(len(df))])
    df.to_csv(DATA_DIR / "login_events.csv", index=False)

    # Separate group: 300 genuine employees working from abroad.
    # Their behaviour is completely normal; only the location is different.
    travellers = normal_logins(300, location_probs=(0.0, 0.0, 1.0))
    travellers.insert(0, "event_id", [f"TRV{500000 + i}" for i in range(300)])
    travellers.to_csv(DATA_DIR / "foreign_travellers.csv", index=False)

    print(f"login_events.csv       -> {len(df)} rows "
          f"({df['label'].mean():.1%} suspicious)")
    print(f"foreign_travellers.csv -> {len(travellers)} rows (all genuine)")


if __name__ == "__main__":
    main()
