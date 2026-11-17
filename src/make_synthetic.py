"""Generate synthetic season-ticket data with the same schema as the real files.

The real UCLA Athletics data is confidential and not included in this repo.
This script creates fake customers so anyone can run the full pipeline:

    python src/make_synthetic.py --out data/synthetic
    python src/features.py --raw data/synthetic --out data/processed/features_synthetic.csv
    python src/train.py --features data/processed/features_synthetic.csv --figs reports/figures_synthetic \
        --metrics reports/metrics_synthetic.json

Relationships are loosely inspired by patterns in the real data (new holders,
cheaper seats and distant holders churn more), but every number is simulated.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ["MBB21", "MBB22", "MBB23", "MBB24", "MBB25"]
START = {"MBB21": "2021-09-26", "MBB22": "2022-10-09", "MBB23": "2023-10-25", "MBB24": "2024-10-21", "MBB25": "2025-10-23"}
# Public ZIP codes at varying distances from campus (not tied to any real customer)
NEAR = ["90024", "90049", "90025", "90064", "90034", "90066", "90272", "90210", "91403", "91436"]
MID = ["90501", "90275", "91302", "91361", "91101", "92660", "92618", "91320", "90803", "91711"]
FAR = ["92101", "93101", "94305", "92262", "85251", "98101", "10001", "60611", "89109", "97201"]
TIERS = [  # (name, base price per seat, share)
    ("Courtside", 1599, 0.02), ("Lower Center", 1019, 0.25), ("Lower Side", 753, 0.30),
    ("Upper Center", 433, 0.25), ("Upper", 230, 0.18),
]


def new_holder(rng, cid):
    t = rng.choice(len(TIERS), p=[x[2] for x in TIERS])
    r = rng.random()
    z = rng.choice(NEAR) if r < 0.6 else rng.choice(MID) if r < 0.85 else rng.choice(FAR)
    return dict(cid=cid, tier=t, zip=z, far=int(z in FAR), seats=int(rng.choice([1, 2, 2, 2, 3, 4])),
                donor=int(rng.random() < 0.65), staff=int(rng.random() < 0.10), tenure=0)


def churn_prob(h):
    z = -2.2 + 1.6 * (h["tenure"] == 1) + 0.35 * h["tier"] + 0.8 * h["far"] - 0.7 * h["donor"] - 0.5 * h["staff"]
    return 1 / (1 + np.exp(-z))


def main(out, n=1600, seed=7):
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    holders = [new_holder(rng, 10_000_000 + i) for i in range(n)]
    for h in holders:  # the starting base is mostly established holders
        h["tenure"] = int(rng.integers(1, 8))
    next_id = 10_000_000 + n
    single_rows = []
    for si, s in enumerate(SEASONS):
        if si > 0:  # renewals + new sign-ups
            holders = [h for h in holders if rng.random() > churn_prob(h)]
            for _ in range(int(rng.integers(60, 150))):
                holders.append(new_holder(rng, next_id)); next_id += 1
        rows = []
        start = pd.Timestamp(START[s])
        for h in holders:
            h["tenure"] += 1
            name, price, _ = TIERS[h["tier"]]
            renew = "Renewal" if h["tenure"] > 1 else "New"
            pt = ("Courtside Row 1" if name == "Courtside" else "Donation Req" if h["donor"] else "No Donation")
            staff = "Fac/Staff STX" if h["staff"] else "Season Ticket"
            ptype = f"{renew} {staff} - {pt}" if si > 0 else ("Fac/Staff Season Ticket" if h["staff"] else "Season Ticket")
            unit = round(price * (1 + 0.04 * si) * rng.uniform(0.9, 1.1))
            n_orders = 1 if rng.random() < 0.7 else int(rng.integers(2, 4))
            for k in range(n_orders):
                day = rng.integers(0, 5) if k == 0 else rng.integers(5, 120)
                q = h["seats"] if k == 0 else 1
                rows.append({
                    "Customer ID": h["cid"], "Billing Zip Code": h["zip"],
                    "Orig Date-Time": (start + pd.Timedelta(days=int(day), minutes=int(rng.integers(0, 1440)))).strftime("%Y-%m-%d %H:%M:%S"),
                    "Order Qty": q, "Item": f"20{21 + si}-20{22 + si} UCLA Men's Basketbal",
                    "Price Type Code": "SYN", "Price Type Full Name": ptype,
                    "Item Payment": unit * q, "Item Price": unit, "Item Class Code": "SEASON",
                })
            if rng.random() < 0.2:
                single_rows.append({"Season": s, "Customer ID": h["cid"], "Order Qty": int(rng.integers(1, 6))})
        pd.DataFrame(rows).to_csv(out / f"season_{s}.csv", index=False)
        print(s, len(holders), "holders")
    pd.DataFrame(single_rows).to_csv(out / "mbb_joined_data.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    main(ap.parse_args().out)
