"""Build one row per (season-ticket holder, season) with churn label.

Churn label: holder appears in season t's season-ticket file but NOT in season t+1's.
All features use only data from season t and earlier, so there is no leakage
from the renewal decision (which happens before season t+1 starts).

Usage:
    python src/features.py --raw data/raw --out data/processed/features.csv
"""
import argparse
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ["MBB21", "MBB22", "MBB23", "MBB24", "MBB25"]
PAULEY = (34.0703, -118.4469)  # Pauley Pavilion lat/long


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * asin(sqrt(a))


def zip_distance(zips: pd.Series) -> pd.Series:
    """Miles from Pauley Pavilion using the offline `zipcodes` package."""
    import zipcodes

    cache = {}
    out = []
    for z in zips.fillna("").astype(str).str[:5]:
        if z not in cache:
            try:
                m = zipcodes.matching(z) if len(z) == 5 and z.isdigit() else []
                cache[z] = haversine_miles(*PAULEY, float(m[0]["lat"]), float(m[0]["long"])) if m else np.nan
            except Exception:
                cache[z] = np.nan
        out.append(cache[z])
    return pd.Series(out, index=zips.index)


def load_season_files(raw: Path) -> pd.DataFrame:
    frames = []
    for s in SEASONS:
        f = raw / f"season_{s}.csv"
        d = pd.read_csv(f, dtype={"Billing Zip Code": str})
        d["Season"] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["Orig Date-Time"] = pd.to_datetime(d["Orig Date-Time"])
    d["Price Type Full Name"] = d["Price Type Full Name"].fillna("").str.strip()
    return d


def season_features(d: pd.DataFrame) -> pd.DataFrame:
    pt = d["Price Type Full Name"]
    d = d.assign(
        seats=d["Order Qty"].clip(lower=0),
        comp_seats=np.where(d["Item Payment"] <= 0, d["Order Qty"].clip(lower=0), 0),
        donation_req=pt.str.contains("Donation Req").astype(int),
        courtside=pt.str.contains("Courtside").astype(int),
        pav_club=pt.str.contains("Pav Club").astype(int),
        fac_staff=pt.str.contains("Fac/Staff").astype(int),
        ada=pt.str.contains("ADA").astype(int),
        locked_in=pt.str.contains("Locked-In").astype(int),
    )
    season_open = d.groupby("Season")["Orig Date-Time"].transform("min")
    d["days_after_open"] = (d["Orig Date-Time"] - season_open).dt.days

    g = d.groupby(["Season", "Customer ID"])
    f = g.agg(
        seats=("seats", "sum"),
        spend=("Item Payment", "sum"),
        max_item_price=("Item Price", "max"),
        n_line_items=("Item Price", "size"),
        n_orders=("Orig Date-Time", "nunique"),
        comp_seats=("comp_seats", "sum"),
        first_purchase_day=("days_after_open", "min"),
        purchase_span_days=("days_after_open", lambda x: x.max() - x.min()),
        donation_req=("donation_req", "max"),
        courtside=("courtside", "max"),
        pav_club=("pav_club", "max"),
        fac_staff=("fac_staff", "max"),
        ada=("ada", "max"),
        locked_in=("locked_in", "max"),
        zip=("Billing Zip Code", "first"),
    ).reset_index()
    f["avg_price_per_seat"] = np.where(f.seats > 0, f.spend / f.seats.replace(0, np.nan), 0)
    f["comp_share"] = np.where(f.seats > 0, f.comp_seats / f.seats.replace(0, np.nan), 0)
    return f


def add_history(f: pd.DataFrame) -> pd.DataFrame:
    """Tenure, new-holder flag, and year-over-year change in seats/spend."""
    idx = {s: i for i, s in enumerate(SEASONS)}
    f["season_idx"] = f["Season"].map(idx)
    f = f.sort_values(["Customer ID", "season_idx"])

    # consecutive-season tenure (left-censored at MBB21)
    tenure = []
    prev_cid, prev_idx, t = None, None, 0
    for cid, si in zip(f["Customer ID"], f["season_idx"]):
        t = t + 1 if (cid == prev_cid and si == prev_idx + 1) else 1
        tenure.append(t)
        prev_cid, prev_idx = cid, si
    f["tenure"] = tenure

    prev = f[["Customer ID", "season_idx", "seats", "spend"]].copy()
    prev["season_idx"] += 1
    f = f.merge(prev, on=["Customer ID", "season_idx"], how="left", suffixes=("", "_prev"))
    f["is_new_holder"] = f["seats_prev"].isna().astype(int)
    f["seat_change"] = (f["seats"] - f["seats_prev"]).fillna(0)
    f["spend_change_pct"] = ((f["spend"] - f["spend_prev"]) / f["spend_prev"].replace(0, np.nan)).fillna(0).clip(-1, 3)
    return f.drop(columns=["seats_prev", "spend_prev"])


def add_single_game(f: pd.DataFrame, raw: Path) -> pd.DataFrame:
    """Did the holder also buy single-game tickets that season (extra engagement)?"""
    p = raw / "mbb_joined_data.csv"
    if not p.exists():
        f["single_game_tix"] = 0
        return f
    j = pd.read_csv(p, usecols=["Season", "Customer ID", "Order Qty"])
    sg = j.groupby(["Season", "Customer ID"])["Order Qty"].sum().rename("single_game_tix").reset_index()
    f = f.merge(sg, on=["Season", "Customer ID"], how="left")
    f["single_game_tix"] = f["single_game_tix"].fillna(0)
    return f


def add_label(f: pd.DataFrame) -> pd.DataFrame:
    present = set(zip(f["Customer ID"], f["season_idx"]))
    f["churn"] = [int((c, i + 1) not in present) for c, i in zip(f["Customer ID"], f["season_idx"])]
    # last season has no next-season outcome yet
    f.loc[f["Season"] == SEASONS[-1], "churn"] = np.nan
    return f


def build(raw: Path) -> pd.DataFrame:
    d = load_season_files(raw)
    f = season_features(d)
    f = add_history(f)
    f = add_single_game(f, raw)
    f["distance_miles"] = zip_distance(f.pop("zip"))
    f = add_label(f)
    return f.sort_values(["season_idx", "Customer ID"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/features.csv")
    a = ap.parse_args()
    out = build(Path(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby("Season").agg(holders=("Customer ID", "size"), churn_rate=("churn", "mean")).round(3))
