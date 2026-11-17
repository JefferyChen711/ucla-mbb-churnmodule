"""Train and evaluate season-ticket churn models with an out-of-time test.

Train: holders in MBB22 and MBB23 (did they renew for MBB23 / MBB24?)
Test:  holders in MBB24 (did they renew for MBB25?)

MBB21 is excluded from training because it is the first season in the data:
every MBB21 holder looks "new" and tenure is unknown (left-censored).

Usage:
    python src/train.py --features data/processed/features.csv --figs reports/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

TRAIN_SEASONS = ["MBB22", "MBB23"]
TEST_SEASON = "MBB24"

NUMERIC = [
    "seats", "spend", "avg_price_per_seat", "max_item_price", "n_orders",
    "comp_share", "first_purchase_day", "purchase_span_days", "tenure",
    "seat_change", "spend_change_pct", "single_game_tix", "distance_miles",
]
FLAGS = ["is_new_holder", "donation_req", "courtside", "pav_club", "fac_staff", "ada", "locked_in"]
FEATURES = NUMERIC + FLAGS
SKEWED = ["seats", "spend", "n_orders", "purchase_span_days", "single_game_tix", "distance_miles", "first_purchase_day"]

LABELS = {
    "seats": "Seats", "spend": "Total spend ($)", "avg_price_per_seat": "Avg price / seat",
    "max_item_price": "Max seat price", "n_orders": "# orders", "comp_share": "Comp share",
    "first_purchase_day": "Days after sales open (1st buy)", "purchase_span_days": "Purchase span (days)",
    "tenure": "Tenure (seasons, capped 3)", "seat_change": "Seat change vs last yr",
    "spend_change_pct": "Spend change vs last yr", "single_game_tix": "Single-game tix bought",
    "distance_miles": "Distance to Pauley (mi)", "is_new_holder": "New holder", "donation_req": "Donation-required seat",
    "courtside": "Courtside", "pav_club": "Pavilion Club", "fac_staff": "Faculty/staff",
    "ada": "ADA seat", "locked_in": "Locked-in price",
}

COLORS = {"lr": "#8C8C8C", "xgb": "#2774AE", "base": "#BDBDBD"}  # UCLA blue for the main model


def prep(df):
    df = df.copy()
    df["tenure"] = df["tenure"].clip(upper=3)  # consistent with what training can observe
    return df


def lift_table(y, p, bins=10):
    t = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    t["decile"] = (np.arange(len(t)) * bins // len(t)) + 1
    g = t.groupby("decile").agg(n=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean"))
    g["churn_rate"] = g.churners / g.n
    g["lift"] = g.churn_rate / t.y.mean()
    g["cum_capture"] = g.churners.cumsum() / t.y.sum()
    return g


def main(features_path, figs, metrics_out):
    figs = Path(figs)
    figs.mkdir(parents=True, exist_ok=True)
    df = prep(pd.read_csv(features_path))
    train = df[df.Season.isin(TRAIN_SEASONS)]
    test = df[df.Season == TEST_SEASON]
    Xtr, ytr = train[FEATURES], train.churn.astype(int)
    Xte, yte = test[FEATURES], test.churn.astype(int)
    print(f"train n={len(train)} churn={ytr.mean():.3f} | test n={len(test)} churn={yte.mean():.3f}")

    # --- Baseline: logistic regression -------------------------------------------------
    pre = ColumnTransformer([
        ("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(lambda x: np.log1p(np.clip(x, 0, None)), feature_names_out="one-to-one")),
                          ("sc", StandardScaler())]), SKEWED),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         [c for c in NUMERIC if c not in SKEWED]),
        ("flag", "passthrough", FLAGS),
    ])
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))])
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    # --- XGBoost: pick params on a time-respecting validation (train MBB22 -> validate MBB23)
    tr0, va0 = train[train.Season == "MBB22"], train[train.Season == "MBB23"]
    grid = [dict(max_depth=d, learning_rate=lr_, n_estimators=n, min_child_weight=m)
            for d in (2, 3, 4) for lr_ in (0.03, 0.07) for n in (200, 400) for m in (1, 5)]
    best, best_auc = None, -1
    for g in grid:
        m = XGBClassifier(**g, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          eval_metric="logloss", random_state=42, n_jobs=4)
        m.fit(tr0[FEATURES], tr0.churn.astype(int))
        auc = roc_auc_score(va0.churn.astype(int), m.predict_proba(va0[FEATURES])[:, 1])
        if auc > best_auc:
            best, best_auc = g, auc
    print("best xgb params", best, "val AUC", round(best_auc, 3))
    xgb = XGBClassifier(**best, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        eval_metric="logloss", random_state=42, n_jobs=4)
    xgb.fit(Xtr, ytr)
    p_xgb = xgb.predict_proba(Xte)[:, 1]

    # --- Metrics ---------------------------------------------------------------------
    lt = lift_table(yte.values, p_xgb)
    lt_lr = lift_table(yte.values, p_lr)
    test = test.assign(p_churn=p_xgb, revenue_at_risk=p_xgb * test.spend)
    rev_lost = test.loc[yte == 1, "spend"].sum()
    top = test.sort_values("p_churn", ascending=False).head(int(len(test) * 0.2))
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_churn_rate": round(float(ytr.mean()), 3), "test_churn_rate": round(float(yte.mean()), 3),
        "xgb_params": best, "xgb_val_auc": round(best_auc, 3),
        "lr_test_auc": round(roc_auc_score(yte, p_lr), 3), "xgb_test_auc": round(roc_auc_score(yte, p_xgb), 3),
        "lr_test_pr_auc": round(average_precision_score(yte, p_lr), 3),
        "xgb_test_pr_auc": round(average_precision_score(yte, p_xgb), 3),
        "xgb_top10_lift": round(float(lt.loc[1, "lift"]), 2),
        "xgb_top10_capture": round(float(lt.loc[1, "cum_capture"]), 3),
        "xgb_top20_capture": round(float(lt.loc[2, "cum_capture"]), 3),
        "lr_top20_capture": round(float(lt_lr.loc[2, "cum_capture"]), 3),
        "test_churned_revenue": round(float(rev_lost), 0),
        "top20_churned_revenue_share": round(float(top.loc[top.churn == 1, "spend"].sum() / rev_lost), 3),
    }
    print(json.dumps(metrics, indent=2))
    Path(metrics_out).write_text(json.dumps(metrics, indent=2))
    lt.round(3).to_csv(Path(metrics_out).with_name("lift_table_xgb.csv"))

    # --- Figures ---------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p, c in [("Logistic regression", p_lr, COLORS["lr"]), ("XGBoost", p_xgb, COLORS["xgb"])]:
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(yte, p):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=COLORS["base"], lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Out-of-time test: MBB24 → MBB25 renewals")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "roc_curve.png", dpi=160); plt.close(fig)

    # Cumulative gains
    fig, ax = plt.subplots(figsize=(5.5, 5))
    x = np.r_[0, np.arange(1, 11) * 10]
    ax.plot(x, np.r_[0, lt.cum_capture * 100], marker="o", color=COLORS["xgb"], lw=2, label="XGBoost")
    ax.plot(x, np.r_[0, lt_lr.cum_capture * 100], marker="o", color=COLORS["lr"], lw=2, label="Logistic regression")
    ax.plot([0, 100], [0, 100], ls="--", color=COLORS["base"], lw=1, label="Random outreach")
    ax.set(xlabel="% of holders contacted (highest risk first)", ylabel="% of churners reached",
           title="Cumulative gains")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "cumulative_gains.png", dpi=160); plt.close(fig)

    # SHAP
    explainer = shap.TreeExplainer(xgb)
    sv = explainer(Xte.rename(columns=LABELS))
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("What drives churn risk (SHAP, test set)")
    plt.tight_layout(); plt.savefig(figs / "shap_beeswarm.png", dpi=160, bbox_inches="tight"); plt.close()
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plt.tight_layout(); plt.savefig(figs / "shap_importance.png", dpi=160, bbox_inches="tight"); plt.close()

    # Churn rate by tenure (simple EDA chart for README)
    # MBB21 excluded: tenure is unknown there (first season in the data)
    eda = df[df.Season.isin(TRAIN_SEASONS + [TEST_SEASON])].groupby("tenure").churn.mean() * 100
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar([("1 (new)" if t == 1 else ("3+" if t == 3 else str(t))) for t in eda.index], eda.values, color=COLORS["xgb"])
    for i, v in enumerate(eda.values):
        ax.text(i, v + 0.4, f"{v:.0f}%", ha="center")
    ax.set(xlabel="Consecutive seasons as holder", ylabel="Churn rate (%)", title="New holders churn the most")
    fig.tight_layout(); fig.savefig(figs / "churn_by_tenure.png", dpi=160); plt.close(fig)

    # Prioritized outreach list (IDs stay local; data/ is git-ignored)
    out = test.sort_values("revenue_at_risk", ascending=False)[
        ["Customer ID", "p_churn", "spend", "revenue_at_risk", "seats", "tenure", "churn"]]
    out.to_csv(Path(features_path).with_name("retention_priority_list.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--figs", default="reports/figures")
    ap.add_argument("--metrics", default="reports/metrics.json")
    a = ap.parse_args()
    main(a.features, a.figs, a.metrics)
