#!/usr/bin/env python3
"""
compare_cart.py
---------------
Compare TabEDT (frozen, synthetic-trained policy, zero-shot) against CART
(sklearn DecisionTreeClassifier) MATCHED on max_depth and min_samples_split,
on the same 30 CC-18 datasets / splits, scored with the same AUC.

CART is otherwise standard (exact thresholds, Gini, best splitter) -- note that
TabEDT is additionally restricted to `thresholds` binned cut-points per feature,
so if anything this handicaps TabEDT, making a match a strong result.

Outputs per-dataset AUC for both, the paired difference, tree sizes, a Wilcoxon
signed-rank test over datasets, win/tie/loss, and mean ranks.

Run:  python compare_cart.py            # uses tabedt_synthetic_seed*.pt
      python compare_cart.py --checkpoints "tabedt_synthetic_seed*.pt"
"""
import glob
import re
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier

import config_TabEDT as cfg
from meta_tabedt_cc18 import (PolicyNetwork, evaluate, load_all_datasets, _auc)

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def cart_rows(eval_sets):
    """Fit matched CART per (dataset, split); return per-split AUC + leaves."""
    rows = []
    for ds in eval_sets:
        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=cfg.max_depth,
            min_samples_split=cfg.min_samples_split,
            random_state=0)
        clf.fit(ds.X, ds.y)
        proba = clf.predict_proba(ds.X_test)
        # align columns to 0..classes-1 (a class may be absent from a train split)
        full = np.zeros((ds.X_test.shape[0], ds.classes), dtype=np.float64)
        for j, c in enumerate(clf.classes_):
            full[:, int(c)] = proba[:, j]
        rows.append({"dataset": ds.name, "split": ds.split,
                     "auc": _auc(ds.y_test, full, ds.classes),
                     "leaves": int(clf.get_n_leaves())})
    return pd.DataFrame(rows)


def tabedt_rows(eval_sets, ckpt_paths):
    """Evaluate each seed checkpoint zero-shot; return per-(seed,split) AUC + leaves."""
    desc_dim = 6 + 2 * cfg.embed_dim
    frames = []
    for cp in ckpt_paths:
        m = re.search(r"seed(\d+)", cp)            # ..._seed{k}[_{tag}][_iter{N}].pt
        seed = int(m.group(1)) if m else 0
        policy = PolicyNetwork(cfg.embed_dim, desc_dim, hidden_dim=cfg.hidden_dim).to(cfg.device)
        policy.load_state_dict(torch.load(cp, map_location=cfg.device))
        policy.eval()
        d = pd.DataFrame(evaluate(policy, eval_sets))
        d["seed"] = seed
        frames.append(d)
        print(f"  evaluated {cp} (seed {seed}): mean AUC "
              f"{np.nanmean(d['auc']):.4f}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="auc", choices=["auc", "brier"],
                    help="which reward arm's checkpoints to evaluate.")
    ap.add_argument("--checkpoints", default=None,
                    help="glob for checkpoints (default: tabedt_synthetic_<reward>_seed*.pt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pattern = args.checkpoints or f"tabedt_synthetic_{args.reward}_seed*.pt"

    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise SystemExit(f"No checkpoints match {pattern!r}. "
                         f"Run the (multi-seed) training first.")
    print(f"Checkpoints: {ckpts}")

    eval_sets = load_all_datasets()
    print(f"Eval: {len(eval_sets)} (dataset, split) across "
          f"{len({d.name for d in eval_sets})} datasets "
          f"(CART matched: max_depth={cfg.max_depth}, "
          f"min_samples_split={cfg.min_samples_split})\n")

    print("TabEDT (zero-shot):")
    tab = tabedt_rows(eval_sets, ckpts)
    print("CART (matched, per-split fit)...")
    cart = cart_rows(eval_sets)

    # --- aggregate to one number per dataset ---
    # TabEDT: per-seed mean over splits, then mean +/- std over seeds
    tab_ds_seed = tab.groupby(["dataset", "seed"])["auc"].mean().reset_index()
    tab_ds = tab_ds_seed.groupby("dataset")["auc"].agg(["mean", "std"]).rename(
        columns={"mean": "tabedt_auc", "std": "tabedt_std"})
    tab_leaves = tab.groupby("dataset")["leaves"].mean().rename("tabedt_leaves")
    # CART: deterministic per split -> mean over splits
    cart_ds = cart.groupby("dataset")["auc"].mean().rename("cart_auc")
    cart_leaves = cart.groupby("dataset")["leaves"].mean().rename("cart_leaves")

    comp = pd.concat([tab_ds, tab_leaves, cart_ds, cart_leaves], axis=1).reset_index()
    comp["diff"] = comp["tabedt_auc"] - comp["cart_auc"]
    comp = comp.sort_values("diff", ascending=False)

    pd.set_option("display.width", 160)
    print("\nPer-dataset (TabEDT mean +/- std over seeds vs matched CART):")
    show = comp.copy()
    for c in ["tabedt_auc", "tabedt_std", "cart_auc", "diff",
              "tabedt_leaves", "cart_leaves"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "nan")
    print(show.to_string(index=False))

    valid = comp.dropna(subset=["tabedt_auc", "cart_auc"])
    t_mean, c_mean = valid["tabedt_auc"].mean(), valid["cart_auc"].mean()
    wins = int((valid["tabedt_auc"] > valid["cart_auc"]).sum())
    losses = int((valid["tabedt_auc"] < valid["cart_auc"]).sum())
    ties = int((valid["tabedt_auc"] == valid["cart_auc"]).sum())
    # mean ranks (1=better) across the two methods, averaged over datasets
    ranks = valid[["tabedt_auc", "cart_auc"]].rank(axis=1, ascending=False)
    mr_tab, mr_cart = ranks["tabedt_auc"].mean(), ranks["cart_auc"].mean()

    print("\n" + "=" * 60)
    print(f"datasets compared:     {len(valid)}")
    print(f"mean AUC  TabEDT:      {t_mean:.4f}")
    print(f"mean AUC  CART:        {c_mean:.4f}")
    print(f"mean diff (TabEDT-CART): {t_mean - c_mean:+.4f}")
    print(f"win / tie / loss:      {wins} / {ties} / {losses}  (TabEDT vs CART)")
    print(f"mean rank TabEDT/CART: {mr_tab:.2f} / {mr_cart:.2f}  (lower better)")
    if HAVE_SCIPY and len(valid) >= 1:
        try:
            stat, p = wilcoxon(valid["tabedt_auc"], valid["cart_auc"])
            print(f"Wilcoxon signed-rank:  stat={stat:.1f}, p={p:.4g}")
        except ValueError as e:
            print(f"Wilcoxon: not computable ({e})")
    else:
        print("Wilcoxon: scipy not available")
    print("=" * 60)
    print("Note: a critical-difference diagram needs >=3 methods; with TabEDT vs "
          "CART, the Wilcoxon test + win/tie/loss is the right report. Add CART "
          "as one column when you bring in more baselines (per-dataset TabEDT, "
          "fixed-head ablation, TabPFN upper bound) for the full CD diagram.")

    out = args.out or f"TabEDT_vs_CART_{args.reward}_summary.csv"
    comp.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
