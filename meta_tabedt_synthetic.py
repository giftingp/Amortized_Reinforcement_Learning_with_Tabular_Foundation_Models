"""
meta_tabedt_synthetic.py
------------------------
MotherNet-protocol training for TabEDT:

  * META-TRAIN the policy on a STREAM of SYNTHETIC datasets drawn from the same
    SCM prior MotherNet was pretrained on;
  * EVALUATE zero-shot on all 30 small-CC18 datasets (Table 3 protocol), never
    seen during training;
  * MotherNet embeddings for each synthetic dataset are computed ON THE FLY with
    the frozen encoder (get_embeddings is unsupervised and whole-set in-context,
    so it needs only X).

All the TabEDT components (policy, rollout, GRPO, evaluation) are imported from
meta_tabedt_cc18.py unchanged.

>>> IMPORTANT (prior fidelity) <<<
sample_synthetic_task() below ships with a *labelled approximation* of the prior
so the pipeline runs end to end. For numbers that match the MotherNet paper you
MUST replace its body with a call to ticl's own prior sampler, configured with the
SAME prior config that trained your checkpoint. See the marked block.

Run (after re-running the categorical preprocessing + extract_embeddings):
    python meta_tabedt_synthetic.py --iterations 20000 --eval-every 200 \
        --checkpoint /path/to/mn_b64_E1000_n8192_..._epoch_320.cpkt
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import config_TabEDT as cfg
import wandb

# Reuse the entire TabEDT stack unchanged:
from sklearn.model_selection import train_test_split
from meta_tabedt_cc18 import (
    PolicyNetwork, NodeSummary, gini, rollout, grpo_update, episode_return,
    count_leaves, evaluate, _mean_auc, load_all_datasets,
    predict, _auc, _terminal_brier,
)

DEVICE = cfg.device
DEFAULT_CKPT = ("/mimer/NOBACKUP/groups/oovgen/philip/hypernet/ticl/models_diff/"
                "mn_b64_E1000_n8192_continue_02_11_2026_09_55_33_epoch_320.cpkt")
MAX_FEATURES = 100   # the MotherNet encoder's input width


# ----------------------------------------------------------------------------
# Frozen MotherNet encoder (built exactly as in extract_embeddings.py)
# ----------------------------------------------------------------------------
def load_mothernet_encoder(checkpoint_path):
    from mothernet import MotherNet     # imported lazily so this file parses without ticl
    model = MotherNet(n_out=2, emsize=cfg.embed_dim, nhead=4, nhid_factor=2,
                      nlayers=12, n_features=MAX_FEATURES, y_encoder_layer=None)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt[0] if isinstance(ckpt, tuple) else ckpt
    filtered = {k: v for k, v in state.items()
                if k.startswith("encoder.") or k.startswith("transformer_encoder.")}
    model.load_state_dict(filtered, strict=False)
    model.to(DEVICE).eval()
    return model


def embed_X(model, X):
    """Standardise -> pad to 100 -> per-row MotherNet embeddings (N, embed_dim)."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    X = (X - mu) / (sd + 1e-6)                      # encoder expects standardised inputs
    n, d = X.shape
    if d < MAX_FEATURES:
        Xp = np.zeros((n, MAX_FEATURES), dtype=np.float32); Xp[:, :d] = X
    else:
        Xp = X[:, :MAX_FEATURES]
    with torch.no_grad():
        emb = model.get_embeddings(torch.tensor(Xp, dtype=torch.float32, device=DEVICE))
    return np.nan_to_num(np.asarray(emb, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------------------------------------------------------
# Synthetic prior
# ----------------------------------------------------------------------------
def _approx_mlp_scm_task(rng, max_classes=10):
    """APPROXIMATION ONLY -- a crude MLP-based SCM, NOT ticl's prior.
    Latent causes z -> random MLP -> (features X, target y). Use it to smoke-test
    the full pipeline; replace with the ticl hook below for paper fidelity."""
    n = int(rng.integers(200, 1001))            # samples
    d = int(rng.integers(4, 31))                # features
    C = int(rng.integers(2, max_classes + 1))   # classes
    k = int(rng.integers(2, 8))                 # latent causes
    h = int(rng.integers(8, 33))                # hidden width
    z = rng.standard_normal((n, k)).astype(np.float32)
    W1 = (rng.standard_normal((k, h)) / np.sqrt(k)).astype(np.float32)
    a = np.tanh(z @ W1)
    W2 = (rng.standard_normal((h, d)) / np.sqrt(h)).astype(np.float32)
    X = (a @ W2 + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    W3 = (rng.standard_normal((h, C)) / np.sqrt(h)).astype(np.float32)
    logits = a @ W3 + 0.1 * rng.standard_normal((n, C)).astype(np.float32)
    y = logits.argmax(axis=1).astype(np.int64)
    if len(np.unique(y)) < 2:                   # guarantee a 2nd class exists
        y = (X[:, 0] > np.median(X[:, 0])).astype(np.int64)
    return X, y


# Faithful MotherNet prior: ticl's BagPrior built from the checkpoint's own
# prior config. Lazily instantiated so this module still imports without ticl
# (e.g. for the approximate smoke-test). Set PRIOR_CONFIG_PATH via --prior-config.
PRIOR_CONFIG_PATH = "prior_config_1.pkl"
PRIOR_DEVICE = "cpu"
# Difficulty curriculum for the prior (enrich multiclass). Set via CLI in __main__.
PRIOR_MIN_CLASSES = 2
PRIOR_CLASS_ACCEPT_BASE = 1.0
PRIOR_CLASS_ACCEPT_SLOPE = 0.0
PRIOR_BALANCED_FRAC = None      # None = checkpoint default (balanced=False); else mix
_SAMPLER = None


def _get_sampler():
    global _SAMPLER
    if _SAMPLER is None:
        from ticl_prior_hook import TiclPriorSampler
        _SAMPLER = TiclPriorSampler(PRIOR_CONFIG_PATH, device=PRIOR_DEVICE,
                                    min_classes=PRIOR_MIN_CLASSES,
                                    class_accept_base=PRIOR_CLASS_ACCEPT_BASE,
                                    class_accept_slope=PRIOR_CLASS_ACCEPT_SLOPE,
                                    balanced_frac=PRIOR_BALANCED_FRAC,
                                    verbose=True)
    return _SAMPLER


def sample_synthetic_task(rng, max_classes=10):
    # Real MotherNet prior (prior_type='prior_bag', weights mlp=0.961/gp=0.039).
    # max_classes is ignored here -- the prior's 'classification' config controls it.
    return _get_sampler().sample(rng)
    # --- smoke-test WITHOUT ticl: comment the line above and use the line below ---
    # return _approx_mlp_scm_task(rng, max_classes)


# ----------------------------------------------------------------------------
# In-memory synthetic dataset with a pluggable reward:
#   reward_mode='brier' -> baseline: in-sample Gini penalty + in-sample Brier
#   reward_mode='auc'   -> held-out AUC: build on these rows, score on X_ho/y_ho
# ----------------------------------------------------------------------------
class SyntheticMetaDataset:
    def __init__(self, X, y, emb, reward_mode="brier",
                 X_ho=None, y_ho=None, n_classes=None, name="synthetic"):
        self.X = np.nan_to_num(np.asarray(X, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, np.int64)
        if n_classes is None:
            uniq = np.unique(y)
            remap = {c: i for i, c in enumerate(uniq)}       # contiguous 0..C-1 labels
            y = np.array([remap[v] for v in y], dtype=np.int64)
            self.classes = len(uniq)
        else:
            self.classes = int(n_classes)                    # labels already 0..C-1
        self.y = y
        self.features = self.X.shape[1]
        self.emb = np.nan_to_num(np.asarray(emb, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.samples = len(self.y)
        self.reward_mode = reward_mode
        # held-out rows used only for the AUC reward (routed by thresholds, no emb needed)
        self.X_ho = None if X_ho is None else np.nan_to_num(
            np.asarray(X_ho, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.y_ho = None if y_ho is None else np.asarray(y_ho, np.int64)
        self.name, self.split = name, 0
        self.bins = self.lam = self.summary = None

    def prepare(self):
        self.bins = np.zeros((self.features, cfg.thresholds + 1), dtype=np.float32)
        for f in range(self.features):
            self.bins[f] = np.linspace(self.X[:, f].min(), self.X[:, f].max(),
                                       cfg.thresholds + 1)
        G_root = gini(np.arange(self.samples), self.y, self.classes)
        self.lam = cfg.beta / (self.samples * G_root + 1e-8)
        self.summary = NodeSummary(self.X, self.y, self.emb, self.samples,
                                   self.classes, cfg.embed_dim)
        return self

    # --- reward interface (overrides MetaDataset's; selected by reward_mode) ---
    def leaf_reward(self, node):
        if self.reward_mode == "auc":
            return 0.0                                        # no in-sample leaf penalty
        return - self.lam * len(node.indices) * gini(node.indices, self.y, self.classes)

    def terminal_reward(self, root):
        if self.reward_mode == "auc":
            # route held-out rows through the tree; leaf probs from build labels
            probs = predict(root, self.X_ho, self.y, self.classes, self.bins)
            auc = _auc(self.y_ho, probs, self.classes)
            if np.isnan(auc):                                # degenerate held-out split
                return cfg.beta * 0.5
            return cfg.beta * auc
        if self.reward_mode == "brier_holdout":
            # held-out Brier: leaf probs from build labels, Brier scored on held-out rows
            probs = predict(root, self.X_ho, self.y, self.classes, self.bins)
            brier = np.mean(np.sum((probs - np.eye(self.classes)[self.y_ho]) ** 2, axis=1))
            return cfg.beta * (1.0 - brier)
        return _terminal_brier(root, self.X, self.y, self.classes, cfg.beta, self.bins)


def make_synthetic_dataset(encoder, X, y, reward_mode, holdout_frac):
    """Build a SyntheticMetaDataset, embedding only the BUILD rows (build-only
    context, matching how CC-18 train embeddings are extracted)."""
    n_classes = int(np.max(y)) + 1 if len(y) else 1          # sampler returns 0..C-1
    if reward_mode in ("auc", "brier_holdout"):
        # both hold out rows: build the tree on tr, score the TERMINAL reward on ho.
        # (brier_holdout still uses the in-sample Gini leaf shaping on the build rows.)
        idx = np.arange(len(y))
        try:
            tr, ho = train_test_split(idx, test_size=holdout_frac,
                                      random_state=0, stratify=y)
        except ValueError:                                   # a class too small to stratify
            tr, ho = train_test_split(idx, test_size=holdout_frac, random_state=0)
        emb_tr = embed_X(encoder, X[tr])
        return SyntheticMetaDataset(X[tr], y[tr], emb_tr, reward_mode=reward_mode,
                                    X_ho=X[ho], y_ho=y[ho], n_classes=n_classes).prepare()
    # brier baseline: build on all rows, in-sample reward (unchanged behaviour)
    emb = embed_X(encoder, X)
    return SyntheticMetaDataset(X, y, emb, reward_mode="brier",
                                n_classes=n_classes).prepare()


# ----------------------------------------------------------------------------
# Meta-training on a STREAM of FRESH synthetic tasks (one new draw per iteration,
# so no dataset is ever reused), evaluating zero-shot on all 30 CC-18.
#
# Simple by design: on-the-fly sampling (the original procedure), both reward
# modes (--reward auc|brier), and a periodic checkpoint every --checkpoint-every
# iterations that (a) writes a permanent iter snapshot you can evaluate later and
# (b) lets the run resume after the 24h wall limit. No cache, no workers, no shards.
# ----------------------------------------------------------------------------
def meta_train_synthetic(eval_sets, encoder, n_iterations, eval_every, seed,
                         max_classes=10, reward_mode="auc", holdout_frac=0.2,
                         init_ckpt=None, checkpoint_every=5000, resume=True,
                         run_tag=""):
    # Seed torch for reproducible POLICY INIT (weights drawn at construction below).
    # The per-iteration data stream is seeded separately inside the loop so a resume
    # continues the exact same sequence of unique tasks.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    desc_dim = 6 + 2 * cfg.embed_dim
    policy_net = PolicyNetwork(cfg.embed_dim, desc_dim, hidden_dim=cfg.hidden_dim).to(DEVICE)
    if init_ckpt is not None:
        policy_net.load_state_dict(torch.load(init_ckpt, map_location=DEVICE))
        print(f"Warm-started policy from {init_ckpt}")
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.learning_rate)

    # Per-(reward,seed) names so reward arms and seeds never collide.
    base_ckpt = getattr(cfg, "meta_ckpt_path", "tabedt_synthetic.pt")
    tag = f"{reward_mode}_seed{seed}" + (f"_{run_tag}" if run_tag else "")
    best_path = base_ckpt.replace(".pt", f"_{tag}_best.pt")        # best CC-18 AUC so far
    resume_path = base_ckpt.replace(".pt", f"_{tag}_resume.pt")    # latest, for restart

    # --- simple resume: pick up from the last periodic checkpoint if present ---
    start_it, best_auc = 0, -1.0
    if resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=DEVICE)
        policy_net.load_state_dict(st["policy"])
        optimizer.load_state_dict(st["optim"])
        start_it = st["it"]
        best_auc = st.get("best_auc", -1.0)
        print(f"Resumed from {resume_path} at iter {start_it} (best_auc {best_auc:.4f})")

    # Deterministic run id so a restart continues the SAME W&B run.
    wandb.init(project="TabEDT-synthetic", name=f"synth_{tag}",
               id=f"synth_{tag}", resume="allow",
               config={"seed": seed, "reward_mode": reward_mode, "holdout_frac": holdout_frac,
                       "max_depth": cfg.max_depth, "thresholds": cfg.thresholds,
                       "learning_rate": cfg.learning_rate, "group_size": cfg.group_size,
                       "kl_coeff": cfg.kl_coeff, "min_samples_split": cfg.min_samples_split,
                       "beta": cfg.beta, "embed_dim": cfg.embed_dim,
                       "max_candidates": getattr(cfg, "max_candidates", 96),
                       "grpo_std_skip": getattr(cfg, "grpo_std_skip", 1e-3),
                       "n_eval_sets": len(eval_sets), "prior": "ticl_bag_prior",
                       "checkpoint_every": checkpoint_every,
                       "protocol": "mothernet_synthetic_onthefly_cc18_eval"})

    for it in range(start_it, n_iterations):
        # Seed the whole iteration from (seed, it): makes the task draw AND the rollout
        # sampling a deterministic function of the iteration index, so (a) every
        # iteration is a distinct task -> 100k iters = 100k unique datasets, and
        # (b) a resume at iter N reproduces the exact same stream from N onward.
        s = (seed * 1_000_003 + it) % (2 ** 31 - 1)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        it_rng = np.random.default_rng(s)

        X, y = sample_synthetic_task(it_rng, max_classes)
        ds = make_synthetic_dataset(encoder, X, y, reward_mode, holdout_frac)

        group = [rollout(policy_net, ds) for _ in range(cfg.group_size)]
        trajs = [g[0] for g in group]; roots = [g[1] for g in group]
        returns = [episode_return(t[5], cfg.gamma) for t in trajs]
        stats = grpo_update(policy_net, optimizer, trajs, returns, desc_dim)
        if stats is None:
            wandb.log({"iter": it, "skipped": 1}); continue

        leaf_counts = [count_leaves(r) for r in roots]
        log = {"iter": it, "mean_return": float(np.mean(returns)),
               "mean_leaves": float(np.mean(leaf_counts)),
               "n_stumps_in_group": sum(lc == 1 for lc in leaf_counts),
               "syn_classes": ds.classes, "syn_features": ds.features,
               "syn_samples": ds.samples, **stats}
        if reward_mode == "auc":
            log["train_heldout_auc"] = float(np.mean(returns)) / cfg.beta

        if it % eval_every == 0:
            rows = evaluate(policy_net, eval_sets)
            m = _mean_auc(rows)
            log["cc18_zero_shot_auc"] = m
            log["cc18_zero_shot_leaves"] = float(np.mean([r["leaves"] for r in rows]))
            if not np.isnan(m) and m > best_auc:
                best_auc = m
                torch.save(policy_net.state_dict(), best_path)
        wandb.log(log)

        # Periodic checkpoint: atomic resume file + a permanent iter snapshot.
        if (it + 1) % checkpoint_every == 0:
            torch.save({"it": it + 1, "policy": policy_net.state_dict(),
                        "optim": optimizer.state_dict(), "best_auc": best_auc},
                       resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)         # never half-written
            snap = base_ckpt.replace(".pt", f"_{tag}_iter{it + 1}.pt")
            torch.save(policy_net.state_dict(), snap)
            print(f"[ckpt] iter {it + 1}: wrote {snap} (+ resume, best_auc {best_auc:.4f})",
                  flush=True)

    wandb.finish()
    return policy_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="save a resume file + a permanent iter snapshot every N iters.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--reward", type=str, default="auc", choices=["auc", "brier", "brier_holdout"],
                        help="auc = held-out AUC reward; brier = in-sample Brier baseline.")
    parser.add_argument("--holdout-frac", type=float, default=0.2,
                        help="fraction of each synthetic task held out for the AUC reward.")
    parser.add_argument("--init-ckpt", type=str, default=None,
                        help="optional warm-start policy weights.")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any existing resume checkpoint and start fresh.")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix appended to the run name. Use it to start a NEW W&B run "
                             "and a fresh set of checkpoints (e.g. --tag restart) instead of "
                             "resuming/overwriting the original run.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                        help="frozen MotherNet encoder checkpoint.")
    parser.add_argument("--prior-config", type=str, default=PRIOR_CONFIG_PATH,
                        help="ticl prior config pickle (ckpt[3]['prior']).")
    parser.add_argument("--prior-device", type=str, default=PRIOR_DEVICE,
                        choices=["cpu", "cuda"], help="device for sampling the prior.")
    parser.add_argument("--min-classes", type=int, default=2,
                        help="hard floor on realised class count (drops degenerate draws).")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="TRAINING cap on candidate splits per node (overrides config). "
                             "Eval always uses cfg.eval_max_candidates (None = all).")
    args = parser.parse_args()

    if args.max_candidates is not None:
        cfg.max_candidates = args.max_candidates          # training-time cap for this run

    # Point the (lazily-built) ticl sampler at the requested config / device.
    PRIOR_CONFIG_PATH = args.prior_config
    PRIOR_DEVICE = args.prior_device
    PRIOR_MIN_CLASSES = args.min_classes

    eval_sets = load_all_datasets()
    print(f"CC-18 eval: {len(eval_sets)} (dataset, split) instances "
          f"across {len({d.name for d in eval_sets})} datasets.")
    print(f"Prior config: {PRIOR_CONFIG_PATH} (sampling on {PRIOR_DEVICE})")
    print(f"Reward: {args.reward}"
          + (f" (held-out frac {args.holdout_frac})" if args.reward in ("auc", "brier_holdout") else ""))

    encoder = load_mothernet_encoder(args.checkpoint)
    print("Loaded frozen MotherNet encoder.")

    policy = meta_train_synthetic(eval_sets, encoder, args.iterations,
                                  args.eval_every, args.seed, args.max_classes,
                                  reward_mode=args.reward, holdout_frac=args.holdout_frac,
                                  init_ckpt=args.init_ckpt,
                                  checkpoint_every=args.checkpoint_every,
                                  resume=not args.no_resume, run_tag=args.tag)

    rows = evaluate(policy, eval_sets)
    df = pd.DataFrame(rows)
    _sfx = f"{args.reward}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = cfg.results_file.replace(".csv", "") + f"_synthetic_cc18_{_sfx}.csv"
    df.to_csv(out, index=False)
    print(f"\nCC-18 zero-shot AUC (mean over instances): {_mean_auc(rows):.4f}")
    print(df.groupby("dataset")["auc"].agg(["mean", "std"]).to_string())
    print(f"Results saved to {out}")
