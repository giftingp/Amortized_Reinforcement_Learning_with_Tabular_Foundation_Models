"""
meta_tabedt_cc18.py
-------------------
Train ONE shared PolicyNetwork across all of (small) CC-18.

Robustness fixes in this version:
  * build_candidates: skip degenerate splits and sanitize descriptors (no NaN/inf
    can reach the network);
  * PolicyNetwork: LayerNorm applied to the raw MotherNet embedding blocks of both
    the node summary and the candidate descriptors (raw embeddings extracted from
    un-standardised features have large magnitudes that blow up the scorer);
  * grpo_update: gradient clipping + skip of any non-finite update.

Run:  python meta_tabedt_cc18.py --iterations 20000 --eval-every 500
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import config_TabEDT as cfg
import wandb

NEG = -1e9
DEVICE = cfg.device


# ----------------------------------------------------------------------------
# Tree node (no positional A tensor: it is never fed to the policy)
# ----------------------------------------------------------------------------
class Node:
    def __init__(self, depth, indices, parent=None):
        self.depth = depth
        self.indices = indices
        self.parent = parent
        self.D = None
        self.is_leaf = False
        self.split_f = None
        self.split_t = None
        self.left = None
        self.right = None

    def apply_split(self, f, t):
        self.split_f = f
        self.split_t = t


# ----------------------------------------------------------------------------
# Class-count-invariant node summary:  [gini, norm_entropy, max_prob, norm_size, mean_emb]
# ----------------------------------------------------------------------------
class NodeSummary:
    D_DIM_FIXED = 4  # + embed_dim

    def __init__(self, X, y, emb, samples, classes, embed_dim):
        self.y = y
        self.emb = emb
        self.samples = samples
        self.classes = classes
        self.embed_dim = embed_dim

    def compute(self, indices):
        n = len(indices)
        if n == 0:
            return torch.zeros(self.D_DIM_FIXED + self.embed_dim, dtype=torch.float32)
        p = np.bincount(self.y[indices], minlength=self.classes) / n
        g = 1.0 - np.sum(p * p)
        nz = p[p > 0]
        ent = -np.sum(nz * np.log(nz)) / (np.log(max(self.classes, 2)) + 1e-12)  # in [0,1]
        D = np.concatenate([[g, ent, p.max(), n / self.samples],
                            self.emb[indices].mean(axis=0)])
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(D, dtype=torch.float32)


# ----------------------------------------------------------------------------
# Metrics / prediction (per-dataset class count is fine here)
# ----------------------------------------------------------------------------
def predict(node, X_data, y_train, classes, bins):
    if node.is_leaf:
        leaf_y = y_train[node.indices]
        if len(leaf_y) == 0:
            probs = np.ones(classes) / classes
        else:
            probs = np.bincount(leaf_y, minlength=classes) / len(leaf_y)
        return np.tile(probs, (X_data.shape[0], 1))
    thr = bins[node.split_f][node.split_t + 1]
    mask = X_data[:, node.split_f] <= thr
    out = np.zeros((X_data.shape[0], classes))
    out[mask] = predict(node.left, X_data[mask], y_train, classes, bins)
    out[~mask] = predict(node.right, X_data[~mask], y_train, classes, bins)
    return out


def count_leaves(node):
    return 1 if node.is_leaf else count_leaves(node.left) + count_leaves(node.right)


def gini(indices, y, classes):
    if len(indices) == 0:
        return 0.0
    p = np.bincount(y[indices], minlength=classes) / len(indices)
    return 1.0 - np.sum(p ** 2)


def _terminal_brier(root, X, y, classes, beta, bins):
    """In-sample terminal reward: beta * (1 - training Brier)."""
    probs = predict(root, X, y, classes, bins)
    brier = np.mean(np.sum((probs - np.eye(classes)[y]) ** 2, axis=1))
    return beta * (1.0 - brier)


def _gini_y(yv, classes):
    if len(yv) == 0:
        return 0.0
    p = np.bincount(yv, minlength=classes) / len(yv)
    return 1.0 - np.sum(p * p)


def _maxprob_y(yv, classes):
    if len(yv) == 0:
        return 0.0
    return np.bincount(yv, minlength=classes).max() / len(yv)


# ----------------------------------------------------------------------------
# Candidate construction: cheap validity pass -> subsample -> expensive descriptors
# ----------------------------------------------------------------------------
def build_candidates(indices, X, y, emb, bins, classes, features, thresholds,
                     embed_dim, max_candidates=None, rng=None):
    n = len(indices)
    Xi, yi, embi = X[indices], y[indices], emb[indices]
    desc_dim = 6 + 2 * embed_dim

    # --- cheap pass: which (f, t) split into two non-empty children? (no embeddings) ---
    valid = []
    for f in range(features):
        col = Xi[:, f]
        cuts = bins[f][1:thresholds + 1]                       # (T,)
        left_counts = (col[:, None] <= cuts[None, :]).sum(0)   # (T,)
        ok = (left_counts > 0) & (left_counts < n)
        for t in np.nonzero(ok)[0]:
            valid.append((f, int(t)))
    if len(valid) == 0:
        return [], np.zeros((0, desc_dim), dtype=np.float32)

    # --- subsample BEFORE computing the costly embedding means ---
    if max_candidates is not None and len(valid) > max_candidates:
        rng = rng or np.random
        sel = rng.choice(len(valid), size=max_candidates, replace=False)
        valid = [valid[i] for i in sel]

    # --- expensive pass: descriptors only for the (subsampled) valid candidates ---
    G_node = _gini_y(yi, classes)
    cand_ft, descs = [], []
    for (f, t) in valid:
        left = Xi[:, f] <= bins[f][t + 1]
        nL = int(left.sum()); nR = n - nL
        if nL == 0 or nR == 0:                       # guard: skip degenerate splits
            continue
        yL, yR = yi[left], yi[~left]
        GL, GR = _gini_y(yL, classes), _gini_y(yR, classes)
        impred = G_node - (nL / n) * GL - (nR / n) * GR
        eL = embi[left].mean(axis=0)                 # safe: nL > 0
        eR = embi[~left].mean(axis=0)                # safe: nR > 0
        descs.append(np.concatenate([
            [impred, nL / n, GL, _maxprob_y(yL, classes), GR, _maxprob_y(yR, classes)],
            eL, eR]))
        cand_ft.append((f, t))
    if len(descs) == 0:
        return [], np.zeros((0, desc_dim), dtype=np.float32)
    D = np.stack(descs).astype(np.float32)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)   # belt-and-braces
    return cand_ft, D


# ----------------------------------------------------------------------------
# Policy (shape-agnostic; LayerNorm on the raw embedding blocks)
# ----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, embed_dim, desc_dim, hidden_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        ctx_in = NodeSummary.D_DIM_FIXED + embed_dim + 1   # D + normalized depth
        self.emb_norm = nn.LayerNorm(embed_dim)            # standardize raw MotherNet embeddings
        self.trunk = nn.Sequential(
            nn.Linear(ctx_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.cand_scorer = nn.Sequential(
            nn.Linear(hidden_dim + desc_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def context(self, D, depth_norm):
        # D = [4 invariant scalars | embed_dim mean embedding]; normalize only the embedding
        scal = D[..., :NodeSummary.D_DIM_FIXED]
        emb = self.emb_norm(D[..., NodeSummary.D_DIM_FIXED:])
        D = torch.cat([scal, emb], dim=-1)
        h = self.trunk(torch.cat([D, depth_norm], dim=-1))
        return h, self.stop_head(h)

    def score(self, h, cand_desc):                          # (B,hid),(B,K,desc) -> (B,K)
        # cand_desc = [6 scalars | embed_dim left | embed_dim right]; normalize the two embeddings
        E = self.embed_dim
        scal = cand_desc[..., :6]
        embL = self.emb_norm(cand_desc[..., 6:6 + E])
        embR = self.emb_norm(cand_desc[..., 6 + E:6 + 2 * E])
        cand_desc = torch.cat([scal, embL, embR], dim=-1)
        K = cand_desc.size(1)
        z = torch.cat([h.unsqueeze(1).expand(-1, K, -1), cand_desc], dim=-1)
        return self.cand_scorer(z).squeeze(-1)


def factored_log_prob(stop_logit, scores, chosen, cand_mask):
    stop_logit = stop_logit.squeeze(-1)
    logp_cand = F.log_softmax(scores.masked_fill(~cand_mask, NEG), dim=-1)
    is_leaf = (chosen < 0)
    chosen_lp = logp_cand.gather(1, chosen.clamp(min=0).unsqueeze(1)).squeeze(1)
    return torch.where(is_leaf, F.logsigmoid(stop_logit),
                       F.logsigmoid(-stop_logit) + chosen_lp)


def factored_kl(new_stop, new_scores, old_stop, old_scores, cand_mask):
    eps = 1e-8
    ns = torch.sigmoid(new_stop.squeeze(-1)).clamp(eps, 1 - eps)
    os_ = torch.sigmoid(old_stop.squeeze(-1)).clamp(eps, 1 - eps)
    kl_stop = ns * (ns.log() - os_.log()) + (1 - ns) * ((1 - ns).log() - (1 - os_).log())
    nlp = F.log_softmax(new_scores.masked_fill(~cand_mask, NEG), dim=-1)
    olp = F.log_softmax(old_scores.masked_fill(~cand_mask, NEG), dim=-1)
    return (kl_stop + (nlp.exp() * (nlp - olp)).sum(-1)).mean()


# ----------------------------------------------------------------------------
# Dataset container (one CC-18 (name, split) instance)
# ----------------------------------------------------------------------------
class MetaDataset:
    def __init__(self, npz_path, emb_path):
        raw = np.load(npz_path)
        emb = np.load(emb_path)
        self.X = raw['X_train'].astype(np.float32)
        self.y = raw['y_train'].astype(np.int64)
        self.X_test = raw['X_test'].astype(np.float32)
        self.y_test = raw['y_test'].astype(np.int64)
        self.classes = int(raw['classes'])
        self.features = int(raw['features'])
        self.name = str(raw['dataset_name'])
        self.split = int(raw['split'])
        self.emb = emb['emb_train'].astype(np.float32)
        # sanitize any non-finite values in features/embeddings up front
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        self.emb = np.nan_to_num(self.emb, nan=0.0, posinf=0.0, neginf=0.0)
        self.samples = len(self.y)
        assert self.emb.shape[1] == cfg.embed_dim, \
            f"{self.name}: emb dim {self.emb.shape[1]} != cfg.embed_dim {cfg.embed_dim}"
        self.bins = self.lam = self.summary = None

    def prepare(self):
        self.bins = np.zeros((self.features, cfg.thresholds + 1), dtype=np.float32)
        for f in range(self.features):
            self.bins[f] = np.linspace(self.X[:, f].min(), self.X[:, f].max(), cfg.thresholds + 1)
        G_root = gini(np.arange(self.samples), self.y, self.classes)
        self.lam = cfg.beta / (self.samples * G_root + 1e-8)
        self.summary = NodeSummary(self.X, self.y, self.emb, self.samples, self.classes, cfg.embed_dim)
        return self

    # --- reward interface (rollout calls these; subclasses can override) ---
    def leaf_reward(self, node):
        """Immediate per-leaf reward: in-sample Gini penalty."""
        return - self.lam * len(node.indices) * gini(node.indices, self.y, self.classes)

    def terminal_reward(self, root):
        """Terminal reward: in-sample Brier on the dataset's own rows."""
        return _terminal_brier(root, self.X, self.y, self.classes, cfg.beta, self.bins)


# ----------------------------------------------------------------------------
# Rollout (one tree on one dataset)
# ----------------------------------------------------------------------------
def rollout(policy_net, ds, deterministic=False):
    desc_dim = 6 + 2 * cfg.embed_dim
    max_cand = getattr(cfg, "max_candidates", 96)
    D_list, depth_list, desc_list = [], [], []
    chosen_list, logp_list, reward_list, decision_list = [], [], [], []

    root = Node(depth=0, indices=np.arange(ds.samples))
    root.D = ds.summary.compute(root.indices).to(DEVICE)
    queue = deque([root])

    leaf_penalty = ds.leaf_reward          # dataset defines the per-leaf reward

    while queue:
        batch = [queue.popleft() for _ in range(min(cfg.batch_size, len(queue)))]
        D_batch = torch.stack([nd.D for nd in batch]).to(DEVICE)
        depth_batch = torch.tensor([[nd.depth / cfg.max_depth] for nd in batch],
                                   dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            H, STOP = policy_net.context(D_batch, depth_batch)

        for i, node in enumerate(batch):
            D_pre = node.D.detach().clone().to(DEVICE)
            forced = (node.depth >= cfg.max_depth or len(node.indices) < cfg.min_samples_split)
            cand_ft, cand_desc = [], np.zeros((0, desc_dim), dtype=np.float32)
            if not forced:
                cand_ft, cand_desc = build_candidates(
                    node.indices, ds.X, ds.y, ds.emb, ds.bins, ds.classes,
                    ds.features, cfg.thresholds, cfg.embed_dim, max_candidates=max_cand)
                if len(cand_ft) == 0:
                    forced = True
                elif not np.isfinite(cand_desc).all():        # guard: descriptors must be finite
                    raise ValueError(f"non-finite descriptor in {ds.name} split {ds.split} "
                                     f"(n={len(node.indices)})")

            if forced:
                node.is_leaf = True
                D_list.append(D_pre); depth_list.append(node.depth)
                desc_list.append(cand_desc); chosen_list.append(-1)
                logp_list.append(torch.tensor(0.0, device=DEVICE))
                reward_list.append(leaf_penalty(node)); decision_list.append(False)
                continue

            stop_logit_i = STOP[i]
            p_stop = torch.sigmoid(stop_logit_i).item()
            do_stop = (p_stop >= 0.5) if deterministic else \
                      (torch.rand(1, device=DEVICE).item() < p_stop)

            if do_stop:
                node.is_leaf = True
                chosen = -1
                logp = F.logsigmoid(stop_logit_i).squeeze()
                reward = leaf_penalty(node)
            else:
                with torch.no_grad():
                    desc_t = torch.tensor(cand_desc, device=DEVICE).unsqueeze(0)
                    scores = policy_net.score(H[i:i + 1], desc_t).squeeze(0)
                logp_cand = F.log_softmax(scores, dim=-1)
                k = int(torch.argmax(scores).item()) if deterministic else \
                    int(torch.distributions.Categorical(logits=scores).sample().item())
                chosen = k
                f, t = cand_ft[k]
                logp = F.logsigmoid(-stop_logit_i).squeeze() + logp_cand[k]
                reward = 0.0
                m = ds.X[node.indices, f] <= ds.bins[f][t + 1]
                li, ri = node.indices[m], node.indices[~m]
                node.apply_split(f, t)
                node.left = Node(node.depth + 1, li, parent=node)
                node.right = Node(node.depth + 1, ri, parent=node)
                node.left.D = ds.summary.compute(li).to(DEVICE)
                node.right.D = ds.summary.compute(ri).to(DEVICE)
                queue.append(node.left); queue.append(node.right)

            D_list.append(D_pre); depth_list.append(node.depth)
            desc_list.append(cand_desc); chosen_list.append(chosen)
            logp_list.append(logp.detach().clone().to(DEVICE))
            reward_list.append(reward); decision_list.append(True)

    term_r = ds.terminal_reward(root)
    if reward_list:
        reward_list[-1] += term_r
    else:
        D_list.append(root.D.detach().clone().to(DEVICE)); depth_list.append(0)
        desc_list.append(np.zeros((0, desc_dim), dtype=np.float32)); chosen_list.append(-1)
        logp_list.append(torch.tensor(0.0, device=DEVICE)); reward_list.append(term_r)
        decision_list.append(False)

    return (D_list, depth_list, desc_list, chosen_list, logp_list, reward_list, decision_list), root


def episode_return(rewards, gamma):
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


# ----------------------------------------------------------------------------
# One GRPO update from a group of rollouts on a SINGLE dataset
# ----------------------------------------------------------------------------
def grpo_update(policy_net, optimizer, trajs, returns, desc_dim):
    mean_r = float(np.mean(returns))
    raw_std = float(np.std(returns))
    # Skip no-signal groups: when the within-group return spread is tiny, the trees are
    # essentially equivalent, and standardising (r_i - mean)/std would amplify rollout
    # noise into unit-scale advantages -- a confident gradient step on no real signal.
    # The threshold is on the spread, NOT the mean, so negative-mean groups are unaffected.
    if raw_std < getattr(cfg, "grpo_std_skip", 1e-3):
        return None
    std_r = raw_std + 1e-8
    all_D, all_depth, all_desc, all_chosen, all_lp, all_adv, all_w = [], [], [], [], [], [], []
    for i, traj in enumerate(trajs):
        D_list, depth_list, desc_list, chosen_list, lps, _, dlist = traj
        adv = (returns[i] - mean_r) / std_r
        w_i = 1.0 / max(sum(dlist), 1)
        for j in range(len(dlist)):
            if not dlist[j]:
                continue
            all_D.append(D_list[j]); all_depth.append(depth_list[j])
            all_desc.append(desc_list[j]); all_chosen.append(chosen_list[j])
            all_lp.append(lps[j]); all_adv.append(adv); all_w.append(w_i)
    if not all_chosen:
        return None

    M = len(all_desc)
    K_max = max(max((d.shape[0] for d in all_desc), default=1), 1)
    cand_desc = torch.zeros(M, K_max, desc_dim, device=DEVICE)
    cand_mask = torch.zeros(M, K_max, dtype=torch.bool, device=DEVICE)
    for m, d in enumerate(all_desc):
        if d.shape[0] > 0:
            cand_desc[m, :d.shape[0]] = torch.tensor(d, device=DEVICE)
            cand_mask[m, :d.shape[0]] = True

    states_D = torch.stack(all_D).to(DEVICE)
    states_depth = (torch.tensor(all_depth, dtype=torch.float32, device=DEVICE)
                    / cfg.max_depth).unsqueeze(1)
    chosen = torch.tensor(all_chosen, device=DEVICE)
    old_lp = torch.stack(all_lp).to(DEVICE).detach()
    adv_t = torch.tensor(all_adv, dtype=torch.float32, device=DEVICE)
    w_t = torch.tensor(all_w, dtype=torch.float32, device=DEVICE)
    w_sum = w_t.sum()

    leaf_frac = (chosen < 0).float().mean().item()
    root_mask = states_depth.squeeze(1) == 0
    rl = root_mask & (chosen < 0); rs = root_mask & (chosen >= 0)
    root_leaf_adv = adv_t[rl].mean().item() if rl.any() else 0.0
    root_split_adv = adv_t[rs].mean().item() if rs.any() else 0.0

    with torch.no_grad():
        oh, ostop = policy_net.context(states_D, states_depth)
        oscores = policy_net.score(oh, cand_desc)

    last_pl, last_kl = 0.0, 0.0
    target_kl = getattr(cfg, "target_kl", None)      # None disables; e.g. 0.05 caps per-update KL
    stopped_on_kl = 0
    for _ in range(cfg.epochs_per_batch):
        h, stop_logit = policy_net.context(states_D, states_depth)
        scores = policy_net.score(h, cand_desc)
        kl = factored_kl(stop_logit, scores, ostop, oscores, cand_mask)
        # Target-KL early stop: once the policy has already moved past the KL budget
        # (from earlier epochs of THIS update), stop before taking another step. This
        # caps the per-iteration KL directly and prevents runaway spikes.
        if target_kl is not None and kl.item() > target_kl:
            stopped_on_kl = 1
            break
        new_lp = factored_log_prob(stop_logit, scores, chosen, cand_mask)
        ratio = torch.exp(new_lp - old_lp)
        s1 = ratio * adv_t
        s2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_t
        policy_loss = (-torch.min(s1, s2) * w_t).sum() / w_sum
        loss = policy_loss + cfg.kl_coeff * kl

        if not torch.isfinite(loss):                 # never step on a non-finite update
            optimizer.zero_grad(set_to_none=True)
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()
        last_pl, last_kl = policy_loss.item(), kl.item()

    return {"policy_loss": last_pl, "kl": last_kl, "kl_early_stop": stopped_on_kl,
            "leaf_frac": leaf_frac, "root_leaf_adv": root_leaf_adv, "root_split_adv": root_split_adv}


# ----------------------------------------------------------------------------
# Evaluation (zero-shot: frozen policy, one deterministic rollout per dataset)
# ----------------------------------------------------------------------------
def _auc(y_true, probs, classes):
    try:
        if classes == 2:
            return roc_auc_score(y_true, probs[:, 1])
        yb = label_binarize(y_true, classes=list(range(classes)))
        return roc_auc_score(yb, probs, multi_class='ovr', average='macro')
    except ValueError:
        return np.nan  # e.g. a class absent from this test split


@torch.no_grad()
def evaluate(policy_net, datasets):
    policy_net.eval()
    # Eval always uses the full candidate set by default (best-split selection),
    # independent of the training-time max_candidates cap. Restore afterwards.
    eval_cap = getattr(cfg, "eval_max_candidates", None)
    saved_cap = getattr(cfg, "max_candidates", 96)
    cfg.max_candidates = eval_cap
    try:
        rows = []
        for ds in datasets:
            _, tree = rollout(policy_net, ds, deterministic=True)
            probs = predict(tree, ds.X_test, ds.y, ds.classes, ds.bins)
            rows.append({"dataset": ds.name, "split": ds.split,
                         "auc": _auc(ds.y_test, probs, ds.classes),
                         "leaves": count_leaves(tree)})
    finally:
        cfg.max_candidates = saved_cap
    policy_net.train()
    return rows


def _mean_auc(rows):
    aucs = [r["auc"] for r in rows if not np.isnan(r["auc"])]
    return float(np.mean(aucs)) if aucs else np.nan


# ----------------------------------------------------------------------------
# Meta-training loop
# ----------------------------------------------------------------------------
def meta_train(meta_sets, holdout_sets, n_iterations, eval_every, seed):
    rng = np.random.default_rng(seed)
    desc_dim = 6 + 2 * cfg.embed_dim
    policy_net = PolicyNetwork(cfg.embed_dim, desc_dim, hidden_dim=cfg.hidden_dim).to(DEVICE)  # ONCE
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.learning_rate)                      # ONCE
    ckpt_path = getattr(cfg, "meta_ckpt_path", "tabedt_meta.pt")
    eval_pool = holdout_sets if holdout_sets else meta_sets   # fall back to seen if no holdout

    wandb.init(project="TabEDT-cc18-meta",
               config={"max_depth": cfg.max_depth, "thresholds": cfg.thresholds,
                       "learning_rate": cfg.learning_rate, "group_size": cfg.group_size,
                       "beta": cfg.beta, "min_samples_split": cfg.min_samples_split,
                       "batch_size": cfg.batch_size, "embed_dim": cfg.embed_dim,
                       "max_candidates": getattr(cfg, "max_candidates", 96),
                       "n_meta_sets": len(meta_sets), "n_holdout_sets": len(holdout_sets),
                       "policy": "candidate_scorer_meta_layernorm"})

    best_auc, best_state = -1.0, None
    for it in range(n_iterations):
        ds = meta_sets[int(rng.integers(len(meta_sets)))]
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
               "ds_name": ds.name, "ds_classes": ds.classes, "ds_features": ds.features, **stats}

        if it % eval_every == 0:
            rows = evaluate(policy_net, eval_pool)
            m = _mean_auc(rows)
            log["zero_shot_auc"] = m
            log["zero_shot_leaves"] = float(np.mean([r["leaves"] for r in rows]))
            if not np.isnan(m) and m > best_auc:
                best_auc = m
                best_state = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
                torch.save(best_state, ckpt_path)
        wandb.log(log)

    if best_state is not None:
        policy_net.load_state_dict(best_state)
    wandb.finish()
    return policy_net


# ----------------------------------------------------------------------------
# Deployment: frozen policy + a labelled dataset -> tree, zero gradient updates
# ----------------------------------------------------------------------------
@torch.no_grad()
def build_tree_in_context(policy_net, X, y, classes):
    policy_net.eval()
    raise NotImplementedError("Build a MetaDataset-like object: set X, y, classes, features, "
                              "and emb (MotherNet embeddings of X, same convention as "
                              "extract_embeddings.py), then call .prepare() and "
                              "rollout(policy_net, ds, deterministic=True).")


# ----------------------------------------------------------------------------
def load_all_datasets():
    pairs = []
    for fname in sorted(os.listdir(cfg.data_dir)):
        if not fname.endswith('.npz'):
            continue
        emb_path = os.path.join(cfg.embed_dir, fname.replace('.npz', '_emb.npz'))
        if not os.path.exists(emb_path):
            print(f"  [skip] no embeddings for {fname}")
            continue
        pairs.append(MetaDataset(os.path.join(cfg.data_dir, fname), emb_path).prepare())
    return pairs


def split_by_name(datasets, holdout_frac, seed):
    names = sorted({ds.name for ds in datasets})
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_hold = int(round(holdout_frac * len(names)))
    hold_names = set(names[:n_hold])
    meta = [ds for ds in datasets if ds.name not in hold_names]
    hold = [ds for ds in datasets if ds.name in hold_names]
    return meta, hold, hold_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--holdout-frac', type=float,
                        default=getattr(cfg, "holdout_frac", 0.2))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} (dataset, split) instances "
          f"across {len({d.name for d in datasets})} datasets.")

    meta_sets, holdout_sets, hold_names = split_by_name(datasets, args.holdout_frac, args.seed)
    print(f"Meta-train instances: {len(meta_sets)}  |  "
          f"Held-out instances: {len(holdout_sets)}  (datasets: {sorted(hold_names)})")

    policy = meta_train(meta_sets, holdout_sets, args.iterations, args.eval_every, args.seed)

    rows = evaluate(policy, holdout_sets if holdout_sets else meta_sets)
    df = pd.DataFrame(rows)
    out = cfg.results_file.replace('.csv', '') + "_meta_holdout.csv"
    df.to_csv(out, index=False)
    print(f"\nHeld-out zero-shot AUC (mean over instances): {_mean_auc(rows):.4f}")
    print(df.groupby('dataset')['auc'].agg(['mean', 'std']).to_string())
    print(f"Results saved to {out}")
