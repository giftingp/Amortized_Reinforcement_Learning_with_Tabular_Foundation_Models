"""
ticl_prior_hook.py
------------------
Faithful MotherNet prior stream for TabEDT: build ticl's BagPrior from the
checkpoint's own prior config (prior_config_*.pkl, i.e. ckpt[3]['prior']) and
sample real synthetic (X, y) tasks from it.

This mirrors ticl/dataloader.py (the construction site at lines ~88-101):
    gp_flexible  = ClassificationAdapterPrior(GPPrior(cfg['gp']),
                       num_features=cfg['num_features'], **cfg['classification'])
    mlp_flexible = ClassificationAdapterPrior(MLPPrior(cfg['mlp']),
                       num_features=cfg['num_features'], **cfg['classification'])
    prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                     prior_weights=...)

NOTE: confirm the prior_weights derivation and whether 'boolean'/'step_function'
are included by reading `sed -n '80,105p' ticl/dataloader.py`; adjust build_bag_prior
to match exactly if your dataloader differs.
"""
import pickle
import numpy as np
import torch

from ticl.priors import (ClassificationAdapterPrior, MLPPrior, GPPrior, BagPrior,
                         BooleanConjunctionPrior, StepFunctionPrior)


def load_prior_config(path="prior_config_1.pkl"):
    """Load ckpt[3]['prior'] (or the full ckpt[3] and pull ['prior'])."""
    with open(path, "rb") as f:
        cfg = pickle.load(f)
    return cfg["prior"] if "prior" in cfg else cfg


def build_bag_prior(prior_config):
    """Reconstruct ticl's prior EXACTLY as ticl/dataloader.get_dataloader does.

    For this checkpoint prior_type == 'prior_bag', which builds a 2-prior bag with
    FIXED weights {'mlp': 0.961, 'gp': 0.039} (no boolean / step_function).
    """
    prior_type = prior_config.get("prior_type", "prior_bag")
    nf = prior_config["num_features"]
    cls = prior_config["classification"]

    gp_flexible = ClassificationAdapterPrior(
        GPPrior(prior_config["gp"]), num_features=nf, **cls)
    mlp_flexible = ClassificationAdapterPrior(
        MLPPrior(prior_config["mlp"]), num_features=nf, **cls)

    if prior_type == "prior_bag":
        return BagPrior(base_priors={"gp": gp_flexible, "mlp": mlp_flexible},
                        prior_weights={"mlp": 0.961, "gp": 0.039})
    elif prior_type == "step_function":
        return StepFunctionPrior(prior_config["step_function"])
    elif prior_type == "boolean_only":
        return BooleanConjunctionPrior(hyperparameters=prior_config["boolean"])
    elif prior_type == "bag_boolean":
        boolean = BooleanConjunctionPrior(hyperparameters=prior_config["boolean"])
        return BagPrior(base_priors={"gp": gp_flexible, "mlp": mlp_flexible,
                                     "boolean": boolean},
                        prior_weights={"mlp": 0.9, "gp": 0.02, "boolean": 0.08})
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")


class TiclPriorSampler:
    """Wraps BagPrior.get_batch to yield single (X, y) tasks for TabEDT."""
    def __init__(self, prior_config_path="prior_config_1.pkl", device="cpu",
                 min_classes=2, class_accept_base=1.0, class_accept_slope=0.0,
                 balanced_frac=None, verbose=False):
        # Curriculum arguments passed by meta_tabedt_synthetic.py. The thesis runs
        # used the defaults (min_classes=2, accept_base=1.0, slope=0.0, no balanced
        # override), which leave the prior unchanged, so only those are supported.
        if class_accept_base != 1.0 or class_accept_slope != 0.0 or balanced_frac is not None:
            raise NotImplementedError(
                "class-acceptance curriculum and balanced mixing are not implemented; "
                "use class_accept_base=1.0, class_accept_slope=0.0, balanced_frac=None")
        self.cfg = load_prior_config(prior_config_path)
        self.prior = build_bag_prior(self.cfg)
        self.device = device
        self.min_classes = max(2, int(min_classes))
        self.n_samples = int(self.cfg.get("n_samples", 1152))
        self.num_features = int(self.cfg.get("num_features", 100))
        if verbose:
            print(f"[prior] classification config: {self.cfg.get('classification')}")
            print(f"[prior] curriculum: min_classes={min_classes} "
                  f"accept_base={class_accept_base} slope={class_accept_slope} "
                  f"| balanced_override={balanced_frac}")

    def sample(self, rng=None, max_tries=20):
        """Sample one synthetic (X, y) task with a VALID multiclass target.

        Uses a balanced context/eval split (single_eval_pos at the configured
        eval position): at any sensible split the adapter assigns class labels
        across the FULL sequence, so we take all rows as one labelled set.
        single_eval_pos = n-1 was the bug -- it left a 1-row eval segment and the
        adapter returned an all -100 (ignore) sentinel target -> 1 class.
        """
        ep = self.cfg.get("eval_positions") or [int(self.n_samples * 0.95)]
        sep = int(round(float(ep[0])))
        sep = max(2, min(self.n_samples - 2, sep))      # never the degenerate n-1

        for _ in range(max_tries):
            x, y, y_, info = self.prior.get_batch(
                batch_size=1, n_samples=self.n_samples, num_features=self.num_features,
                device=self.device, single_eval_pos=sep)
            X = x[:, 0, :].cpu().numpy().astype(np.float32)
            Y = y[:, 0].cpu().numpy()
            Y = np.rint(Y).astype(np.int64) if Y.dtype.kind == "f" else Y.astype(np.int64)

            # drop ignore-sentinel rows (-100) defensively, then relabel 0..C-1
            keep = Y != -100
            if keep.sum() < 10:
                continue
            X, Y = X[keep], Y[keep]
            uniq = np.unique(Y)
            if len(uniq) < self.min_classes:  # reject degenerate draws (default: 1 class)
                continue
            Y = np.searchsorted(uniq, Y).astype(np.int64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X, Y

        raise RuntimeError(f"prior produced no valid multiclass task in {max_tries} tries")

if __name__ == "__main__":
    s = TiclPriorSampler("prior_config_1.pkl", device="cpu")
    X, Y = s.sample()
    print("sampled task:", X.shape, "classes:", len(np.unique(Y)),
          "class counts:", np.bincount(Y))