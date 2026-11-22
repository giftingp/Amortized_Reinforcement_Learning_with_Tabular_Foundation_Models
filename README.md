# Amortized Reinforcement Learning with Tabular Foundation Models

## Files

All scripts except those in `gradtree/` import each other by module name, so keep
them in one folder.

| File | Role |
|---|---|
| `config_TabEDT.py` | hyperparameters; depth from `TABEDT_MAX_DEPTH` |
| `meta_tabedt_cc18.py` | policy network, rollout, GRPO update, CC-18 loading and `evaluate()` |
| `meta_tabedt_synthetic.py` | meta-training on the MotherNet prior (entry point) |
| `mothernet.py` | MotherNet model, used as the frozen encoder |
| `ticl_prior_hook.py` | samples synthetic tasks from ticl's prior |
| `preprocess_small_cc18.py` | five stratified 50/50 splits (seeds 42–46), NaN → 0 |
| `extract_embeddings.py` | MotherNet embeddings of each split |
| `compare_cart.py` | zero-shot evaluation of checkpoints against matched CART (G) |
| `run_train_array.bash` | SLURM launcher with the thesis settings |
| `gradtree/` | GradTree benchmark (self-contained, see its README) |

## Requirements

- Python 3.13 with torch 2.13, numpy, pandas, scikit-learn, scipy and wandb. The
  ticl package does not import under torch 2.15, which dropped `Optional` from
  `torch.nn.modules.transformer`.
- `ticl` from the Healthy-AI mother-tree repository (`pip install -e .` from its
  root; it also needs gpytorch, einops, entmax, graphviz, interpret and mlflow).
  Use the `mothernet.py` in this folder, since ticl's own MotherNet has no
  `get_embeddings`.
- MotherNet checkpoint
  `mn_b64_E1000_n8192_continue_02_11_2026_09_55_33_epoch_320.cpkt` and its prior
  config `prior_config_1.pkl` (`ckpt[3]['prior']`), neither of which is
  included. ticl's default `ticl.model_configs.get_prior_config()` has the same
  classification settings as the thesis runs logged.
- `cc_test_datasets_multiclass_full.pkl` with the CC-18 datasets, not included.

## Run order

```bash
# 1. splits (set PKL_PATH in the script)        -> ./cc18_preprocessed/
python preprocess_small_cc18.py
# 2. embeddings (set CHECKPOINT_PATH)            -> ./cc18_embeddings/
python extract_embeddings.py
# 3. meta-training, one array of 10 seeds per depth
sbatch --export=ALL,DEPTH=3,MOTHERNET_CKPT=/path/to/mothernet.cpkt run_train_array.bash
#    or a single run without SLURM
TABEDT_MAX_DEPTH=3 python meta_tabedt_synthetic.py --reward brier --max-candidates 128 \
    --iterations 40000 --eval-every 100 --checkpoint-every 500 --seed 0 --tag d3_rerun \
    --checkpoint /path/to/mothernet.cpkt --prior-config prior_config_1.pkl --prior-device cuda
# 4. zero-shot evaluation of chosen snapshots against matched CART (G)
TABEDT_MAX_DEPTH=3 python compare_cart.py --reward brier \
    --checkpoints "tabedt_synthetic_brier_seed*_d3_rerun_iter25000.pt"
```

## Thesis runs, for reference

From the W&B files of `synth_brier_seed9_d5_final`:

- **Command.** `--reward brier --max-depth 5 --seed 9 --iterations 40000
  --checkpoint-every 500 --eval-every 100 --val-every 100 --val-tasks 32
  --max-candidates 128 --tag d5_final --prior-device cuda`.
- **Hyperparameters.** As in the thesis table (group size 16, learning rate
  3e-4, KL 0.01, target KL 0.05, n_min 5, K = 20, β = 2, M = 128, skip 1e-3),
  except 32 validation tasks (table says 100) and a checkpoint interval of 500
  (table says 1,000). Check one run per depth before correcting the table.
- **Environment.** Python 3.13.5 on aarch64, torch 2.13.0+cu126, ticl 0.1.0,
  scikit-learn 1.9.0, numpy 2.5.2, pandas 2.3.3, wandb 0.28.2, one GH200 and 8
  CPU cores per run.
