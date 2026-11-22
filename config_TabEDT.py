import os
import torch

# Depth from the environment, as in the thesis runs (TABEDT_MAX_DEPTH=3, 4 or 5).
max_depth = int(os.environ.get("TABEDT_MAX_DEPTH", 5))
thresholds = 20
hidden_dim = 256
learning_rate = 3e-4
epochs = 200 
group_size = 16 
epochs_per_batch = 4 
clip_epsilon = 0.2
gamma = 1
kl_coeff = 0.01
beta = 2.0 
curiosity_coeff = 0.1
min_samples_split = 5
batch_size = 32
max_candidates = None      # TRAINING cap on candidate splits per node (None = all). Sweep e.g. 64/96/128.
eval_max_candidates = None # evaluation always uses this (None = all splits); independent of training cap
grpo_std_skip = 1e-3       # skip GRPO groups whose within-group return std < this (no signal)
target_kl = 0.05           # stops the epoch loop once per-update KL exceeds it (caps KL spikes)
embed_dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = "./cc18_preprocessed"
embed_dir = "./cc18_embeddings"
results_file = "TabEDT_cc18_results.csv"