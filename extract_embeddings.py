import os
import numpy as np
import torch
from mothernet import MotherNet

DATA_DIR = "./cc18_preprocessed"
OUT_DIR  = "./cc18_embeddings"
CHECKPOINT_PATH = "/mimer/NOBACKUP/groups/oovgen/philip/hypernet/ticl/models_diff/mn_b64_E1000_n8192_continue_02_11_2026_09_55_33_epoch_320.cpkt"
MAX_FEATURES = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUT_DIR, exist_ok=True)

model = MotherNet(
    n_out=2,
    emsize=512,
    nhead=4,
    nhid_factor=2,
    nlayers=12,
    n_features=MAX_FEATURES,
    y_encoder_layer=None,
)
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
if isinstance(checkpoint, tuple):
    state_dict = checkpoint[0]
else:
    state_dict = checkpoint

# Keep only encoder and transformer weights – ignore decoder
filtered_state = {
    k: v for k, v in state_dict.items()
    if k.startswith('encoder.') or k.startswith('transformer_encoder.')
}
model.load_state_dict(filtered_state, strict=False)
model.to(DEVICE)
model.eval()

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith('.npz'):
        continue

    data = np.load(os.path.join(DATA_DIR, fname))
    X_train = data['X_train']
    X_test  = data['X_test']
    metadata = {k: data[k] for k in data.files if k not in ['X_train', 'X_test']}

    def pad_features(X, max_feat=MAX_FEATURES):
        n, d = X.shape
        if d == max_feat:
            return X
        elif d < max_feat:
            padded = np.zeros((n, max_feat), dtype=X.dtype)
            padded[:, :d] = X
            return padded
        else:
            print(f"  Warning: {d} features > {max_feat}, truncating")
            return X[:, :max_feat]

    X_train_pad = pad_features(X_train)
    X_test_pad  = pad_features(X_test)

    with torch.no_grad():
        train_t = torch.tensor(X_train_pad, dtype=torch.float32).to(DEVICE)
        test_t  = torch.tensor(X_test_pad, dtype=torch.float32).to(DEVICE)
        emb_train = model.get_embeddings(train_t)
        emb_test  = model.get_embeddings(test_t)

    out_fname = fname.replace('.npz', '_emb.npz')
    out_path  = os.path.join(OUT_DIR, out_fname)
    np.savez_compressed(out_path,
                        emb_train=emb_train,
                        emb_test=emb_test,
                        **metadata)
    print(f"  -> {out_path}")