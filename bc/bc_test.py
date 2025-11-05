import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 固定批量大小；其餘維度由檔案自動決定
BATCH_SIZE = 256
NET_ARCH_PI = [64, 64]
ACTIVATION_FN = 'nn.ELU'

try:
    SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    print("Warning: __file__ not defined. Using current working directory.")
    SCRIPT_DIR = os.getcwd()

EXPERT_DATA_PATH   = os.path.join(SCRIPT_DIR, "expert_data_test.npz")
BODY_WEIGHTS_PATH  = os.path.join(SCRIPT_DIR, "bc_actor_body_weights.pth")
HEAD_WEIGHTS_PATH  = os.path.join(SCRIPT_DIR, "bc_actor_head_weights.pth")
EVAL_SAVE_PATH     = os.path.join(SCRIPT_DIR, "bc_eval_results.npz")

# -----------------------------
# Model
# -----------------------------
ACTIVATION_FNS = {
    'nn.ELU': nn.ELU,
    'nn.ReLU': nn.ReLU,
    'nn.Tanh': nn.Tanh,
}

class ActorBC(nn.Module):
    def __init__(self, obs_dim, act_dim, net_arch_pi=[64, 64], activation_fn_str='nn.ELU'):
        super().__init__()
        if activation_fn_str not in ACTIVATION_FNS:
            raise ValueError(f"Unknown activation function: {activation_fn_str}")
        activation_fn = ACTIVATION_FNS[activation_fn_str]

        layers = []
        last = obs_dim
        for h in net_arch_pi:
            layers.append(nn.Linear(last, h))
            layers.append(activation_fn())
            last = h
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(last, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.policy_net(obs))

# -----------------------------
# Dataset
# -----------------------------
class ExpertDataset(Dataset):
    def __init__(self, obs: np.ndarray, acts: np.ndarray):
        self.obs = torch.from_numpy(obs).float()
        self.acts = torch.from_numpy(acts).float()
        assert self.obs.shape[0] == self.acts.shape[0], "Sample count mismatch."

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        mse_vec = ((pred - target) ** 2).mean(dim=0)  # per-dim
        mae_vec = (pred - target).abs().mean(dim=0)   # per-dim
        mse = mse_vec.mean()
        mae = mae_vec.mean()
    return mse.item(), mae.item(), mse_vec.cpu().numpy(), mae_vec.cpu().numpy()

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) 僅讀取現有資料（不產生 dummy）
    if not os.path.exists(EXPERT_DATA_PATH):
        raise FileNotFoundError(f"Expert data not found: {EXPERT_DATA_PATH}")
    data = np.load(EXPERT_DATA_PATH)
    obs = data["obs"]
    acts = data["actions"]

    # 依檔案自動取得維度
    obs_dim = int(obs.shape[1])
    act_dim = int(acts.shape[1])
    print(f"Loaded expert_data.npz -> obs_dim={obs_dim}, act_dim={act_dim}, samples={obs.shape[0]}")

    dataset = ExpertDataset(obs, acts)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2) 建立模型（維度以檔案為準）
    model = ActorBC(
        obs_dim=obs_dim,
        act_dim=act_dim,
        net_arch_pi=NET_ARCH_PI,
        activation_fn_str=ACTIVATION_FN
    ).to(device)

    # 3) 嚴格載入權重（缺任一就報錯；shape 不符也報錯）
    if not os.path.exists(BODY_WEIGHTS_PATH):
        raise FileNotFoundError(f"Body weights not found: {BODY_WEIGHTS_PATH}")
    if not os.path.exists(HEAD_WEIGHTS_PATH):
        raise FileNotFoundError(f"Head weights not found: {HEAD_WEIGHTS_PATH}")

    body_sd = torch.load(BODY_WEIGHTS_PATH, map_location=device)
    head_sd = torch.load(HEAD_WEIGHTS_PATH, map_location=device)

    # 嚴格檢查
    model.policy_net.load_state_dict(body_sd, strict=True)
    model.action_net.load_state_dict(head_sd, strict=True)
    print("Weights loaded successfully.")

    # 4) 推論與指標
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for ob, ac in loader:
            ob = ob.to(device)
            pred = model(ob)
            preds.append(pred.cpu())
            gts.append(ac)

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)

    mse, mae, mse_vec, mae_vec = compute_metrics(preds, gts)

    print("=== Evaluation Summary ===")
    print(f"Samples: {len(dataset)} | ActDim: {act_dim}")
    print(f"Overall MSE: {mse:.6f}")
    print(f"Overall MAE: {mae:.6f}")
    print("Per-dimension MSE:", np.array2string(mse_vec, precision=6, separator=', '))
    print("Per-dimension MAE:", np.array2string(mae_vec, precision=6, separator=', '))

    # 5) 儲存推論結果（便於後續分析/畫圖）
    np.savez(
        EVAL_SAVE_PATH,
        obs=obs.astype(np.float32),
        gt_actions=acts.astype(np.float32),
        pred_actions=preds.numpy().astype(np.float32),
        mse=mse,
        mae=mae,
        mse_vec=mse_vec.astype(np.float32),
        mae_vec=mae_vec.astype(np.float32),
    )
    print(f"Saved evaluation pack to: {EVAL_SAVE_PATH}")

if __name__ == "__main__":
    main()
