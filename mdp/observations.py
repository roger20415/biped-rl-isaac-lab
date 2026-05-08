from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import torch
from typing import Any

from isaaclab.envs.mdp import root_quat_w
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
else:
    ManagerBasedRLEnv = Any

from isaaclab_tasks.manager_based.biped_rl.training_config import TrainingConfig

GetterFn = Callable[[ManagerBasedRLEnv, SceneEntityCfg], torch.Tensor]

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to Euler angles (Roll, Pitch, Yaw).
        Expects quaternion in [w, x, y, z] format (Isaac Sim standard).
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (X-axis)
        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr, cosr)

        # Pitch (Y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        # Yaw (Z-axis)
        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny, cosy)

        return torch.stack((roll, pitch, yaw), dim=1)

def get_norm_euler_W_xyz(env: ManagerBasedRLEnv,
                         asset_cfg: SceneEntityCfg,
                         mean: list[float] | torch.Tensor,
                         std:  list[float] | torch.Tensor) -> torch.Tensor:
    quat: torch.Tensor = root_quat_w(env, asset_cfg)# (N, 4)
    euler: torch.Tensor = quat_to_euler_xyz(quat)# (N, 3)
    mean_t = torch.as_tensor(mean, dtype=euler.dtype, device=euler.device).view(1, -1)# (1,3)
    std_t  = torch.as_tensor(std, dtype=euler.dtype, device=euler.device).view(1, -1)# (1,3)

    if std_t.requires_grad:
        std_t = std_t.clone()
    std_t.masked_fill_(std_t < 1e-8, 1.0)

    delta = torch.atan2(torch.sin(euler - mean_t), torch.cos(euler - mean_t))
    return delta / std_t

def get_norm_vector(env: ManagerBasedRLEnv,
                    asset_cfg: SceneEntityCfg,
                    getter: GetterFn,
                    mean: list[float] | torch.Tensor,
                    std:  list[float]  | torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    x = getter(env, asset_cfg)
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1)
    std_t  = torch.as_tensor(std,  dtype=x.dtype, device=x.device).view(1, -1)

    if std_t.requires_grad:
        std_t = std_t.clone()
    std_t.masked_fill_(std_t < eps, 1.0)
    return (x - mean_t) / std_t

def has_foot_contact(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.0014) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    z = asset.data.body_pos_w[:, body_ids, 2]
    contact = torch.where(z < threshold, 1.0, -1.0)
    return contact

def get_past_state(env: ManagerBasedRLEnv, step_back: int) -> torch.Tensor:
    if not hasattr(env, "state_history"):
        return torch.zeros((env.num_envs, TrainingConfig.STATE_DIM), device=env.device)
    idx = -step_back
    return env.state_history[:, idx, :].clone()

def get_past_action(env: ManagerBasedRLEnv, step_back: int) -> torch.Tensor:
    if not hasattr(env, "action_history"):
        return torch.zeros((env.num_envs, TrainingConfig.ACTION_DIM), device=env.device, dtype=torch.float32)
    
    idx = -step_back 
    return env.action_history[:, idx, :].clone()

def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    # TODO code review here
    # 這裡直接使用環境內建的步數紀錄作為時間步 t。
    # 當環境 reset 時，episode_length_buf 會自動歸零，因此我們不需要寫額外的 reset 邏輯！
    t = env.episode_length_buf.float()
    
    # 預設產生與時間步相同維度的零張量 (對應 Phase 1 的輸出 0.0)
    phase_out = torch.zeros_like(t)
    
    # === 邏輯對應 ===
    # Phase 1: t < 10 (保持為 0.0，不需要額外操作)
    
    # Phase 2: 10 <= t < 60 
    # 原邏輯：transition_alpha 每次加 0.02，達到 1.0 時進入 Phase 3 (相當於經過 50 步)
    phase_2_mask = (t >= 10) & (t < 60)
    phase_out[phase_2_mask] = 1.0 / 6.0
    
    # Phase 3: t >= 60
    # 原邏輯：每 70 步 (3.5 / 0.05) 是一個 stage，滿 4 個 stage (280 步) 循環一次
    phase_3_mask = (t >= 60)
    
    # 扣除前兩個 Phase 用掉的 60 步，取得進入 Phase 3 後的相對時間
    t_p3 = t[phase_3_mask] - 60 
    
    # 計算 continuous_stage (0, 1, 2, 3) 循環
    continuous_stage = (t_p3 % 280) // 70
    
    # 套用 Phase 3 的輸出公式
    phase_out[phase_3_mask] = (2.0 + continuous_stage) / 6.0
    
    # 將形狀從 (num_envs,) 轉換為 (num_envs, 1) 以符合 Observation 的維度需求
    return phase_out.unsqueeze(-1)