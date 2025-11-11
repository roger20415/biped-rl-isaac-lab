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

GetterFn = Callable[[ManagerBasedRLEnv, SceneEntityCfg], torch.Tensor]

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
        # quat: (num_envs, 4) format [x, y, z, w]
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

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
    contact = (z < threshold).float()
    return contact


def obs_affine(
    env: Any,
    inner_func,
    inner_params=None,
    mean=0.0,
    std=1.0
):
    x = inner_func(env, inner_params)
    std_t  = torch.as_tensor(std,  dtype=x.dtype, device=x.device)
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    return x * std_t + mean_t