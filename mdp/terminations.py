# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Tuple, Sequence, TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@torch.jit.script_if_tracing
def _as_long_tensor(x: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure body/joint indices are a 1-D LongTensor on the right device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long).view(-1)
    else:
        return torch.tensor(list(x), device=device, dtype=torch.long).view(-1)


def base_height_out_of_manual_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    bounds: Tuple[float, float],
) -> torch.Tensor:
    """
    Terminate when the specified body/bodies' world Z height are outside [low, high].

    Parameters
    ----------
    env : ManagerBasedRLEnv
        The vectorized RL environment.
    asset_cfg : SceneEntityCfg
        Scene entity config specifying which asset/body names to check.
        Example: SceneEntityCfg("robot", body_names=["base_link"])
    bounds : (low, high)
        Height bounds in meters.

    Returns
    -------
    torch.Tensor (num_envs,) bool
        True where termination should occur.
    """
    low, high = bounds
    asset = env.scene[asset_cfg.name]
    # Resolve body indices for all envs
    body_ids = _as_long_tensor(asset_cfg.body_ids, env.device)  # (num_bodies,)
    # World positions for all bodies: (num_envs, num_bodies, 3)
    # asset.data.body_pos_w shape is (num_envs, num_bodies_total, 3)
    z = asset.data.body_pos_w[:, body_ids, 2]  # (num_envs, num_bodies_selected)
    # If multiple bodies provided, use min/max across them
    z_min = torch.amin(z, dim=1)
    z_max = torch.amax(z, dim=1)
    return (z_min < low) | (z_max > high)


def state_is_invalid(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    root_pos_invalid = ~torch.isfinite(asset.data.root_pos_w).all(dim=-1)
    root_vel_invalid = ~torch.isfinite(asset.data.root_vel_w).all(dim=-1)
    joint_pos_invalid = ~torch.isfinite(asset.data.joint_pos).all(dim=-1)
    joint_vel_invalid = ~torch.isfinite(asset.data.joint_vel).all(dim=-1)
    return root_pos_invalid | root_vel_invalid | joint_pos_invalid | joint_vel_invalid