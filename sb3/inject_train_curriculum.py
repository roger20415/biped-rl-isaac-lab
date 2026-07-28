# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3 using automated curriculum and BC injection."""
"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 加入 argparse 參數設定
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# 附加 AppLauncher cli 參數
AppLauncher.add_app_launcher_args(parser)
# 解析參數
args_cli, hydra_args = parser.parse_known_args()
# 若開啟錄影，強制啟用攝影機
if args_cli.video:
    args_cli.enable_cameras = True

# 清空 sys.argv 供 Hydra 使用
sys.argv = [sys.argv[0]] + hydra_args

# 啟動 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# 覆寫 KeyboardInterrupt 的預設行為
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
import torch
from datetime import datetime

import omni
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.manager_based.biped_rl.training_config import TrainingConfig

# PLACEHOLDER: Extension template (do not remove this comment)

def _validate_state_dict_finite(state_dict: dict, label: str) -> None:
    """
    Validates if all tensors in a state dictionary are finite.
    """
    invalid_keys = []
    for key, value in state_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            invalid_keys.append(key)
    if invalid_keys:
        raise ValueError(f"{label} contains non-finite tensors: {invalid_keys[:5]}")

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    
    curriculum_schedule = [
        {"log_std": -4.5, "timesteps": 1_000_000},
        {"log_std": -4.7, "timesteps": 1_000_000},
        {"log_std": -5.0, "timesteps": 1_000_000}
    ]

    # 如果 seed = -1，隨機抽樣一個種子
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 覆蓋 CLI 傳入的設定
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # 設定環境種子
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 設定 Log 儲存目錄
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_Curriculum_BC")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = os.path.join(log_root_path, run_info)
    
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    
    # 移除設定檔中的 n_timesteps，統一由 curriculum_schedule 接管
    if "n_timesteps" in agent_cfg:
        agent_cfg.pop("n_timesteps")

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env_cfg.log_dir = log_dir

    # 建立 Isaac 環境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    is_norm_obs = norm_args.get("normalize_input", False)
    is_norm_reward = norm_args.get("normalize_value", False)
    clip_obs_val = norm_args.get("clip_obs", 100.0)
    gamma_val = agent_cfg.get("gamma", 0.99)

    # 處理 VecNormalize
    if norm_args and norm_args.get("normalize_input"):
        vec_norm_path = None
        if args_cli.checkpoint is not None:
            checkpoint_dir = os.path.dirname(args_cli.checkpoint)
            vec_norm_path = os.path.join(checkpoint_dir, "model_vecnormalize.pkl")

        if vec_norm_path and os.path.exists(vec_norm_path):
            print(f"\033[1;36m[INFO] Found saved VecNormalize at {vec_norm_path}, loading.\033[0m")
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_obs = is_norm_obs
            env.norm_reward = is_norm_reward
        else:
            print("\033[1;33m[WARNING] No saved VecNormalize found; creating a new VecNormalize.\033[0m")
            env = VecNormalize(
                env,
                training=True,
                norm_obs=is_norm_obs,
                norm_reward=is_norm_reward,
                clip_obs=clip_obs_val,
                gamma=gamma_val,
                clip_reward=np.inf,
            )

    # 建立基礎 PPO Agent
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    
    # 載入初始 Checkpoint (若有提供)
    if args_cli.checkpoint is not None:
        use_checkpoint_cfg = getattr(TrainingConfig, "USE_CHECKPOINT_AGENT_CFG", True)
        if use_checkpoint_cfg:          
            agent = agent.load(args_cli.checkpoint, env, tensorboard_log=log_dir, print_system_info=True)
        else:
            print("\n" + "-"*65)
            print("\033[1;33m[INFO] USE_CHECKPOINT_AGENT_CFG: False\033[0m")
            print("\033[1;33m[INFO] Using current global agent_cfg. Only loading weights from checkpoint...\033[0m")
            print("-"*65 + "\n")
            agent.set_parameters(args_cli.checkpoint, exact_match=False)

    # === 載入行為仿效 (BC) 權重至 Actor 網路 ===
    try:
        actor_body_net = agent.policy.mlp_extractor.policy_net
        actor_head_net = agent.policy.action_net

        body_weights = torch.load(TrainingConfig.MLP_BODY_WEIGHTS_PATH, map_location=agent.device)
        head_weights = torch.load(TrainingConfig.MLP_HEAD_WEIGHTS_PATH, map_location=agent.device)

        _validate_state_dict_finite(body_weights, "BC body weights")
        _validate_state_dict_finite(head_weights, "BC head weights")

        actor_body_net.load_state_dict(body_weights)
        actor_head_net.load_state_dict(head_weights)
        print("\033[1;32m[INFO] 成功注入 BC 權重至 Actor 網路！\033[0m")
    
    except AttributeError as e:
        print("==========================================================")
        print(f" FAILED: AttributeError: {e}")
        print("   This likely means your agent config YAML file configuration is incorrect.")
        print("   Please ensure your agent config YAML file is using 'separate networks':")
        print("   net_arch:")
        print("     pi: [..., ...]")
        print("     vf: [..., ...]")
        print("==========================================================")
        raise
    except Exception as e:
        print(f"--- [ERROR] Inject Behavior cloning pipeline failed: {e} ---")
        raise
    # ========================================================

    # === 自動化階段訓練迴圈 ===
    for phase_idx, phase in enumerate(curriculum_schedule):
        target_log_std = phase["log_std"]
        phase_timesteps = phase["timesteps"]
        
        print("\n" + "="*70)
        print(f"\033[1;32m[自動排程] 開始執行階段 {phase_idx + 1} / {len(curriculum_schedule)}\033[0m")
        print(f"\033[1;32m[自動排程] 目標 log_std_init: {target_log_std}\033[0m")
        print(f"\033[1;32m[自動排程] 訓練步數 (Timesteps): {phase_timesteps}\033[0m")
        print("="*70 + "\n")

        # 1. 根據階段強制覆寫 log_std
        with torch.no_grad():
            agent.policy.log_std.fill_(target_log_std)
        
        # 2. 確保 Actor 在每個階段都被正確凍結 (Critic Warm-up)
        if TrainingConfig.FREEZE_ACTOR:
            print("\033[1;33m[INFO] 確保 Actor 網路權重已被凍結 (Critic Warm-up)\033[0m")
            for name, param in agent.policy.named_parameters():
                if "action_net" in name or "policy_net" in name or "log_std" in name:
                    param.requires_grad = False
                elif "value_net" in name:
                    param.requires_grad = True

        # 3. 建立這個階段專屬的儲存子目錄，避免 Checkpoint 覆蓋混淆
        phase_dir = os.path.join(log_dir, f"phase_{phase_idx+1}_std_{target_log_std}")
        os.makedirs(phase_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=TrainingConfig.SAVE_FREQUENCY, 
            save_path=phase_dir, 
            name_prefix=f"model_p{phase_idx+1}", 
            verbose=2
        )
        callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]

        # 4. 執行訓練
        # 注意：只有第一個階段 reset_num_timesteps=True，後面的階段設為 False
        # 這樣 TensorBoard 的曲線才會從左到右連續畫下去，不會斷掉歸零
        reset_timesteps = (phase_idx == 0)
        
        with contextlib.suppress(KeyboardInterrupt):
            agent.learn(
                total_timesteps=phase_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=None,
                reset_num_timesteps=reset_timesteps
            )
            
        # 5. 儲存該階段完成後的最終權重與環境正規化參數
        agent.save(os.path.join(phase_dir, "model_final"))
        if isinstance(env, VecNormalize):
            env.save(os.path.join(phase_dir, "model_vecnormalize.pkl"))
            
        print(f"\033[1;36m[自動排程] 階段 {phase_idx + 1} 完成！已儲存至 {phase_dir}\033[0m")

    # 全局訓練結束
    print("\n\033[1;32m[自動排程] 所有自動化訓練階段皆已執行完畢！\033[0m")
    
    # 儲存最終的 Global 模型供未來使用
    agent.save(os.path.join(log_dir, "model"))
    if isinstance(env, VecNormalize):
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))

    # 關閉模擬器
    env.close()

if __name__ == "__main__":
    # 執行 main 函數
    main()
    # 關閉 sim app
    simulation_app.close()