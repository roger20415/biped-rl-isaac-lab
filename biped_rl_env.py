import torch
from isaaclab.envs import ManagerBasedRLEnv
from pathlib import Path
from .biped_rl_env_cfg import BipedRlEnvCfg
from .training_config import TrainingConfig
from .bc.preprocess_cfg import PreprocessCfg
from . import mdp
import pandas as pd
import torch


class BipedRlEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: BipedRlEnvCfg, **kwargs):

        super().__init__(cfg, **kwargs)

        excel_path = "./obs_for_rl_test.csv" 
        try:
            df = pd.read_csv(excel_path, header=None, sep=",")
            
            self.dummy_obs_tensor = torch.tensor(df.values, dtype=torch.float32, device=self.device)
            self.dummy_obs_length = len(self.dummy_obs_tensor)
            self.dummy_step_idx = 0
            print(f"✅ [Debug] 成功讀取 Excel 觀測值，共 {self.dummy_obs_length} 筆資料。")
        except Exception as e:
            print(f"❌ [Debug] 無法讀取 Excel: {e}")
            self.dummy_obs_tensor = None

        self._state_obs_mean = torch.as_tensor(
            PreprocessCfg.OBS_MEAN,
            device=self.device,
            dtype=torch.float32,
        )

        self._state_obs_std = torch.as_tensor(
            PreprocessCfg.OBS_STD,
            device=self.device,
            dtype=torch.float32,
        )

        self._action_scales = torch.as_tensor(
            mdp.load_action_scales(),
            device=self.device,
            dtype=torch.float32,
        )

    def reset(self, env_ids: torch.Tensor | None = None, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        # 1. 先執行預設的 reset 來更新物理引擎狀態，並取得尚未_state_obs_meany 的 obs
        obs, extras = super().reset(env_ids, seed, options)
        
        # ... (這裡保留你原本 history reset 的邏輯，如果有的話) ...

        # ---------------------------------------------------------
        # [修改處：重置時注入第一筆 Excel 資料，並將其正規化]
        # ---------------------------------------------------------
        if self.dummy_obs_tensor is not None:
            self.dummy_step_idx = 0 # 重置時，將索引歸零
            dummy_obs = self.dummy_obs_tensor[self.dummy_step_idx] # 這是 Excel 裡的「未正規化」真實數值
            
            # [提示：為了讓模型看得懂，我們必須將 Excel 的數值正規化]
            normalized_dummy_obs = dummy_obs.clone()
            sd = self.cfg.state_dim
            ad = self.cfg.action_dim
            
            # 針對前 34*3 維的 State 進行正規化: (x - mean) / std
            normalized_dummy_obs[0:sd] = (dummy_obs[0:sd] - self._state_obs_mean) / self._state_obs_std
            normalized_dummy_obs[sd:2*sd] = (dummy_obs[sd:2*sd] - self._state_obs_mean) / self._state_obs_std
            normalized_dummy_obs[2*sd:3*sd] = (dummy_obs[2*sd:3*sd] - self._state_obs_mean) / self._state_obs_std
            
            # 針對後 11*2 維的 Action history 進行正規化: x / scale
            a_start = 3 * sd
            normalized_dummy_obs[a_start:a_start+ad] = dummy_obs[a_start:a_start+ad] / self._action_scales
            normalized_dummy_obs[a_start+ad:a_start+2*ad] = dummy_obs[a_start+ad:a_start+2*ad] / self._action_scales
            
            # 將 1D 的「已正規化」dummy_obs 擴展以符合 num_envs 的維度
            if isinstance(obs, dict) and "policy" in obs:
                obs["policy"][:] = normalized_dummy_obs.unsqueeze(0).expand(self.num_envs, -1)
            else:
                obs[:] = normalized_dummy_obs.unsqueeze(0).expand(self.num_envs, -1)
                
            print(f"[Debug] Reset 觸發，已注入第 0 筆 Excel 觀測值 (並已完成正規化轉換)")

        return obs, extras


    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment and inject the subsequent observations from the dataset.
        """
        # ---------------------------------------------------------
        # [修改處：印出未正規化的 Observation 與 Action]
        # ---------------------------------------------------------
        if self.dummy_obs_tensor is not None:
            print(f"\n{'='*40}")
            print(f"Step Index: {self.dummy_step_idx}")
            
            # dummy_obs 來自 Excel，本身就是未正規化的數值
            dummy_obs = self.dummy_obs_tensor[self.dummy_step_idx]
            # action 是模型輸出的，是已正規化的數值
            a_t = action[0]

            # [提示：擷取當前未正規化的 State S(t)]
            s_t_unnormalized = dummy_obs[2 * self.cfg.state_dim:3 * self.cfg.state_dim]
            
            # [提示：將模型輸出的 Action 反正規化，還原為真實物理量]
            a_t_denormalized = a_t * self._action_scales

            def _format_rows(values: list[float], row_size: int = 5, precision: int = 6) -> str:
                rows = []
                for i in range(0, len(values), row_size):
                    chunk = values[i : i + row_size]
                    rows.append(" ".join(f"{v:.{precision}f}" for v in chunk))
                return "\n".join(rows)

            print("Observation S(t) (Input) [Unnormalized]:")
            print(_format_rows(s_t_unnormalized.tolist(), row_size=5))
            
            print("\nAction (Output) [Unnormalized]:")
            print(_format_rows(a_t_denormalized.tolist(), row_size=5))
            print(f"{'='*40}\n")

        # 1. 執行底層 step
        obs, rewards, dones, truncated, extras = super().step(action)

        # ---------------------------------------------------------
        # [修改處：將 Excel 的資料正規化後，強制替換下一步的 obs]
        # ---------------------------------------------------------
        if self.dummy_obs_tensor is not None:
            # 準備下一筆資料的索引
            self.dummy_step_idx = (self.dummy_step_idx + 1) % self.dummy_obs_length
            dummy_obs = self.dummy_obs_tensor[self.dummy_step_idx]

            # [提示：再次執行正規化轉換，確保餵給 Policy 的是正規化後的數值]
            normalized_dummy_obs = dummy_obs.clone()
            sd = self.cfg.state_dim
            ad = self.cfg.action_dim
            
            # State 正規化
            normalized_dummy_obs[0:sd] = (dummy_obs[0:sd] - self._state_obs_mean) / self._state_obs_std
            normalized_dummy_obs[sd:2*sd] = (dummy_obs[sd:2*sd] - self._state_obs_mean) / self._state_obs_std
            normalized_dummy_obs[2*sd:3*sd] = (dummy_obs[2*sd:3*sd] - self._state_obs_mean) / self._state_obs_std
            
            # Action 正規化
            a_start = 3 * sd
            normalized_dummy_obs[a_start:a_start+ad] = dummy_obs[a_start:a_start+ad] / self._action_scales
            normalized_dummy_obs[a_start+ad:a_start+2*ad] = dummy_obs[a_start+ad:a_start+2*ad] / self._action_scales

            if isinstance(obs, dict) and "policy" in obs:
                obs["policy"][:] = normalized_dummy_obs.unsqueeze(0).expand(self.num_envs, -1)
            else:
                obs[:] = normalized_dummy_obs.unsqueeze(0).expand(self.num_envs, -1)

            # 強制擋掉 dones，避免物理引擎出界導致強制重置而打斷連續測試
            dones[:] = False
            truncated[:] = False

        return obs, rewards, dones, truncated, extras