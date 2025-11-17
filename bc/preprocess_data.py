import numpy as np
import os

from preprocess_cfg import PreprocessCfg

try:
    script_dir = os.path.abspath(os.path.dirname(__file__))
except NameError:
    script_dir = os.getcwd()

DATA_DIR = os.path.join(script_dir, "data")
INPUT_RAW_DATA_PATH = os.path.join(DATA_DIR, "expert_data.npz")
OUTPUT_TRAIN_DATA_PATH = os.path.join(DATA_DIR, "expert_data_train.npz")


def preprocess_observations(obs_data: np.ndarray, 
                       obs_mean: np.ndarray, 
                       obs_std: np.ndarray) -> np.ndarray:

    if obs_data.shape[1] != obs_mean.shape[0] or obs_data.shape[1] != obs_std.shape[0]:
        raise ValueError(
            f"Obs dimension mismatch! Data has {obs_data.shape[1]} dimensions, "
            f"but provided μ has {obs_mean.shape[0]} dimensions, and σ has {obs_std.shape[0]} dimensions."
        )
    
    # Prevent division by zero
    obs_std[obs_std < 1e-8] = 1.0
    print(f"[Obs] Applying (obs - μ) / σ ...")
    normalized_obs = (obs_data - obs_mean) / obs_std
    print(f"[Obs] Normalization complete.")
    return normalized_obs.astype(np.float32)


def preprocess_actions(act_data: np.ndarray, 
                       default_pose: np.ndarray, 
                       action_max: np.ndarray,
                       action_min: np.ndarray,
                       ) -> np.ndarray:
    if not (act_data.shape[1] == default_pose.shape[0] == 
            action_max.shape[0] == action_min.shape[0]):
        raise ValueError(
            f"Action dimension mismatch! "
            f"Data ({act_data.shape[1]}), "
            f"Default Pose ({default_pose.shape[0]}), "
            f"Action Max ({action_max.shape[0]}), "
            f"Action Min ({action_min.shape[0]}) "
            f"dimensions must all be the same."
        )
    
    # --- 1. Compute Action Scale ---
    print(f"[Act] Using provided ACTION MAX/MIN to compute A_delta MAX/MIN...")
    action_delta_max = action_max - default_pose
    action_delta_min = action_min - default_pose
    
    print(f"[Act] Computing symmetric scale = max(abs(delta_max), abs(delta_min))...")
    scale = np.maximum(np.abs(action_delta_max), np.abs(action_delta_min))

    # Prevent division by zero
    scale[scale < 1e-8] = 1.0
    
    # --- 2. Apply Scale ---
    print(f"[Act] Using computed scale to transform back...")
    action_delta = act_data - default_pose
    
    # Inverse transform A_norm_target = A_delta / scale
    action_norm_target = action_delta / scale
    
    print(f"[Act] Clipping action data to [-1.0, 1.0]...")
    # Clip actions to [-1, 1]
    over_limit = np.sum(np.abs(action_norm_target) > 1.0)
    total_elements = action_norm_target.size
    print(f"[Act] {over_limit / total_elements * 100:.5f}% of action values were clipped.")
    action_norm_target_clipped = np.clip(action_norm_target, -1.0, 1.0)
    
    print(f"[Act] Normalization complete.")
    np.save("./data/action_scales.npy", scale)
    return action_norm_target_clipped.astype(np.float32)

def main():
    # --- 1. Load Raw Data ---
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        data = np.load(INPUT_RAW_DATA_PATH)
        raw_obs = data['obs']
        raw_actions = data['actions']
    except FileNotFoundError:
        print(f"Cannot find the raw data file: {INPUT_RAW_DATA_PATH}")
        return
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        return

    print(f"Successfully loaded {raw_obs.shape[0]} data points.")
    
    # --- 2. Process Observations ---
    normalized_obs = preprocess_observations(
        raw_obs,
        PreprocessCfg.OBS_MEAN,
        PreprocessCfg.OBS_STD
    )
    
    # --- 3. Process Actions ---
    normalized_actions = preprocess_actions(
        raw_actions,
        PreprocessCfg.DEFAULT_JOINT_POSITIONS,
        PreprocessCfg.ACTION_MAX,
        PreprocessCfg.ACTION_MIN,
    )
    
    # --- 4. Saving processed training file ---
    print(f"Saving processed training file: {OUTPUT_TRAIN_DATA_PATH} ...")
    try:
        np.savez(
            OUTPUT_TRAIN_DATA_PATH,
            obs=normalized_obs,
            actions=normalized_actions
        )
        print("\n" + "="*30)
        print(f"Saving {OUTPUT_TRAIN_DATA_PATH}")
        print("="*30)
    except Exception as e:
        print(f"Error saving the final training file: {e}")

if __name__ == "__main__":
    main()