import numpy as np
import os
import glob

from preprocess_cfg import PreprocessCfg

try:
    script_dir = os.path.abspath(os.path.dirname(__file__))
except NameError:
    script_dir = os.getcwd()

MODE = 'train_data'
RAW_DATA_DIR = os.path.join(script_dir, MODE, "raw_data")
OUTPUT_DIR = os.path.join(script_dir, MODE)
OUTPUT_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_expert_data.npz")
ACTION_SCALE_PATH = os.path.join(OUTPUT_DIR, "action_scales.npy")
CLOCK_DIM = 0

# TODO: bug! eular angle (obs[1:4]) normalization should be done with sin/cos instead of mean/std, otherwise the discontinuity at ±π will cause huge spikes in normalized values and destabilize training. This is a critical issue that must be fixed before training. The current code is only a temporary workaround to get some results, but it is not a proper solution. The correct way is to convert angles to sin/cos representation before normalization, and convert back to angles after denormalization. This way we can avoid the discontinuity issue and have a more stable training process.
# TODO: MLP collect data and train again

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
    np.save(ACTION_SCALE_PATH, scale)
    return action_norm_target_clipped.astype(np.float32)

def main():
    # --- 1. Dynamic File Search ---
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory does not exist: {RAW_DATA_DIR}")
        return
    
    search_pattern = os.path.join(RAW_DATA_DIR, "*.npz")
    npz_files = glob.glob(search_pattern)
    npz_files.sort()
    if len(npz_files) == 0:
        print(f"Error: No .npz files found in {RAW_DATA_DIR}")
        return
    print(f"Found {len(npz_files)} files in {RAW_DATA_DIR}:")
    for f in npz_files:
        print(f" - {os.path.basename(f)}")
    print("-" * 30)
    
    # --- 2. Load and Merge Data ---
    raw_obs_list = []
    raw_act_list = []
    for file_path in npz_files:
        try:
            data = np.load(file_path)
            if 'obs' not in data or 'actions' not in data:
                print(f"Warning: Skipping {os.path.basename(file_path)} (missing 'obs' or 'actions' key)")
                continue
            raw_obs_list.append(data['obs'])
            raw_act_list.append(data['actions'])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not raw_obs_list:
        print("Error: No valid data loaded.")
        return
    print("Merging data...")
    full_raw_obs = np.concatenate(raw_obs_list, axis=0)
    full_raw_actions = np.concatenate(raw_act_list, axis=0)
    print(f"Total merged samples: {full_raw_obs.shape[0]}")

    # --- 3. Process Observations ---

    obs_dim = PreprocessCfg.OBS_MEAN.shape[0]
    act_dim = PreprocessCfg.DEFAULT_JOINT_POSITIONS.shape[0]

    idx_s_t2_end = obs_dim
    idx_s_t1_end = obs_dim * 2
    idx_s_t0_end = obs_dim * 3
    idx_a_t2_end = obs_dim * 3 + act_dim
    idx_a_t1_end = obs_dim * 3 + act_dim * 2

    raw_s_t2 = full_raw_obs[:, 0            : idx_s_t2_end]
    raw_s_t1 = full_raw_obs[:, idx_s_t2_end : idx_s_t1_end]
    raw_s_t0 = full_raw_obs[:, idx_s_t1_end : idx_s_t0_end]
    raw_a_t2 = full_raw_obs[:, idx_s_t0_end : idx_a_t2_end]
    raw_a_t1 = full_raw_obs[:, idx_a_t2_end : idx_a_t1_end]
    raw_phase_num = full_raw_obs[:, -1:]

    # 1. Normalize Observations (S)
    norm_s_t2 = preprocess_observations(raw_s_t2, PreprocessCfg.OBS_MEAN, PreprocessCfg.OBS_STD)
    norm_s_t1 = preprocess_observations(raw_s_t1, PreprocessCfg.OBS_MEAN, PreprocessCfg.OBS_STD)
    norm_s_t0 = preprocess_observations(raw_s_t0, PreprocessCfg.OBS_MEAN, PreprocessCfg.OBS_STD)

    # 2. Normalize History Actions (A_history)
    norm_a_t2 = preprocess_actions(
        raw_a_t2, 
        PreprocessCfg.DEFAULT_JOINT_POSITIONS, 
        PreprocessCfg.ACTION_MAX, 
        PreprocessCfg.ACTION_MIN
    )
    norm_a_t1 = preprocess_actions(
        raw_a_t1, 
        PreprocessCfg.DEFAULT_JOINT_POSITIONS, 
        PreprocessCfg.ACTION_MAX, 
        PreprocessCfg.ACTION_MIN
    )

    # 3. Normalize Target Actions (Label)
    norm_act_target = preprocess_actions(
        full_raw_actions,
        PreprocessCfg.DEFAULT_JOINT_POSITIONS,
        PreprocessCfg.ACTION_MAX,
        PreprocessCfg.ACTION_MIN
    )

    print("Reconstructing stacked vector...")
    final_obs_normalized = np.concatenate(
        [norm_s_t2, norm_s_t1, norm_s_t0, norm_a_t2, norm_a_t1, raw_phase_num], 
        axis=1
    )

    print(f"Final Normalized Input Shape: {final_obs_normalized.shape}")
    
    # --- 5. Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving processed data file to: {OUTPUT_DATA_PATH} ...")
    try:
        np.savez(
            OUTPUT_DATA_PATH,
            obs=final_obs_normalized,
            actions=norm_act_target
        )
        print("\n" + "="*30)
        print(f"Processing Finished!")
        print(f"Input Directory:  {RAW_DATA_DIR}")
        print(f"Output File:      {OUTPUT_DATA_PATH}")
        print(f"Total Files Used: {len(npz_files)}")
        print("="*30)
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()