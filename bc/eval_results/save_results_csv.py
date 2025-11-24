import numpy as np
import pandas as pd
import os

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EVAL_SAVE_PATH = os.path.join(SCRIPT_DIR, "../eval_results/bc_eval_results.npz")

ACTION_COMPARASION_CSV_PATH = os.path.join(SCRIPT_DIR, "../eval_results/action_comparison.csv")
FULL_DATA_CSV_PATH = os.path.join(SCRIPT_DIR, "../eval_results/full_data_export.csv")

def load_and_save_data():
    
    if not os.path.exists(EVAL_SAVE_PATH):
        print(f"FAIL: can't found {EVAL_SAVE_PATH}")
        return

    print(f"loading data: {EVAL_SAVE_PATH}")
    data = np.load(EVAL_SAVE_PATH)
    
    gt_actions = data['gt_actions']
    pred_actions = data['pred_actions']
    obs = data['obs']
    
    N, act_dim = gt_actions.shape
    obs_dim = obs.shape[1]
    
    print(f"Successfully loaded {N} samples, Action dimension: {act_dim}, Obs dimension: {obs_dim}")

    # =================================================================
    # Output file 1: Action pairwise comparison
    # Format: [GT_action1, Pred_action1, GT_action2, Pred_action2, ...]
    # =================================================================
    print(f"\nGenerating file 1: {ACTION_COMPARASION_CSV_PATH} (Action pairwise comparison)")
    
    # 1. Create an empty list to store columns to combine
    columns_to_combine = []
    
    # 2. Create column names and data one by one
    for i in range(act_dim):
        # Ground truth column
        col_gt_name = f'GT_Action_{i+1}'
        columns_to_combine.append(pd.Series(gt_actions[:, i], name=col_gt_name))
        
        # Predicted value column
        col_pred_name = f'Pred_Action_{i+1}'
        columns_to_combine.append(pd.Series(pred_actions[:, i], name=col_pred_name))

    # 3. Use concat to horizontally combine all Series into a single DataFrame
    df_actions = pd.concat(columns_to_combine, axis=1)
    
    # 4. Save as CSV
    df_actions.to_csv(ACTION_COMPARASION_CSV_PATH, index=False)
    print(f"File 1 saved successfully.")

    # =================================================================
    # Output file 2: Obs, GT Action, Pred Action (each row corresponds)
    # Format: [Obs1, Obs2, ..., GT_Action1, ..., Pred_Action1, ...]
    # =================================================================
    print(f"\nGenerating file 2: {FULL_DATA_CSV_PATH} (Full data export)")

    # 1. Create DataFrame for Obs
    obs_cols = [f'Obs_{i+1}' for i in range(obs_dim)]
    df_obs = pd.DataFrame(obs, columns=obs_cols)

    # 2. Create DataFrame for GT Actions
    gt_cols = [f'GT_Action_{i+1}' for i in range(act_dim)]
    df_gt_actions = pd.DataFrame(gt_actions, columns=gt_cols)

    # 3. Create DataFrame for Pred Actions
    pred_cols = [f'Pred_Action_{i+1}' for i in range(act_dim)]
    df_pred_actions = pd.DataFrame(pred_actions, columns=pred_cols)
    
    # 4. Horizontally combine all DataFrames
    df_full = pd.concat([df_obs, df_gt_actions, df_pred_actions], axis=1)

    # 5. Save as CSV
    df_full.to_csv(FULL_DATA_CSV_PATH, index=False)
    print(f"File 2 saved successfully.")

if __name__ == "__main__":
    load_and_save_data()