from __future__ import annotations

import numpy as np

ACTION_SCALES_PATH = "bc/train_data/action_scales.npy"

def load_action_scales() -> np.ndarray:
    scales = np.load(ACTION_SCALES_PATH)
    return scales