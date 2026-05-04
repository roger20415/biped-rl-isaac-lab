class TrainingConfig:
    STATE_DIM: int = 34
    ACTION_DIM: int = 11
    # obs = STATE_DIM*3+ACTION_DIM*2+1
    # obs: [S(t-2), S(t-1), S(t), A(t-2), A(t-1)]