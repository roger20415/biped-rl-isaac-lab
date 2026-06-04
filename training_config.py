class TrainingConfig:
    STATE_DIM: int = 34
    ACTION_DIM: int = 11
    # obs = STATE_DIM*3+ACTION_DIM*2+1
    # obs: [S(t-2), S(t-1), S(t), A(t-2), A(t-1)]

    FREEZE_ACTOR = False

    MLP_HEAD_WEIGHTS_PATH = "./bc/model/bc_actor_head_weights.pth"
    MLP_BODY_WEIGHTS_PATH = "./bc/model/bc_actor_body_weights.pth"

    SAVE_FREQUENCY = 200
    USE_CHECKPOINT_AGENT_CFG: bool = False