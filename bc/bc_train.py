import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

from expert_data import ExpertDataset

ACTIVATION_FNS = {
    'nn.ELU': nn.ELU,
    'nn.ReLU': nn.ReLU,
    'nn.Tanh': nn.Tanh,
}

RESUME_TRAINING = True

class ActorBC(nn.Module):
    def __init__(self, obs_dim, act_dim, 
                 net_arch_pi=[64, 64], 
                 activation_fn_str='nn.ELU'):
        
        super(ActorBC, self).__init__()
        
        try:
            activation_fn = ACTIVATION_FNS[activation_fn_str]
        except KeyError:
            raise ValueError(f"Unknown activation function: {activation_fn_str}")

        # --- 1. Build "Body" (corresponding to mlp_extractor.policy_net) ---
        policy_net_layers = []
        last_dim = obs_dim
        
        for layer_dim in net_arch_pi:
            policy_net_layers.append(nn.Linear(last_dim, layer_dim))
            policy_net_layers.append(activation_fn())
            last_dim = layer_dim
        self.policy_net = nn.Sequential(*policy_net_layers)

        # --- 2. Build "Head" (corresponding to action_net) ---
        self.action_net = nn.Linear(last_dim, act_dim)
        print(f"--- ActorBC Model Initialized ---")
        print(f"Body (policy_net):\n{self.policy_net}")
        print(f"Head (action_net):\n{self.action_net}")
        print("---------------------------------")

    def forward(self, obs):
        features = self.policy_net(obs)
        mean = self.action_net(features)
        return mean
    

if __name__ == "__main__":
    # must match rl env cfg
    OBS_DIM = 34*3+11*2
    ACT_DIM = 11
    
    # must match rl yaml
    NET_ARCH_PI = [64, 64]
    ACTIVATION_FN = 'nn.ELU'
    SQUASH_OUTPUT = False
    
    # hyperparameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 500
    VALIDATION_SPLIT = 0.05
    RANDOM_SEED = 42

    try:
        script_dir = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        print("Warning: __file__ not defined. Saving to current working directory.")
        script_dir = os.getcwd()

    # file paths
    EXPERT_DATA_PATH = os.path.join(script_dir, "train_data/processed_expert_data.npz")
    BODY_WEIGHTS_PATH = os.path.join(script_dir, "model/bc_actor_body_weights.pth")
    HEAD_WEIGHTS_PATH = os.path.join(script_dir, "model/bc_actor_head_weights.pth")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using: {DEVICE}")

    # --- 1. Load, Split, and Setup DataLoaders ---
    try:
        data = np.load(EXPERT_DATA_PATH)
        all_obs = torch.from_numpy(data['obs']).float()
        all_actions = torch.from_numpy(data['actions']).float()
        print(f"Successfully loaded expert data: {all_obs.shape[0]} total samples")
    except FileNotFoundError:
        print(f"Error: Expert data file not found {EXPERT_DATA_PATH}")
        exit()
    except Exception as e:
        print(f"Error occurred while loading expert data: {e}")
        exit()

    obs_train, obs_val, act_train, act_val = train_test_split(
        all_obs, 
        all_actions, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_SEED
    )
    print(f"Data split: {len(obs_train)} train samples, {len(obs_val)} validation samples")

    # Create two datasets and two dataloaders
    train_dataset = ExpertDataset(obs_train, act_train)
    val_dataset = ExpertDataset(obs_val, act_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 1. Model setup (no change)
    model = ActorBC(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        net_arch_pi=NET_ARCH_PI,
        activation_fn_str=ACTIVATION_FN
    ).to(DEVICE)
    
    if RESUME_TRAINING:
            print("Checking for existing weights to resume training...")
            if os.path.exists(BODY_WEIGHTS_PATH) and os.path.exists(HEAD_WEIGHTS_PATH):
                try:
                    body_sd = torch.load(BODY_WEIGHTS_PATH, map_location=DEVICE)
                    head_sd = torch.load(HEAD_WEIGHTS_PATH, map_location=DEVICE)
                    model.policy_net.load_state_dict(body_sd)
                    model.action_net.load_state_dict(head_sd)
                    
                    print(f"Successfully loaded weights from:\n  - {BODY_WEIGHTS_PATH}\n  - {HEAD_WEIGHTS_PATH}")
                except Exception as e:
                    print(f"Error loading weights: {e}")
                    print("Starting training from scratch.")
            else:
                print(f"RESUME_TRAINING is True, but weights not found.")
                print("Starting training from scratch.")
    else:
        print("RESUME_TRAINING is False. Starting fresh.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 2. Training loop
    print(f"--- Start Behavior Cloning ({EPOCHS} Epochs) ---")
    
    for epoch in range(EPOCHS):
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(DEVICE)
            act_batch = act_batch.to(DEVICE)
            
            # Forward pass
            pred_act_mean = model(obs_batch)
            loss = criterion(pred_act_mean, act_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(DEVICE)
                act_batch = act_batch.to(DEVICE)
                
                pred_act_mean = model(obs_batch)
                val_loss = criterion(pred_act_mean, act_batch)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print("--- Training finish ---")

    # 3. Save weights
    model.eval()

    # save old weights with _old suffix
    body_weights_old = BODY_WEIGHTS_PATH.replace('.pth', '_old.pth')
    head_weights_old = HEAD_WEIGHTS_PATH.replace('.pth', '_old.pth')

    print("Backing up old weights...")
    if os.path.exists(BODY_WEIGHTS_PATH):
        if os.path.exists(body_weights_old):
            os.remove(body_weights_old)
        os.rename(BODY_WEIGHTS_PATH, body_weights_old)
        print(f" -> Body weights backed up to: {body_weights_old}")
        
    if os.path.exists(HEAD_WEIGHTS_PATH):
        if os.path.exists(head_weights_old):
            os.remove(head_weights_old)
        os.rename(HEAD_WEIGHTS_PATH, head_weights_old)
        print(f" -> Head weights backed up to: {head_weights_old}")

    # save new weights
    try:
        print(f"Saving body weights: {BODY_WEIGHTS_PATH}")
        torch.save(model.policy_net.state_dict(), BODY_WEIGHTS_PATH)
        
        print(f"Saving head weights: {HEAD_WEIGHTS_PATH}")
        torch.save(model.action_net.state_dict(), HEAD_WEIGHTS_PATH)

        print("--- Weights saved successfully ---")

    except Exception as e:
        print(f"Error occurred while saving weights: {e}")