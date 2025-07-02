import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    TIMEFRAMES,
    PPO_CONFIG,
)
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.imitation.train_imitation import (
    validate_data,
    align_timeframes,
    generate_mtf_features,
    calculate_tp_sl_hits_optimized,
    generate_rewards,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValueDataset(Dataset):
    """Dataset for value head pretraining."""

    def __init__(self, mtf_features: dict, rewards: np.ndarray):
        self.mtf_features = {tf: torch.FloatTensor(features) for tf, features in mtf_features.items()}
        self.rewards = torch.FloatTensor(rewards)

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        features_dict = {tf: feats[idx] for tf, feats in self.mtf_features.items()}
        return features_dict, self.rewards[idx]


def load_dataset(direction: str):
    """Load MTF dataset and reward targets for the given direction."""
    data_path = TRAIN_PICKLE_PATHS[direction]
    logger.info(f"📁 데이터 로드: {data_path}")
    raw = pd.read_pickle(data_path)

    entry_tf, eval_tf = "15min", "5min"
    df_entry = validate_data(raw[entry_tf].copy(), f"{direction}-entry")
    df_eval = validate_data(raw[eval_tf].copy(), f"{direction}-eval")

    df_entry, df_eval = align_timeframes(df_entry, df_eval)
    mtf_features_df = generate_mtf_features(raw)

    min_len = min([features.shape[0] for features in mtf_features_df.values()])
    min_len = min(min_len, len(df_entry))

    df_entry = df_entry.iloc[:min_len].copy()
    tp_hit, sl_hit = calculate_tp_sl_hits_optimized(df_entry, df_eval, direction)
    rewards = generate_rewards(tp_hit, sl_hit)

    mtf_features_array = {}
    input_dims = {}
    for tf, feat_df in mtf_features_df.items():
        feat_df = feat_df.iloc[:min_len]
        X = feat_df.values
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        num_windows = feat_df.shape[0]
        num_features = int(feat_df.shape[1] / PPO_CONFIG["seq_len"])
        X = X.reshape(num_windows, PPO_CONFIG["seq_len"], num_features)
        mtf_features_array[tf] = X
        input_dims[tf] = num_features

    dataset = ValueDataset(mtf_features_array, rewards)
    return dataset, input_dims


def train_value_head(direction: str):
    dataset, input_dims = load_dataset(direction)
    output_model_path = VALUE_PRETRAIN_OUTPUT_PATH[direction]
    imitation_path = PPO_IMITATION_MODEL_PATHS[direction]

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPOPolicyNetwork(timeframe_dims=input_dims, hidden_dim=PPO_CONFIG["hidden_dim"], action_dim=PPO_CONFIG["action_dim"]).to(device)
    model.load_model(imitation_path, allow_partial=True)

    for name, param in model.named_parameters():
        if 'value_head' not in name:
            param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PPO_CONFIG["learning_rate"])

    best_val_loss = float('inf')
    for epoch in range(PPO_CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for mtf_data, reward in train_loader:
            mtf_data = {tf: t.to(device) for tf, t in mtf_data.items()}
            reward = reward.to(device)
            optimizer.zero_grad()
            _, value = model(mtf_data)
            loss = criterion(value, reward)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mtf_data, reward in val_loader:
                mtf_data = {tf: t.to(device) for tf, t in mtf_data.items()}
                reward = reward.to(device)
                _, value = model(mtf_data)
                val_loss += criterion(value, reward).item()
        val_loss /= len(val_loader)

        logger.info(f"[{direction}] Epoch {epoch+1}/{PPO_CONFIG['epochs']} - TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)
            logger.info(f"✅ [{direction}] 모델 저장: {output_model_path}")

    return {
        "direction": direction,
        "val_loss": best_val_loss,
        "model_path": output_model_path,
    }


if __name__ == "__main__":
    results = []
    for direction in ["long", "short"]:
        result = train_value_head(direction)
        results.append(result)
    for res in results:
        logger.info(f"{res['direction'].upper()} → Best Val Loss: {res['val_loss']:.4f} | Saved: {res['model_path']}")
