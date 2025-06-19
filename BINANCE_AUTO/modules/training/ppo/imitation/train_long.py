import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.training.ppo.core.model import PPOPolicyNetwork

CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'label', 'train_long.csv')
LGBM_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'lgbm', 'lgbm_long.pkl')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'ppo_staging', 'long_imitation.pt')


def load_dataset(csv_path: str, model_path: str):
    df = pd.read_csv(csv_path)
    exclude_keys = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
    feature_cols = [c for c in df.columns if not any(key in c.lower() for key in exclude_keys)]

    X_np = df[feature_cols].values.astype(np.float32)
    lgbm_model = joblib.load(model_path)
    prob = lgbm_model.predict_proba(X_np)[:, 1]
    y_np = np.stack([1 - prob, prob], axis=1).astype(np.float32)

    X_tensor = torch.tensor(X_np.reshape(len(X_np), 1, X_np.shape[1]), dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    return X_tensor, y_tensor


def train(lr: float = 1e-4, batch_size: int = 1024, epochs: int = 5, hidden_dim: int = 128, device: str = None, save_model: bool = True):
    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    X, y = load_dataset(CSV_PATH, LGBM_PATH)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[-1]
    model = PPOPolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim, action_dim=2).to(device)
    for p in model.value_head.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_X)
            log_prob = F.log_softmax(logits, dim=-1)
            loss = -(batch_y * log_prob).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}')

    if save_model:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"[✅ 모델 저장 완료] {SAVE_PATH}")
    return avg_loss


if __name__ == '__main__':
    train()
