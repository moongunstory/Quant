import os
import sys
import pickle
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

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'lgbm', 'lgbm_short.pkl')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'data', 'models', 'ppo_staging', 'short_imitation.pt')


def load_dataset(path: str):
    obj = pickle.load(open(path, 'rb'))
    if isinstance(obj, dict):
        X = obj.get('X')
        y = obj.get('y')
    elif isinstance(obj, pd.DataFrame):
        df = obj
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    else:
        raise ValueError('Unsupported data format')

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values

    X = np.array(X, dtype=np.float32)
    if X.ndim == 2:
        X = X.reshape(len(X), 1, X.shape[1])

    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train(lr: float = 1e-4, batch_size: int = 1024, epochs: int = 5, hidden_dim: int = 128, device: str = None, save_model: bool = True):
    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    X, y = load_dataset(DATA_PATH)
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
            if batch_y.dim() == 1 and not batch_y.dtype.is_floating_point:
                loss = F.cross_entropy(logits, batch_y.long())
            else:
                if batch_y.dim() == 1:
                    target = torch.stack([1 - batch_y, batch_y], dim=-1)
                else:
                    target = batch_y
                loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()
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
