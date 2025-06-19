import os
import sys
import json
import optuna

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.training.ppo.imitation import train_long, train_short

HPO_DIR = os.path.join(PROJECT_ROOT, 'data', 'models', 'hpo')
os.makedirs(HPO_DIR, exist_ok=True)


def objective(trial, direction: str):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 256, 2048, step=256)
    epochs = trial.suggest_int('epochs', 3, 15)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)

    if direction == 'long':
        loss = train_long.train(lr=lr, batch_size=batch_size, epochs=epochs, hidden_dim=hidden_dim, save_model=False)
    else:
        loss = train_short.train(lr=lr, batch_size=batch_size, epochs=epochs, hidden_dim=hidden_dim, save_model=False)
    return loss


def run_hpo(direction: str = 'long', n_trials: int = 20):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, direction), n_trials=n_trials)

    print(f"[✅ HPO 완료] Best Loss: {study.best_value:.6f}")
    print(f"Best Params: {study.best_params}")

    result_path = os.path.join(HPO_DIR, f'ppo_{direction}_imitation_hpo.json')
    with open(result_path, 'w') as f:
        json.dump({'best_value': study.best_value, 'best_params': study.best_params}, f, indent=2)
    print(f"[📄 결과 저장] {result_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', choices=['long', 'short'], default='long')
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()

    run_hpo(args.direction, args.trials)
