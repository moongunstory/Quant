# train/hpo/core/feature_registry.py

# 고정형 피처: 사용 여부만 결정 (True / False)
fixed_features = {
    "ohlcv": [
        "log_return",
        "vwap",
        "heikin_ashi",
        "ichimoku",
        "candlestick",
        "fibonacci"
    ],
    "funding": [
        "funding_sign"
    ],
    "index": [
        "pct_change"
    ],
    "dune": [
        "netflow",
        "inflow_outflow_ratio",
        "deposit_withdraw_ratio"
    ]
}

# 튜닝형 피처: 사용 여부 + 파라미터(범위) 튜닝
# type: 'range_int' -> 범위 내에서 정수 1개 선택
# type: 'range_multi_int' -> 범위 내에서 정수 0~N개 선택
tunable_features = {
    "ohlcv": {
        "sma": {"param_name": "windows", "type": "range_multi_int", "n_windows": {"min": 0, "max": 3}, "range": {"min": 5, "max": 100}},
        "ema": {"param_name": "windows", "type": "range_multi_int", "n_windows": {"min": 0, "max": 3}, "range": {"min": 5, "max": 100}},
        "rsi": {"param_name": "window", "type": "range_int", "range": {"min": 7, "max": 28}},
        "bbands": {"param_name": "window", "type": "range_int", "range": {"min": 10, "max": 50}},
        "atr": {"param_name": "window", "type": "range_int", "range": {"min": 7, "max": 50}},
        "stoch": {"param_name": "window", "type": "range_int", "range": {"min": 7, "max": 30}}
    },
    "funding": {
        "funding_ma": {"param_name": "window", "type": "range_int", "range": {"min": 24, "max": 200}},
        "funding_z": {"param_name": "window", "type": "range_int", "range": {"min": 24, "max": 200}}
    },
    "index": {
        "index_ma": {"param_name": "window", "type": "range_int", "range": {"min": 12, "max": 100}},
        "index_z": {"param_name": "window", "type": "range_int", "range": {"min": 12, "max": 100}}
    },
    "dune": {
        "window_features": {
            "param_name": "windows",
            "type": "range_multi_int",
            "n_windows": {"min": 0, "max": 4},
            "range": {"min": 3, "max": 30},
            "base_cols": ["netflow"],
            "feature_types": ["ma", "momentum", "zscore"]
        }
    }
}

# 모델 하이퍼파라미터 정의
model_hparams = {
    "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "weight_decay": {"type": "float", "min": 0.0, "max": 0.01, "log": False}
}

# RL 에이전트 하이퍼파라미터 정의
rl_hparams = {
    "hidden_dim": {"type": "categorical", "choices": [64, 128, 256]},
    "actor_lr": {"type": "float", "min": 1e-5, "max": 1e-3, "log": True},
    "critic_lr": {"type": "float", "min": 1e-5, "max": 1e-3, "log": True},
    "gamma": {"type": "float", "min": 0.95, "max": 0.999, "log": False},
    "tau": {"type": "float", "min": 0.001, "max": 0.01, "log": False},
    "alpha": {"type": "float", "min": 0.1, "max": 0.5, "log": False}
}


# 유틸: trial에서 모델 파라미터 추출
def suggest_model_params(trial):
    params = {}
    for name, cfg in model_hparams.items():
        if cfg["type"] == "float":
            params[name] = trial.suggest_float(
                name, cfg["min"], cfg["max"], log=cfg.get("log", False)
            )
        elif cfg["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
    return params