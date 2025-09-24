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
        "sma": {"param_name": "windows", "type": "range_multi_int", "n_windows": {"min": 0, "max": 20}, "range": {"min": 5, "max": 500}, "step": 5},
        "ema": {"param_name": "windows", "type": "range_multi_int", "n_windows": {"min": 0, "max": 20}, "range": {"min": 5, "max": 500}, "step": 5},
        "rsi": {"param_name": "window", "type": "range_int", "range": {"min": 7, "max": 100}, "step": 3},
        "bbands": {"param_name": "window", "type": "range_int", "range": {"min": 10, "max": 200}, "step": 5},
        "atr": {"param_name": "window", "type": "range_int", "range": {"min": 7, "max": 200}, "step": 5},
        "stoch": {"param_name": "window", "type": "range_int", "range": {"min": 7, "max": 100}, "step": 3}
    },
    "funding": {
        "funding_ma": {"param_name": "window", "type": "range_int", "range": {"min": 24, "max": 500}, "step": 12},
        "funding_z": {"param_name": "window", "type": "range_int", "range": {"min": 24, "max": 500}, "step": 12}
    },
    "index": {
        "index_ma": {"param_name": "window", "type": "range_int", "range": {"min": 12, "max": 500}, "step": 6},
        "index_z": {"param_name": "window", "type": "range_int", "range": {"min": 12, "max": 500}, "step": 6}
    },
    "dune": {
        "window_features": {
            "param_name": "windows",
            "type": "range_multi_int",
            "n_windows": {"min": 0, "max": 20},
            "range": {"min": 3, "max": 200},
            "step": 2,
            "base_cols": ["netflow"],
            "feature_types": ["ma", "momentum", "zscore"]
        }
    }
}


# RL 하이퍼파라미터 튜닝 범위
rl_hparams = {
    "actor_lr":  {"type": "float", "min": 5e-5, "max": 5e-4, "log": True},
    "critic_lr": {"type": "float", "min": 5e-5, "max": 5e-4, "log": True},
    "gamma": {"type": "float", "min": 0.9, "max": 0.999},
    "tau": {"type": "float", "min": 0.02, "max": 0.1},
    "alpha": {"type": "float", "min": 0.1, "max": 0.8},
    "clip_grad": {"type": "float", "min": 1.0, "max": 3.0},
    "action_threshold_close": {"type": "float", "min": 0.2, "max": 0.4},
    "min_hold_bars": {"type": "int", "min": 8, "max": 15},
    "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "lstm_layers": {"type": "categorical", "choices": [1, 2]},
}


def get_max_required_window(tunable_feats: dict, fixed_feats: list = None) -> int:
    """
    피처 정의에서 사용되는 최대 윈도우 크기를 계산.
    - 튜닝형 피처의 range.max
    - 고정형 피처 중 'ichimoku'는 별도로 52를 부여
    """
    max_window = 0

    # 튜닝형 피처에서 최대 window 추출
    for feat in tunable_feats.values():
        if isinstance(feat, dict):
            if "range" in feat:
                max_window = max(max_window, feat["range"]["max"])
            elif "window_features" in feat and "range" in feat["window_features"]:
                max_window = max(max_window, feat["window_features"]["range"]["max"])

    # 고정형 피처에서 ichimoku 고려 (선행 스팬 52, 후행 26 포함 → 안전하게 56)
    if fixed_feats and "ichimoku" in fixed_feats:
        max_window = max(max_window, 56)

    return max_window


def get_representative_config(tunable_feats: dict) -> dict:
    rep_config = {}

    for group, feats in tunable_feats.items():
        rep_config[group] = {}

        for feat_name, cfg in feats.items():
            if "range" in cfg:
                min_val = cfg["range"]["min"]
                max_val = cfg["range"]["max"]
                step = cfg["step"]
                rep_config[group][feat_name] = list(range(min_val, max_val + 1, step))

            elif "window_features" in cfg:
                inner = cfg["window_features"]
                rep_config[group]["windows"] = list(range(inner["range"]["min"], inner["range"]["max"] + 1, inner["step"]))
                rep_config[group]["base_cols"] = inner["base_cols"]
                rep_config[group]["feature_types"] = inner["feature_types"]

                # dune용 Boolean flags 자동 삽입
                for col in inner["base_cols"]:
                    rep_config[group][f"use_{col}_for_window"] = True
                for ftype in inner["feature_types"]:
                    rep_config[group][f"use_dune_{ftype}"] = True

        # dune simple 피처들도 기본 포함
        if group == "dune":
            rep_config[group]["use_dune_netflow"] = True
            rep_config[group]["use_dune_inflow_outflow_ratio"] = True
            rep_config[group]["use_dune_deposit_withdraw_ratio"] = True

    return rep_config