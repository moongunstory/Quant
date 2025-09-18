# train/hpo/core/feature_selector.py

from optuna import Trial
from ai_binance.train.hpo.core.feature_registry import fixed_features, tunable_features


def select_features_from_trial(trial: Trial) -> dict:
    """
    Optuna trial에서 선택된 피처 + 튜닝형 파라미터 조합을 반환
    """
    selected_features = {}

    # 1. 고정형 피처 처리
    for group, features in fixed_features.items():
        selected_features[group] = {}
        for feat in features:
            # use_ohlcv_log_return, use_dune_netflow 등
            use = trial.suggest_categorical(f"use_{group}_{feat}", [True, False])
            if use:
                # compute 함수에서 `config.get("use_dune_netflow")` 처럼 사용될 것을 예상
                # 따라서 `selected_features["dune"]["use_dune_netflow"] = True` 형태로 저장
                selected_features[group][f"use_{group}_{feat}"] = True

    # 2. 튜닝형 피처 처리
    for group, features in tunable_features.items():
        if group not in selected_features:
            selected_features[group] = {}

        # Dune은 구조가 복잡해서 별도 처리
        if group == 'dune':
            cfg = features["window_features"]
            use = trial.suggest_categorical(f"use_{group}_window_features", [True, False])
            if not use:
                continue

            # 어떤 피처 타입(ma, momentum 등)을 사용할지 선택
            for f_type in cfg["feature_types"]:
                selected_features[group][f"use_{group}_{f_type}"] = trial.suggest_categorical(f"use_{group}_{f_type}", [True, False])

            # 어떤 기본 컬럼에 적용할지 선택
            for b_col in cfg["base_cols"]:
                selected_features[group][f"use_{group}_{b_col}_for_window"] = trial.suggest_categorical(f"use_{group}_{b_col}_for_window", [True, False])

            # 동적 범위에서 N개 윈도우 선택
            n_windows = trial.suggest_int(f"{group}_n_windows", cfg["n_windows"]["min"], cfg["n_windows"]["max"])
            selected_windows = []
            for i in range(n_windows):
                # 윈도우 값이 겹치지 않도록, 이전 값보다 큰 값만 선택
                min_val = selected_windows[i-1] + 1 if i > 0 else cfg["range"]["min"]
                if min_val >= cfg["range"]["max"]:
                    break
                w = trial.suggest_int(f"{group}_window_{i}", min_val, cfg["range"]["max"])
                selected_windows.append(w)
            
            selected_features[group][cfg["param_name"]] = sorted(list(set(selected_windows)))
            continue

        # 일반 튜닝 피처 처리
        for feat, cfg in features.items():
            use = trial.suggest_categorical(f"use_{group}_{feat}", [True, False])
            if not use:
                continue

            param_name = cfg["param_name"]
            feat_type = cfg.get("type", "range_int")
            param_value = None

            if feat_type == "range_int":
                param_value = trial.suggest_int(
                    f"{group}_{feat}_{param_name}", cfg["range"]["min"], cfg["range"]["max"]
                )
            
            elif feat_type == "range_multi_int":
                n_windows = trial.suggest_int(f"{group}_{feat}_n_windows", cfg["n_windows"]["min"], cfg["n_windows"]["max"])
                selected_windows = []
                for i in range(n_windows):
                    min_val = selected_windows[i-1] + 1 if i > 0 else cfg["range"]["min"]
                    if min_val >= cfg["range"]["max"]:
                        break
                    w = trial.suggest_int(f"{group}_{feat}_window_{i}", min_val, cfg["range"]["max"])
                    selected_windows.append(w)
                param_value = sorted(list(set(selected_windows)))

            if param_value is not None:
                selected_features[group][feat] = {
                    "use": True,
                    param_name: param_value
                }

    return selected_features