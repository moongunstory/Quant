"""directional — 롱숏 비율을 신호가 정하게 하는 '영속 규칙'(PLAN 2026-07-11 ②).

문제: 모든 알파를 매일 demean 해 순노출을 0(5:5)으로 강제하면, 신호가 진짜로 한
방향에 확신이 있어도 그 정보가 죽는다. 하지만 아무 신호나 순노출을 열면 통제 안 되는
시장 베타가 생긴다(특히 rank 신호는 항상 전종목 양수라 상수 롱 편향만 주입).

해법(영속 구조, 개별 알파를 손으로 바꾸지 않음):
  자격 규칙 — 다음 둘을 모두 만족할 때만 그 알파의 neutralization 을 "partial" 로
  자동 승격한다:
    (1) 알파의 패밀리 ∈ policy["eligible_families"]  (예: 펀딩·오더북 임밸런스 —
        raw 값의 시장 평균이 경제적으로 의미 있는 계열),
    (2) 원신호(neutralization 전)가 실제로 부호를 가짐 = raw 패널에 음수 존재.
        rank(→(0,1]) / zscore(→합 0) 로 감싼 신호는 순노출이 상수/0 이라 여기서
        자동 탈락한다(유동적이지 않은 상수 편향을 막는 핵심 가드).
  승격되면 순노출은 포트폴리오 market_neutrality 밴드(net_exposure_limit)가 유일한
  통제기가 된다. 장세 판별/예측 없음 — 신호 자율.

이 규칙은 영속이다: 새 알파가 자격 패밀리에 signed 신호로 들어오면 자동 승격되고,
알파가 빠지면 자동으로 사라지며, rank 로 감싼 알파는 영원히 신경 안 써도 된다.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from src.config.backtest_settings import SETTINGS


def load_policy(path=None):
    """data/directional_policy.json 로드. 없으면 None(규칙 비활성)."""
    p = Path(path) if path else SETTINGS.data_dir / "directional_policy.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def is_enabled(policy, cfg_override=None):
    """규칙 on/off. 포트폴리오 config 의 directional(True/False)이 policy 를 덮어씀.
    None 이면 policy["enabled"](기본 False)를 따른다."""
    if cfg_override is not None:
        return bool(cfg_override)
    return bool(policy and policy.get("enabled", False))


def is_signed(raw_panel):
    """원신호 패널에 음수가 하나라도 있으면 True(=순노출이 의미 있는 signed 신호).
    rank/zscore/전종목 양수 신호는 False → 방향성 승격 대상이 아님."""
    if raw_panel is None:
        return False
    arr = raw_panel.to_numpy()
    import numpy as np
    return bool(np.nanmin(arr) < 0.0)


def resolve_spec(spec, family, policy, signed):
    """자격을 통과하면 neutralization="market" 인 spec 을 "partial"(neut_beta=policy.beta)
    로 승격해 반환. 그 외에는 원본 그대로. AlphaSpec 은 frozen 이라 replace 로 새 인스턴스."""
    if not policy:
        return spec
    eligible = family in policy.get("eligible_families", [])
    if eligible and signed and spec.neutralization == "market":
        beta = float(policy.get("beta", 0.5))
        return dataclasses.replace(spec, neutralization="partial", neut_beta=beta)
    return spec


def has_market_neutrality(risk_pipeline):
    """리스크 파이프라인에 '활성화된' market_neutrality 단계가 있는지 — 방향성 on 일 때
    순노출 밴드가 반드시 걸려야 하므로 배선 검사에 쓴다."""
    for item in risk_pipeline or []:
        if item.get("type") == "market_neutrality" and item.get("enabled", True):
            return True
    return False
