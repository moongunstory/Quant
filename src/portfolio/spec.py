"""config — portfolio.json 로더/스키마.

포트폴리오 하나 = JSON 파일 하나. 어떤 알파를 / 어떻게 결합하고 / 어떤 리스크
파이프라인을 씌울지를 한곳에 선언한다(코드 수정 없이 실험).

스키마:
  name          포트폴리오 이름.
  alphas        알파 이름 리스트. 빈 리스트 [] = data/strategy/alphas 전체 사용.
  weighting     {"method": "equal"|"inverse_vol", "params": {...}}.
  risk_pipeline [{"type","enabled","params"}, ...]  (실행순서 = 리스트순서).
                type 는 risk.RISK_REGISTRY 키. enabled=false 면 건너뜀(리포트엔 표시).

알 수 없는 최상위 키는 조용히 무시하지 않고 에러(오타 조기 발견).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_ALLOWED_TOP = {"name", "alphas", "weighting", "risk_pipeline", "description",
                "rebalance_band", "selection", "directional"}

VALID_SELECTION = ("low_correlation", "manual")


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    alphas: list = field(default_factory=list)
    weighting: dict = field(default_factory=lambda: {"method": "equal", "params": {}})
    risk_pipeline: list = field(default_factory=list)
    # selection(선택): {"method": "low_correlation"|"manual", "params": {...}}.
    # 없거나 빈 dict 면 selection 생략 = specs 전부 사용(하위호환).
    # low_correlation params 예: {"max_corr_threshold":0.5,"max_per_family":2,"top_n":8,"min_fitness":0.0}
    selection: dict = field(default_factory=dict)
    # directional(선택): 방향성 규칙 on/off 오버라이드. None 이면 directional_policy.json
    # 의 enabled 를 따름. True/False 로 이 포트폴리오만 강제 on/off(5:5 강제 버전과 A/B).
    directional: bool | None = None
    # 라이브 드리프트 밴드(Phase 3-B): 현재 대비 총 L1 드리프트가 이 값 미만이면
    # 리밸런싱 보류. 0 = 매 사이클 리밸런싱. 백테스트엔 영향 없음(라이브 orders 만).
    rebalance_band: float = 0.0
    description: str = ""

    def validate(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("PortfolioSpec.name 은 비어있지 않은 문자열")
        if not isinstance(self.alphas, list):
            raise ValueError("alphas 는 리스트여야 함")
        m = (self.weighting or {}).get("method")
        if m not in ("equal", "inverse_vol", "skill"):
            raise ValueError(f"weighting.method 는 equal|inverse_vol|skill, got {m!r}")
        if not isinstance(self.risk_pipeline, list):
            raise ValueError("risk_pipeline 은 리스트여야 함")
        for i, item in enumerate(self.risk_pipeline):
            if "type" not in item:
                raise ValueError(f"risk_pipeline[{i}] 에 'type' 없음")
        if self.selection:
            if not isinstance(self.selection, dict):
                raise ValueError("selection 은 dict 여야 함")
            sm = self.selection.get("method")
            if sm not in VALID_SELECTION:
                raise ValueError(f"selection.method 는 {VALID_SELECTION}, got {sm!r}")
        return self

    @property
    def weighting_method(self):
        return (self.weighting or {}).get("method", "equal")

    @property
    def weighting_params(self):
        return (self.weighting or {}).get("params", {}) or {}


def load_portfolio_spec(path) -> PortfolioSpec:
    p = Path(path)
    d = json.loads(p.read_text(encoding="utf-8"))
    unknown = set(d) - _ALLOWED_TOP
    if unknown:
        raise ValueError(f"PortfolioSpec: 알 수 없는 키 {sorted(unknown)}")
    return PortfolioSpec(
        name=d.get("name", p.stem),
        alphas=d.get("alphas", []),
        weighting=d.get("weighting", {"method": "equal", "params": {}}),
        risk_pipeline=d.get("risk_pipeline", []),
        selection=d.get("selection", {}),
        directional=d.get("directional", None),
        rebalance_band=float(d.get("rebalance_band", 0.0)),
        description=d.get("description", ""),
    ).validate()
