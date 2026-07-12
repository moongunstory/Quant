"""AlphaSpec — 알파 하나의 정의(계약).

알파 1개 = JSON 파일 1개 (data/alphas/<name>.json).
이 모양만 안정적으로 유지되면 엔진/메트릭/검증은 자유롭게 바꿔도 된다.

필드:
  name           고유 이름 (= 파일명 stem과 일치해야 함)
  expression     수식 문자열. 예: "rank(neg(ts_delta(close, 5)))"
                 evaluate.py 가 field 패널들에 대해 안전하게 평가한다.
  freq           리밸런싱 주기. 반드시 "1d" (일 단위). coin의 D2 결정과 동일.
  neutralization "none" | "market". 스코어를 달러중립으로 만드는 방식.
                 "market" = 그날 코인 평균을 빼서 시장 방향성 제거.
  decay          가중치에 적용할 선형 감쇠 창 (0 또는 1이면 없음).
  delay          신호와 체결 사이 지연일수. 반드시 >=1 (당일 체결 금지 = 미래참조 방지).

의존성 최소화(판다스/넘파이 import 없음)로 라이브 레이어에서도 싸게 로드 가능.
알 수 없는 JSON 키는 조용히 무시하지 않고 에러(오타를 일찍 잡기 위해).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

VALID_NEUTRALIZATION = ("none", "market", "turnover_rank", "partial")

# 알파의 '판단(리밸런싱) 주기'. klines 1h 네이티브를 마스터 그리드로 두고, 각 알파는
# 자기 데이터 성격에 맞는 bar 로 신호를 갱신한다(예: funding_carry -> 8h, momentum -> 1d).
# delay 는 이 bar 단위로 해석된다(delay=1 = 1 bar 뒤 체결, 미래참조 방지).
VALID_BARS = ("1h", "2h", "4h", "6h", "8h", "12h", "1d")


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    expression: str
    freq: str = "1d"          # 정보용 라벨(하위호환). 실제 판단주기는 bar.
    bar: str = "1d"           # 판단/리밸런싱 주기 (VALID_BARS). Phase 3.
    neutralization: str = "none"
    neut_beta: float = 1.0    # "partial" 중립 강도(0=중립없음, 1=완전중립). 다른 모드에선 무시.
    decay: int = 0
    delay: int = 1            # bar 단위 지연(>=1, 미래참조 방지)
    cluster: str | None = None
    description: str = ""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """말이 안 되는 값이면 즉시 에러(fail loud)."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError("AlphaSpec.name 은 비어있지 않은 문자열이어야 함")
        if not self.expression or not isinstance(self.expression, str):
            raise ValueError(f"AlphaSpec[{self.name}].expression 이 비어있음")
        if self.bar not in VALID_BARS:
            raise ValueError(
                f"AlphaSpec[{self.name}].bar 는 {VALID_BARS} 중 하나여야 함, "
                f"got {self.bar!r}"
            )
        if self.neutralization not in VALID_NEUTRALIZATION:
            raise ValueError(
                f"AlphaSpec[{self.name}].neutralization 은 {VALID_NEUTRALIZATION} "
                f"중 하나여야 함, got {self.neutralization!r}"
            )
        if not (0.0 <= self.neut_beta <= 1.0):
            raise ValueError(
                f"AlphaSpec[{self.name}].neut_beta 는 0~1 이어야 함, got {self.neut_beta!r}"
            )
        if int(self.decay) != self.decay or self.decay < 0:
            raise ValueError(f"AlphaSpec[{self.name}].decay 는 0 이상 정수")
        if int(self.delay) != self.delay or self.delay < 1:
            raise ValueError(
                f"AlphaSpec[{self.name}].delay 는 1 이상 정수여야 함(미래참조 방지), "
                f"got {self.delay!r}"
            )

    @classmethod
    def from_dict(cls, d):
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"AlphaSpec: 알 수 없는 키 {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path):
        p = Path(path)
        spec = cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        if spec.name != p.stem:
            raise ValueError(
                f"AlphaSpec name {spec.name!r} 은 파일명 {p.stem!r} 과 일치해야 함"
            )
        return spec

    def to_dict(self):
        return asdict(self)

    def save(self, dir_path):
        p = Path(dir_path) / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        return p


def load_all(dir_path):
    """디렉토리 안 모든 alphas/*.json 로드 (이름순)."""
    d = Path(dir_path)
    if not d.exists():
        return []
    return [AlphaSpec.load(p) for p in sorted(d.glob("*.json"))]
