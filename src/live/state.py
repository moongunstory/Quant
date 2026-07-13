"""config — 라이브 봇의 동적 상태 설정 관리.

data/runtime/live/config.json 에 보관되어 텔레그램 봇과 라이브 사이클 실행기가
현재 실매매 모드, 실행 온/오프 상태, 포트폴리오 설정 경로 등을 공유합니다.
"""
from __future__ import annotations

import json
from pathlib import Path

# 모든 데이터 파일은 data 폴더 하위에 위치하게 함
CONFIG_PATH = Path("data/runtime/live/config.json")


def load_live_state() -> dict:
    """설정 파일을 읽고 딕셔너리로 반환합니다. 파일이 없으면 기본값을 생성하여 저장합니다."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "mode": "paper",
            "enabled": True,
            "config_path": "data/strategy/portfolio/config.json"
        }
        save_live_state(default)
        return default
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "mode": "paper",
            "enabled": True,
            "config_path": "data/strategy/portfolio/config.json"
        }


def save_live_state(cfg: dict):
    """지정한 설정을 config.json 파일에 저장합니다."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
