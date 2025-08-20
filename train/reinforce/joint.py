from __future__ import annotations
import os

from .common import MODEL_DIR
from .manager import train_manager
from .worker  import train_worker_warmup, train_worker_with_manager

def train_joint(split: str = "train",
                cycles: int = 6,
                w_chunk: int = 150_000,
                m_chunk: int = 150_000,
                seed: int = 42):
    """
    교대(Co-Training):
      0) 워커 예열 → 1) 매니저 사전학습 → 2) [매니저 고정→워커업데이트] ↔ [매니저업데이트] 반복
    """
    # 0) Warmup Worker (휴리스틱 goal)
    w_path = train_worker_warmup(split=split, steps=max(w_chunk, 300_000), seed=seed,
                                 save_path=os.path.join(MODEL_DIR, "worker_stage1.zip"))

    # 1) Pretrain Manager
    m_path = train_manager(split=split, steps=max(m_chunk, 400_000), seed=seed,
                           save_path=os.path.join(MODEL_DIR, "manager_stage1.zip"))

    # 2) Alternate updates
    for c in range(cycles):
        print(f"[JOINT] Cycle {c+1}/{cycles} - Worker update (Manager frozen)")
        w_path = train_worker_with_manager(m_path, split=split, steps=w_chunk, seed=seed,
                                           save_path=os.path.join(MODEL_DIR, f"worker_joint_{c+1}.zip"))
        print(f"[JOINT] Cycle {c+1}/{cycles} - Manager update")
        m_path = train_manager(split=split, steps=m_chunk, seed=seed,
                               save_path=os.path.join(MODEL_DIR, f"manager_joint_{c+1}.zip"))
    print("[JOINT] Done.")
    return w_path, m_path

def run_all():
    """간단 실행: 교대 학습 파이프라인 전체."""
    train_joint()
