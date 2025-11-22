# model/regression/ensemble/blend.py
"""
Simple ensemble blending (weighted average of multiple models).
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np


class SimpleEnsemble:
    """
    Simple weighted average of multiple models.

    Example:
        models = {'lgbm': lgbm_model, 'xgb': xgb_model}
        weights = {'lgbm': 0.6, 'xgb': 0.4}
        ensemble = SimpleEnsemble(models, weights)
        prediction = ensemble.predict(X)
    """

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize ensemble.

        Args:
            models: Dict mapping model name to trained model
            weights: Dict mapping model name to weight (default: equal weights)
        """
        self.models = models

        # Default: equal weights
        if weights is None:
            n = len(models)
            weights = {name: 1.0 / n for name in models.keys()}

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        self.weights = {name: w / total_weight for name, w in weights.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble prediction (weighted average).

        Args:
            X: Feature matrix

        Returns:
            Weighted average prediction
        """
        predictions = []

        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights.get(name, 0.0)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.

        Args:
            X: Feature matrix

        Returns:
            Dict mapping model name to predictions
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions
