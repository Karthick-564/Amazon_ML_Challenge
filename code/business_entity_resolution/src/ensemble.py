"""Shared CalibratedMetaEnsemble definition for serialization and inference."""

from __future__ import annotations

import numpy as np


class CalibratedMetaEnsemble:
    """Ensemble of LightGBM and CatBoost Meta-Learner with Isotonic Probability Calibration."""

    def __init__(self, lgb_model, cat_model, iso_lgb, iso_cat, weight_lgb: float = 0.5):
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.iso_lgb = iso_lgb
        self.iso_cat = iso_cat
        self.weight_lgb = weight_lgb
        self.weight_cat = 1.0 - weight_lgb

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_lgb = self.lgb_model.predict_proba(X)[:, 1]
        raw_cat = self.cat_model.predict_proba(X)[:, 1]

        cal_lgb = self.iso_lgb.predict(raw_lgb)
        cal_cat = self.iso_cat.predict(raw_cat)

        cal_lgb = np.clip(cal_lgb, 0.0, 1.0)
        cal_cat = np.clip(cal_cat, 0.0, 1.0)

        p1 = self.weight_lgb * cal_lgb + self.weight_cat * cal_cat
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
