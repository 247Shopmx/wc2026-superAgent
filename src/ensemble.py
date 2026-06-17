"""
Ensemble Training Pipeline
Manages the 6-model ensemble architecture.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from src.models import DixonColesModel, MarkovChainModel

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Manages the 6-model ensemble architecture."""

    def __init__(self):
        self.models = {
            "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100),
            "lgbm": lgb.LGBMClassifier(n_estimators=100, verbose=-1),
            "rf": RandomForestClassifier(n_estimators=100, random_state=42),
            "lr": LogisticRegression(max_iter=1000)
        }
        self.dixon_coles = DixonColesModel()
        self.markov = MarkovChainModel()
        self.is_trained = False
        self.weights = {
            "xgb": 0.20,
            "lgbm": 0.20,
            "rf": 0.15,
            "lr": 0.15,
            "dixon": 0.15,
            "markov": 0.15
        }

    def train_ml_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train ML classifiers with calibration."""
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            # Note: In production, use proper train/calibration split
            model.fit(X_train, y_train)
            self.models[name] = model
        self.is_trained = True

    def predict_ensemble(self, features: pd.DataFrame, lambda_home: float, lambda_away: float) -> Dict[str, float]:
        """Generate weighted ensemble probabilities."""
        if not self.is_trained:
            raise RuntimeError("Models not trained yet.")

        ml_probs = {}
        for name, model in self.models.items():
            preds = model.predict_proba(features)
            ml_probs[name] = preds[0] 

        # Mathematical Models
        p_home_dc, p_draw_dc, p_away_dc = self.dixon_coles.get_match_outcome_probs(lambda_home, lambda_away)
        dc_probs = np.array([p_home_dc, p_draw_dc, p_away_dc])

        # Markov (Simplified integration)
        p_goal_home = self.markov.simulate_possession_outcome()
        p_goal_away = self.markov.simulate_possession_outcome() * 0.9
        total_m = p_goal_home + p_goal_away + 0.2
        markov_probs = np.array([p_goal_home/total_m, 0.2/total_m, p_goal_away/total_m])

        # Weighted Average
        final_probs = np.zeros(3)
        final_probs += self.weights["xgb"] * ml_probs["xgb"]
        final_probs += self.weights["lgbm"] * ml_probs["lgbm"]
        final_probs += self.weights["rf"] * ml_probs["rf"]
        final_probs += self.weights["lr"] * ml_probs["lr"]
        final_probs += self.weights["dixon"] * dc_probs
        final_probs += self.weights["markov"] * markov_probs

        # Normalize to ensure sum=1
        final_probs /= final_probs.sum()

        return {
            "home": final_probs[0],
            "draw": final_probs[1],
            "away": final_probs[2]
        }
