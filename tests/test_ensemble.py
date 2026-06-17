import pytest
import pandas as pd
import numpy as np
from src.ensemble import EnsembleTrainer


def test_ensemble_training_and_prediction():
    trainer = EnsembleTrainer()
    
    # Create dummy data
    X_train = pd.DataFrame(np.random.rand(100, 7), columns=[
        "offensive_rating_home", "defensive_rating_home",
        "offensive_rating_away", "defensive_rating_away",
        "rest_days_home", "rest_days_away", "h2h_dominance"
    ])
    y_train = np.random.choice([0, 1, 2], 100)
    
    trainer.train_ml_models(X_train, y_train)
    
    # Test prediction
    features = pd.DataFrame([np.random.rand(7)])
    probs = trainer.predict_ensemble(features, lambda_home=1.5, lambda_away=1.3)
    
    assert "home" in probs
    assert "draw" in probs
    assert "away" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-5
