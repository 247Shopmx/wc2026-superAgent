"""
Tests for Ensemble Training module.
"""
import pytest
import pandas as pd
import numpy as np
from src.ensemble import EnsembleTrainer


def test_ensemble_initialization():
    """Test EnsembleTrainer initialization."""
    trainer = EnsembleTrainer()
    
    assert trainer.is_trained is False
    assert "xgb" in trainer.models
    assert "lgbm" in trainer.models
    assert "rf" in trainer.models
    assert "lr" in trainer.models


def test_ensemble_training_with_numpy_array():
    """Test ensemble training with numpy array targets."""
    trainer = EnsembleTrainer()
    
    # Create dummy data
    X_train = pd.DataFrame(
        np.random.rand(100, 7),
        columns=[
            "offensive_rating_home",
            "defensive_rating_home",
            "offensive_rating_away",
            "defensive_rating_away",
            "rest_days_home",
            "rest_days_away",
            "h2h_dominance",
        ],
    )
    y_train = np.random.choice([0, 1, 2], 100)
    
    trainer.train_ml_models(X_train, y_train)
    
    assert trainer.is_trained is True


def test_ensemble_training_with_pandas_series():
    """Test ensemble training with pandas Series targets."""
    trainer = EnsembleTrainer()
    
    # Create dummy data
    X_train = pd.DataFrame(
        np.random.rand(100, 7),
        columns=[
            "offensive_rating_home",
            "defensive_rating_home",
            "offensive_rating_away",
            "defensive_rating_away",
            "rest_days_home",
            "rest_days_away",
            "h2h_dominance",
        ],
    )
    y_train = pd.Series(np.random.choice([0, 1, 2], 100))
    
    trainer.train_ml_models(X_train, y_train)
    
    assert trainer.is_trained is True


def test_ensemble_prediction_before_training():
    """Test that prediction fails before training."""
    trainer = EnsembleTrainer()
    features = pd.DataFrame(np.random.rand(1, 7))
    
    with pytest.raises(RuntimeError, match="Models not trained yet"):
        trainer.predict_ensemble(features, lambda_home=1.5, lambda_away=1.3)


def test_ensemble_training_and_prediction():
    """Test complete ensemble training and prediction pipeline."""
    trainer = EnsembleTrainer()
    
    # Create dummy data
    X_train = pd.DataFrame(
        np.random.rand(100, 7),
        columns=[
            "offensive_rating_home",
            "defensive_rating_home",
            "offensive_rating_away",
            "defensive_rating_away",
            "rest_days_home",
            "rest_days_away",
            "h2h_dominance",
        ],
    )
    y_train = np.random.choice([0, 1, 2], 100)
    
    trainer.train_ml_models(X_train, y_train)
    
    # Test prediction
    features = pd.DataFrame(np.random.rand(1, 7))
    probs = trainer.predict_ensemble(features, lambda_home=1.5, lambda_away=1.3)
    
    assert "home" in probs
    assert "draw" in probs
    assert "away" in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert all(0 <= p <= 1 for p in probs.values())


def test_ensemble_prediction_probability_bounds():
    """Test that ensemble probabilities are valid."""
    trainer = EnsembleTrainer()
    
    X_train = pd.DataFrame(
        np.random.rand(100, 7),
        columns=[
            "offensive_rating_home",
            "defensive_rating_home",
            "offensive_rating_away",
            "defensive_rating_away",
            "rest_days_home",
            "rest_days_away",
            "h2h_dominance",
        ],
    )
    y_train = np.random.choice([0, 1, 2], 100)
    
    trainer.train_ml_models(X_train, y_train)
    features = pd.DataFrame(np.random.rand(1, 7))
    
    for _ in range(10):  # Run multiple times for robustness
        probs = trainer.predict_ensemble(
            features, lambda_home=np.random.rand() + 1, lambda_away=np.random.rand() + 1
        )
        
        # Check probability constraints
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-5
