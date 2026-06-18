"""
Tests for Mathematical Models module.
"""
import pytest
import numpy as np
from src.models import DixonColesModel, MarkovChainModel


class TestDixonColesModel:
    """Test Dixon-Coles model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DixonColesModel(rho=-0.13)
        assert model.rho == -0.13
    
    def test_predict_score_probabilities_shape(self):
        """Test score probability matrix shape."""
        model = DixonColesModel()
        probs = model.predict_score_probabilities(lambda_home=1.5, lambda_away=1.3)
        
        assert probs.shape == (11, 11)  # max_goals=10 => 0..10
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    
    def test_predict_score_probabilities_values(self):
        """Test that all probabilities are non-negative."""
        model = DixonColesModel()
        probs = model.predict_score_probabilities(lambda_home=1.5, lambda_away=1.3)
        
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_get_match_outcome_probs_sum(self):
        """Test that outcome probabilities sum to 1."""
        model = DixonColesModel()
        p_home, p_draw, p_away = model.get_match_outcome_probs(
            lambda_home=1.5, lambda_away=1.3
        )
        
        total = p_home + p_draw + p_away
        assert np.isclose(total, 1.0, atol=1e-5)
    
    def test_get_match_outcome_probs_bounds(self):
        """Test that outcome probabilities are valid."""
        model = DixonColesModel()
        p_home, p_draw, p_away = model.get_match_outcome_probs(
            lambda_home=1.5, lambda_away=1.3
        )
        
        assert 0 <= p_home <= 1
        assert 0 <= p_draw <= 1
        assert 0 <= p_away <= 1
    
    def test_home_advantage_effect(self):
        """Test that higher home lambda increases home win probability."""
        model = DixonColesModel()
        
        p_home_weak, _, _ = model.get_match_outcome_probs(
            lambda_home=1.0, lambda_away=1.5
        )
        p_home_strong, _, _ = model.get_match_outcome_probs(
            lambda_home=2.0, lambda_away=1.5
        )
        
        assert p_home_strong > p_home_weak


class TestMarkovChainModel:
    """Test Markov Chain model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MarkovChainModel()
        assert model.transition_matrix.shape == (4, 4)
    
    def test_transition_matrix_stochastic(self):
        """Test that transition matrix rows sum to 1."""
        model = MarkovChainModel()
        row_sums = np.sum(model.transition_matrix, axis=1)
        
        assert np.allclose(row_sums, 1.0)
    
    def test_simulate_possession_outcome_bounds(self):
        """Test that possession outcome is a valid probability."""
        model = MarkovChainModel()
        outcome = model.simulate_possession_outcome(steps=10)
        
        assert 0 <= outcome <= 1
    
    def test_simulate_possession_outcome_deterministic(self):
        """Test that simulation is deterministic (for same inputs)."""
        model = MarkovChainModel()
        outcome1 = model.simulate_possession_outcome(steps=10)
        outcome2 = model.simulate_possession_outcome(steps=10)
        
        assert outcome1 == outcome2
    
    def test_simulate_possession_outcome_monotonic(self):
        """Test that longer steps don't decrease goal probability."""
        model = MarkovChainModel()
        outcome_short = model.simulate_possession_outcome(steps=5)
        outcome_long = model.simulate_possession_outcome(steps=20)
        
        # Goal state is absorbing, so more steps = higher goal probability
        assert outcome_long >= outcome_short
