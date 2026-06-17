"""
Mathematical & Distribution Engines
Dixon-Coles and Markov Chain implementations.
"""
import numpy as np
from scipy.stats import poisson
from typing import Tuple


class DixonColesModel:
    """Dixon-Coles modification of Poisson Distribution for ExpG."""

    def __init__(self, rho: float = -0.13):
        self.rho = rho  # Correction parameter for low scores

    def predict_score_probabilities(self, lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
        """Calculate probability matrix for scorelines."""
        probs = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                # Dixon-Coles adjustment
                if i == 0 and j == 0:
                    p *= (1 - self.rho * lambda_home * lambda_away)
                elif i == 1 and j == 0:
                    p *= (1 + self.rho * lambda_away)
                elif i == 0 and j == 1:
                    p *= (1 + self.rho * lambda_home)
                elif i == 1 and j == 1:
                    p *= (1 - self.rho)
                probs[i][j] = p
        return probs

    def get_match_outcome_probs(self, lambda_home: float, lambda_away: float) -> Tuple[float, float, float]:
        """Return P(Home Win), P(Draw), P(Away Win)."""
        probs = self.predict_score_probabilities(lambda_home, lambda_away)
        p_home = np.sum(np.tril(probs, -1))
        p_draw = np.sum(np.diag(probs))
        p_away = np.sum(np.triu(probs, 1))
        return p_home, p_draw, p_away


class MarkovChainModel:
    """Markov Chain Transition Matrices for state progression."""

    def __init__(self):
        # Simplified transition matrix: States [Defensive, Midfield, Attacking, Goal]
        self.transition_matrix = np.array([
            [0.7, 0.2, 0.1, 0.0],
            [0.2, 0.6, 0.15, 0.05],
            [0.1, 0.3, 0.5, 0.1],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def simulate_possession_outcome(self, steps: int = 10) -> float:
        """Simulate probability of reaching 'Goal' state."""
        state = np.array([1, 0, 0, 0])  # Start in Midfield
        for _ in range(steps):
            state = state @ self.transition_matrix
        return state[3]  # Probability of being in Goal state
