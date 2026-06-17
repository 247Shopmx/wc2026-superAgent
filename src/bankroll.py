"""
Bankroll Management & Value Detection
Implements Fractional Kelly Criterion and Value Filtering.
"""
from typing import Dict, List


class BankrollManager:
    """Implements Fractional Kelly Criterion and Value Filtering."""

    def __init__(self, bankroll: float = 10000.0, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction

    def calculate_kelly_stake(self, probability: float, decimal_odds: float) -> float:
        """Calculate optimal stake size."""
        if decimal_odds <= 1.0:
            return 0.0
        b = decimal_odds - 1
        q = 1 - probability
        kelly = (b * probability - q) / b
        stake = kelly * self.kelly_fraction * self.bankroll
        return max(0, stake)  # No negative stakes

    def detect_value(self, ensemble_probs: Dict[str, float], market_odds: Dict[str, float]) -> List[Dict]:
        """Identify value bets where P_ensemble * O_max > 1.0."""
        value_bets = []
        for outcome, prob in ensemble_probs.items():
            if outcome in market_odds:
                odds = market_odds[outcome]
                expected_value = prob * odds
                if expected_value > 1.0:
                    stake = self.calculate_kelly_stake(prob, odds)
                    value_bets.append({
                        "outcome": outcome,
                        "probability": prob,
                        "odds": odds,
                        "expected_value": expected_value,
                        "stake_recommendation": stake
                    })
        return value_bets
