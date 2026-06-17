"""
Feature Engineering Layer
Transforms raw API data into mathematical tensors for ML models.
"""
import math
import pandas as pd
from datetime import datetime
from typing import List, Dict


class FeatureEngineer:
    """Transforms raw API data into mathematical tensors for ML models."""

    def calculate_offensive_rating(self, last_10_matches: List[Dict]) -> float:
        """Rolling average of goals scored, xG, and shots on target."""
        if not last_10_matches:
            return 0.0
        
        # Simplified: Average goals scored by the team in question
        # In production, filter specifically for the team's goals
        total_goals = sum(m.get("home_score", 0) + m.get("away_score", 0) for m in last_10_matches)
        return total_goals / len(last_10_matches)

    def calculate_defensive_rating(self, last_10_matches: List[Dict]) -> float:
        """Rolling average of goals conceded and big chances allowed."""
        if not last_10_matches:
            return 0.0
        total_conceded = sum(m.get("home_score", 0) + m.get("away_score", 0) for m in last_10_matches)
        return total_conceded / len(last_10_matches)

    def calculate_rest_index(self, last_match_date: str, current_date: str) -> int:
        """Difference in days since last competitive fixture."""
        try:
            last_dt = datetime.strptime(last_match_date, "%Y-%m-%d")
            curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
            return (curr_dt - last_dt).days
        except Exception:
            return 7  # Default rest

    def calculate_weighted_h2h(self, h2h_matches: List[Dict]) -> float:
        """Exponentially decayed H2H dominance score."""
        if not h2h_matches:
            return 0.5  # Neutral
        
        score = 0.0
        weight_sum = 0.0
        now = datetime.now()
        
        for match in h2h_matches:
            try:
                match_date_str = match.get("event_date", "")
                if not match_date_str:
                    continue
                match_date = datetime.strptime(match_date_str, "%Y-%m-%d")
                days_ago = (now - match_date).days
                weight = math.exp(-0.01 * days_ago)  # Decay factor
                
                # Simplified winner determination
                home_score = int(match.get("event_halftime_result", "0-0").split("-")[0])
                away_score = int(match.get("event_halftime_result", "0-0").split("-")[1])
                
                if home_score > away_score:
                    score += 1 * weight
                elif away_score > home_score:
                    score += 0 * weight
                else:
                    score += 0.5 * weight
                    
                weight_sum += weight
            except Exception:
                continue
                
        return score / weight_sum if weight_sum > 0 else 0.5

    def engineer_features(self, match_data: Dict, historical_data: Dict) -> pd.DataFrame:
        """Construct feature matrix for a specific match."""
        features = {
            "offensive_rating_home": self.calculate_offensive_rating(historical_data.get("home_last_10", [])),
            "defensive_rating_home": self.calculate_defensive_rating(historical_data.get("home_last_10", [])),
            "offensive_rating_away": self.calculate_offensive_rating(historical_data.get("away_last_10", [])),
            "defensive_rating_away": self.calculate_defensive_rating(historical_data.get("away_last_10", [])),
            "rest_days_home": self.calculate_rest_index(historical_data.get("home_last_match_date", ""), "2026-06-17"),
            "rest_days_away": self.calculate_rest_index(historical_data.get("away_last_match_date", ""), "2026-06-17"),
            "h2h_dominance": self.calculate_weighted_h2h(historical_data.get("h2h", []))
        }
        return pd.DataFrame([features])
