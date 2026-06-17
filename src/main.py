"""
World Cup 2026 Predictive Trading Superagent
Main execution entry point.
"""
import logging
import numpy as np
import pandas as pd
from src.data_extractor import DataExtractor
from src.feature_engineer import FeatureEngineer
from src.ensemble import EnsembleTrainer
from src.bankroll import BankrollManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting WC2026 Superagent...")
    
    # Inicialización de componentes
    engineer = FeatureEngineer()
    trainer = EnsembleTrainer()
    bankroll_mgr = BankrollManager(bankroll=10000.0, kelly_fraction=0.25)
    
    # Flujo de trabajo principal
    try:
        extractor = DataExtractor()
        matches = extractor.get_upcoming_matches()
        if not matches:
            logger.warning("No matches found.")
            return
            
        # Entrenamiento Sintético para Demo (En producción, cargar datos históricos reales)
        logger.info("Generating synthetic training data for pipeline initialization...")
        np.random.seed(42)
        n_samples = 1000
        X_train = pd.DataFrame(np.random.rand(n_samples, 7), columns=[
            "offensive_rating_home", "defensive_rating_home",
            "offensive_rating_away", "defensive_rating_away",
            "rest_days_home", "rest_days_away", "h2h_dominance"
        ])
        y_train = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.25, 0.30])
        trainer.train_ml_models(X_train, y_train)
        
        for match in matches[:5]:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            
            logger.info(f"\nAnalyzing: {home_team} vs {away_team}")

            # Extract Odds (Simplified parsing)
            best_odds = {"home": 0, "draw": 0, "away": 0}
            bookmakers = match.get("bookmakers", [])
            for bm in bookmakers:
                for m in bm.get("markets", []):
                    if m["key"] == "h2h":
                        for outcome in m["outcomes"]:
                            name = outcome["name"]
                            price = outcome["price"]
                            if name == home_team and price > best_odds["home"]:
                                best_odds["home"] = price
                            elif name == away_team and price > best_odds["away"]:
                                best_odds["away"] = price
                            elif name == "Draw" and price > best_odds["draw"]:
                                best_odds["draw"] = price

            # Feature Engineering (Mocked for demo)
            mock_features = pd.DataFrame([{
                "offensive_rating_home": 1.8, "defensive_rating_home": 1.2,
                "offensive_rating_away": 1.5, "defensive_rating_away": 1.4,
                "rest_days_home": 4, "rest_days_away": 3,
                "h2h_dominance": 0.6
            }])

            # Predict
            ensemble_probs = trainer.predict_ensemble(mock_features, lambda_home=1.6, lambda_away=1.3)
            
            # Value Detection
            value_bets = bankroll_mgr.detect_value(ensemble_probs, best_odds)
            
            print(f"  Ensemble Probs: Home={ensemble_probs['home']:.2f}, Draw={ensemble_probs['draw']:.2f}, Away={ensemble_probs['away']:.2f}")
            print(f"  Best Odds: Home={best_odds['home']}, Draw={best_odds['draw']}, Away={best_odds['away']}")
            
            if value_bets:
                for vb in value_bets:
                    print(f"  ⚠️ VALUE BET DETECTED: {vb['outcome'].upper()} | EV: {vb['expected_value']:.2f} | Stake: ${vb['stake_recommendation']:.2f}")
            else:
                print("  No value bets detected.")

    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)


if __name__ == "__main__":
    main()
