"""
Data Extraction Layer
Handles secure data retrieval from All Sports API and The Odds API.
"""
import os
import logging
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ALL_SPORTS_API_KEY = os.getenv("ALL_SPORTS_API_KEY")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")

if not ALL_SPORTS_API_KEY or not THE_ODDS_API_KEY:
    raise EnvironmentError(
        "Missing API keys. Ensure ALL_SPORTS_API_KEY and THE_ODDS_API_KEY are set in environment."
    )

ALL_SPORTS_BASE_URL = "https://apiv2.allsportsapi.com/football/"
THE_ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/"


class DataExtractor:
    """Handles secure data retrieval from All Sports API and The Odds API."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_all_sports_data(self, method: str, params: Dict[str, Any]) -> Dict:
        """Fetch data from All Sports API with error handling."""
        try:
            params["APIkey"] = ALL_SPORTS_API_KEY
            params["met"] = method
            response = self.session.get(
                ALL_SPORTS_BASE_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if data.get("success") != 1:
                logger.warning(f"All Sports API returned success=0 for {method}")
                return {}
            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"All Sports API Request Error ({method}): {e}")
            return {}
        except (KeyError, ValueError) as e:
            logger.error(f"All Sports API Parsing Error ({method}): {e}")
            return {}

    def fetch_odds_data(self, sport_key: str, markets: List[str], regions: str = "us") -> List[Dict]:
        """Fetch odds from The Odds API."""
        try:
            url = f"{THE_ODDS_BASE_URL}{sport_key}/odds"
            params = {
                "apiKey": THE_ODDS_API_KEY,
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": "decimal"
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"The Odds API Request Error: {e}")
            return []
        except (KeyError, ValueError) as e:
            logger.error(f"The Odds API Parsing Error: {e}")
            return []

    def get_upcoming_matches(self) -> List[Dict]:
        """Get upcoming World Cup matches with odds."""
        return self.fetch_odds_data("soccer_fifa_world_cup", ["h2h", "totals", "spreads"])
