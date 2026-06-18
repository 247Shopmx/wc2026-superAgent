"""
Tests for Data Extractor module.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.data_extractor import DataExtractor


def test_data_extractor_initialization_with_mock_keys():
    """Test DataExtractor initialization with injected API keys."""
    api_keys = {"all_sports_key": "test_key_1", "odds_key": "test_key_2"}
    extractor = DataExtractor(api_keys=api_keys)
    
    assert extractor.all_sports_api_key == "test_key_1"
    assert extractor.odds_api_key == "test_key_2"


def test_data_extractor_initialization_from_env():
    """Test DataExtractor initialization from environment variables."""
    extractor = DataExtractor()
    # These should be set by conftest.py
    assert extractor.all_sports_api_key == "test_key_all_sports"
    assert extractor.odds_api_key == "test_key_odds"


def test_fetch_odds_data_success():
    """Test successful odds data fetch with mocking."""
    with patch("src.data_extractor.requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "123", "home_team": "France", "away_team": "Argentina"}
        ]
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        
        api_keys = {"all_sports_key": "test", "odds_key": "test"}
        extractor = DataExtractor(api_keys=api_keys)
        extractor.session = mock_session  # Replace session with mock
        
        result = extractor.fetch_odds_data("soccer_fifa_world_cup", ["h2h"])
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "123"


def test_fetch_odds_data_request_error():
    """Test odds data fetch with request error."""
    import requests
    
    with patch("src.data_extractor.requests.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.RequestException(
            "Network error"
        )
        
        api_keys = {"all_sports_key": "test", "odds_key": "test"}
        extractor = DataExtractor(api_keys=api_keys)
        extractor.session = mock_session
        
        result = extractor.fetch_odds_data("soccer_fifa_world_cup", ["h2h"])
        
        assert result == []


def test_get_upcoming_matches():
    """Test get_upcoming_matches wrapper."""
    with patch.object(DataExtractor, "fetch_odds_data") as mock_fetch:
        mock_fetch.return_value = [{"id": "456", "sport": "soccer"}]
        
        api_keys = {"all_sports_key": "test", "odds_key": "test"}
        extractor = DataExtractor(api_keys=api_keys)
        result = extractor.get_upcoming_matches()
        
        assert len(result) == 1
        assert result[0]["id"] == "456"
        mock_fetch.assert_called_once_with(
            "soccer_fifa_world_cup", ["h2h", "totals", "spreads"]
        )
