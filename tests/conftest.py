"""
Pytest configuration and fixtures.
"""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    os.environ["ALL_SPORTS_API_KEY"] = "test_key_all_sports"
    os.environ["THE_ODDS_API_KEY"] = "test_key_odds"
    yield
    # Cleanup if needed
