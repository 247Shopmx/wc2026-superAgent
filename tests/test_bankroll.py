"""
Tests for Bankroll Management module.
"""
import pytest
from src.bankroll import BankrollManager


def test_bankroll_initialization():
    """Test BankrollManager initialization."""
    mgr = BankrollManager(bankroll=5000.0, kelly_fraction=0.25)
    
    assert mgr.bankroll == 5000.0
    assert mgr.kelly_fraction == 0.25


def test_kelly_stake_calculation_positive_edge():
    """Test Kelly stake calculation with positive expected value."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=1.0)
    
    # Probability=60%, Odds=2.0
    # Kelly = (1.0 * 0.6 - 0.4) / 1.0 = 0.2
    # Stake = 0.2 * 1.0 * 10000 = 2000
    stake = mgr.calculate_kelly_stake(probability=0.6, decimal_odds=2.0)
    
    assert stake == pytest.approx(2000.0, rel=1e-3)


def test_kelly_stake_calculation_fractional():
    """Test Kelly stake with fractional kelly."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=0.25)
    
    # Same as above but with 0.25 fraction
    # Stake = 0.2 * 0.25 * 10000 = 500
    stake = mgr.calculate_kelly_stake(probability=0.6, decimal_odds=2.0)
    
    assert stake == pytest.approx(500.0, rel=1e-3)


def test_kelly_stake_no_edge():
    """Test Kelly stake with no edge (p * o = 1)."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=1.0)
    
    # Probability=50%, Odds=2.0 => EV = 1.0
    stake = mgr.calculate_kelly_stake(probability=0.5, decimal_odds=2.0)
    
    assert stake == pytest.approx(0.0, abs=1e-6)


def test_kelly_stake_negative_edge():
    """Test Kelly stake with negative edge."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=1.0)
    
    # Probability=40%, Odds=2.0 => EV < 1.0 => negative kelly
    stake = mgr.calculate_kelly_stake(probability=0.4, decimal_odds=2.0)
    
    assert stake == 0.0  # Should not bet


def test_kelly_stake_invalid_odds():
    """Test Kelly stake with invalid odds (<=1.0)."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=1.0)
    
    stake = mgr.calculate_kelly_stake(probability=0.6, decimal_odds=0.5)
    
    assert stake == 0.0


def test_detect_value_with_value_bets():
    """Test value bet detection."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=0.25)
    
    ensemble_probs = {"home": 0.6, "draw": 0.25, "away": 0.15}
    market_odds = {"home": 2.0, "draw": 3.0, "away": 4.0}
    
    value_bets = mgr.detect_value(ensemble_probs, market_odds)
    
    # home: 0.6 * 2.0 = 1.2 > 1.0 => value bet
    # draw: 0.25 * 3.0 = 0.75 < 1.0 => no value
    # away: 0.15 * 4.0 = 0.6 < 1.0 => no value
    assert len(value_bets) == 1
    assert value_bets[0]["outcome"] == "home"
    assert value_bets[0]["expected_value"] == pytest.approx(1.2)


def test_detect_value_no_value_bets():
    """Test when no value bets exist."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=0.25)
    
    ensemble_probs = {"home": 0.3, "draw": 0.35, "away": 0.35}
    market_odds = {"home": 2.0, "draw": 2.5, "away": 2.5}
    
    value_bets = mgr.detect_value(ensemble_probs, market_odds)
    
    # All have EV < 1.0
    assert len(value_bets) == 0


def test_detect_value_missing_odds():
    """Test value detection when some outcomes don't have odds."""
    mgr = BankrollManager(bankroll=10000.0, kelly_fraction=0.25)
    
    ensemble_probs = {"home": 0.6, "draw": 0.25, "away": 0.15}
    market_odds = {"home": 2.0}  # Only home odds available
    
    value_bets = mgr.detect_value(ensemble_probs, market_odds)
    
    assert len(value_bets) == 1
    assert value_bets[0]["outcome"] == "home"
