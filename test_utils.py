"""
Unit tests for Monte Carlo Risk Simulator utility functions.
Run with: python -m pytest test_utils.py -v
"""

import pytest
import numpy as np
import pandas as pd
from utils import (
    normalize_weights,
    weights_from_holdings,
    parse_tickers,
    validate_alpha,
    clean_prices,
    compute_log_returns,
    max_drawdown,
    time_to_failure,
)


# ============================================================================
# Test: normalize_weights
# ============================================================================

class TestNormalizeWeights:
    """Test weight normalization."""
    
    def test_normalizes_to_one(self):
        """Weights should sum to 1.0 after normalization."""
        weights = normalize_weights(np.array([1, 2, 3]))
        assert np.isclose(weights.sum(), 1.0), f"Sum is {weights.sum()}, expected 1.0"
    
    def test_already_normalized(self):
        """Weights that already sum to 1 should stay the same."""
        weights = np.array([0.2, 0.3, 0.5])
        result = normalize_weights(weights)
        assert np.allclose(result, weights)
    
    def test_negative_weights_raise_error(self):
        """Negative weights should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            normalize_weights(np.array([-1, 2, 3]))
    
    def test_all_zeros_raise_error(self):
        """All-zero weights should raise ValueError."""
        with pytest.raises(ValueError, match="must be greater than 0"):
            normalize_weights(np.array([0, 0, 0]))
    
    def test_single_asset(self):
        """Single asset should normalize to 1.0."""
        result = normalize_weights(np.array([5.0]))
        assert np.isclose(result[0], 1.0)
    
    def test_large_numbers(self):
        """Should handle large numbers without overflow."""
        weights = np.array([1e10, 2e10, 3e10])
        result = normalize_weights(weights)
        assert np.isclose(result.sum(), 1.0)


# ============================================================================
# Test: weights_from_holdings
# ============================================================================

class TestWeightsFromHoldings:
    """Test conversion from dollar holdings to weights."""
    
    def test_converts_to_weights_and_total(self):
        """Should return normalized weights and total portfolio value."""
        holdings = np.array([5000, 3000, 2000])
        weights, total = weights_from_holdings(holdings)
        
        assert np.isclose(weights.sum(), 1.0)
        assert total == 10000
        assert np.isclose(weights[0], 0.5)
        assert np.isclose(weights[1], 0.3)
        assert np.isclose(weights[2], 0.2)
    
    def test_single_asset_holding(self):
        """Single asset should have weight 1.0."""
        weights, total = weights_from_holdings(np.array([10000]))
        assert np.isclose(weights[0], 1.0)
        assert total == 10000
    
    def test_negative_holdings_raise_error(self):
        """Negative holdings should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            weights_from_holdings(np.array([5000, -1000, 2000]))
    
    def test_all_zero_holdings_raise_error(self):
        """All-zero holdings should raise ValueError."""
        with pytest.raises(ValueError, match="greater than 0"):
            weights_from_holdings(np.array([0, 0, 0]))
    
    def test_very_small_holdings(self):
        """Should handle very small holdings correctly."""
        holdings = np.array([0.01, 0.02, 0.03])
        weights, total = weights_from_holdings(holdings)
        
        assert np.isclose(weights.sum(), 1.0)
        assert np.isclose(total, 0.06)


# ============================================================================
# Test: parse_tickers
# ============================================================================

class TestParseTickers:
    """Test ticker string parsing."""
    
    def test_single_ticker(self):
        """Should parse single ticker."""
        result = parse_tickers("SPY")
        assert result == ["SPY"]
    
    def test_multiple_tickers(self):
        """Should parse comma-separated tickers."""
        result = parse_tickers("SPY,AAPL,MSFT")
        assert result == ["SPY", "AAPL", "MSFT"]
    
    def test_whitespace_handling(self):
        """Should strip whitespace around tickers."""
        result = parse_tickers("SPY, AAPL , MSFT")
        assert result == ["SPY", "AAPL", "MSFT"]
    
    def test_lowercase_to_uppercase(self):
        """Should convert tickers to uppercase."""
        result = parse_tickers("spy,aapl,msft")
        assert result == ["SPY", "AAPL", "MSFT"]
    
    def test_empty_string_raises_error(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="atleast one ticker"):
            parse_tickers("")
    
    def test_only_whitespace_raises_error(self):
        """Only whitespace should raise ValueError."""
        with pytest.raises(ValueError, match="atleast one ticker"):
            parse_tickers("   ,  ,  ")
    
    def test_mixed_case_and_spaces(self):
        """Should handle mixed case and spaces."""
        result = parse_tickers(" GooGL , BRK.B , tsla ")
        assert result == ["GOOGL", "BRK.B", "TSLA"]


# ============================================================================
# Test: validate_alpha
# ============================================================================

class TestValidateAlpha:
    """Test alpha (confidence level) validation."""
    
    def test_valid_alpha(self):
        """Valid alpha values should pass through."""
        result = validate_alpha(0.95)
        assert result == 0.95
    
    def test_alpha_too_low(self):
        """Alpha <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_alpha(0.0)
    
    def test_alpha_too_high(self):
        """Alpha >= 1 should raise ValueError."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_alpha(1.0)
    
    def test_alpha_at_boundaries(self):
        """Alpha just inside bounds should work."""
        assert validate_alpha(0.001) > 0
        assert validate_alpha(0.999) < 1


# ============================================================================
# Test: clean_prices
# ============================================================================

class TestCleanPrices:
    """Test price data cleaning."""
    
    def test_removes_all_nan_rows(self):
        """Should remove rows where all values are NaN."""
        prices = pd.DataFrame({
            'A': [100, np.nan, 102],
            'B': [50, np.nan, 52]
        })
        result = clean_prices(prices)
        assert len(result) == 2
    
    def test_removes_partial_nan_rows(self):
        """Should remove rows with any NaN."""
        prices = pd.DataFrame({
            'A': [100, 101, 102],
            'B': [50, np.nan, 52]
        })
        result = clean_prices(prices)
        assert len(result) == 2
        assert list(result.index) == [0, 2]
    
    def test_valid_prices_unchanged(self):
        """Valid price data should not be modified."""
        prices = pd.DataFrame({
            'A': [100, 101, 102],
            'B': [50, 51, 52]
        })
        result = clean_prices(prices)
        assert result.equals(prices)
    
    def test_empty_dataframe(self):
        """Empty dataframe should return empty."""
        prices = pd.DataFrame({'A': [np.nan, np.nan]})
        result = clean_prices(prices)
        assert len(result) == 0


# ============================================================================
# Test: compute_log_returns
# ============================================================================

class TestComputeLogReturns:
    """Test log return computation."""
    
    def test_basic_log_returns(self):
        """Should compute log returns correctly."""
        prices = pd.DataFrame({'A': [100, 110, 121]})
        returns = compute_log_returns(prices)
        
        # ln(110/100) ≈ 0.0953, ln(121/110) ≈ 0.0953
        expected = np.array([np.log(110/100), np.log(121/110)])
        assert np.allclose(returns['A'].values, expected)
    
    def test_removes_inf_values(self):
        """Should replace infinities with NaN and drop them."""
        prices = pd.DataFrame({'A': [100, 0, 102]})  # 0 creates -inf
        returns = compute_log_returns(prices)
        
        assert not np.any(np.isinf(returns.values))
    
    def test_multiple_assets(self):
        """Should handle multiple assets."""
        prices = pd.DataFrame({
            'A': [100, 110, 121],
            'B': [50, 55, 60.5]
        })
        returns = compute_log_returns(prices)
        
        assert returns.shape == (2, 2)
        assert 'A' in returns.columns
        assert 'B' in returns.columns


# ============================================================================
# Test: max_drawdown
# ============================================================================

class TestMaxDrawdown:
    """Test maximum drawdown calculation."""
    
    def test_no_drawdown(self):
        """Always-increasing path has zero drawdown."""
        path = np.array([100, 110, 120, 130])
        dd = max_drawdown(path)
        assert dd == 0 or np.isclose(dd, 0)
    
    def test_complete_loss(self):
        """Falling to zero should be -100% drawdown."""
        path = np.array([100, 80, 60, 40, 0])
        dd = max_drawdown(path)
        assert np.isclose(dd, -1.0)
    
    def test_partial_recovery(self):
        """Peak-to-trough should catch largest decline."""
        path = np.array([100, 70, 80, 75, 85])
        dd = max_drawdown(path)
        # Max drawdown is 70/100 - 1 = -0.30
        assert np.isclose(dd, -0.30, atol=0.01)
    
    def test_single_value(self):
        """Single value should have zero drawdown."""
        path = np.array([100])
        dd = max_drawdown(path)
        assert dd == 0 or np.isclose(dd, 0)


# ============================================================================
# Test: time_to_failure
# ============================================================================

class TestTimeToFailure:
    """Test time-to-failure calculation."""
    
    def test_immediate_failure(self):
        """Should detect immediate failure."""
        path = np.array([100, 50, 40])
        ttf = time_to_failure(path, threshold=0.70)
        assert ttf == 1  # Fails at step 1
    
    def test_no_failure(self):
        """Should return None if threshold never breached."""
        path = np.array([100, 90, 85, 80, 75])
        ttf = time_to_failure(path, threshold=0.70)
        assert ttf is None
    
    def test_failure_at_end(self):
        """Should detect failure at end of path."""
        path = np.array([100, 90, 85, 70, 69])
        ttf = time_to_failure(path, threshold=0.70)
        assert ttf == 3  # Fails at step 3
    
    def test_different_thresholds(self):
        """Different thresholds should give different results."""
        path = np.array([100, 80, 60, 50, 40])
        
        ttf_strict = time_to_failure(path, threshold=0.50)
        ttf_loose = time_to_failure(path, threshold=0.30)
        
        assert ttf_strict is not None
        assert ttf_loose is None  # Never falls below 30%


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:
    """Test that functions work together."""
    
    def test_parse_and_normalize_workflow(self):
        """Parse tickers, then use with normalized weights."""
        tickers = parse_tickers("SPY, AAPL, MSFT")
        assert len(tickers) == 3
        
        weights = normalize_weights(np.array([1, 2, 3]))
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
    
    def test_holdings_to_weights_workflow(self):
        """Convert holdings to weights."""
        holdings = np.array([5000, 3000, 2000])
        weights, total = weights_from_holdings(holdings)
        
        assert len(weights) == 3
        assert np.isclose(weights.sum(), 1.0)
        assert total == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
