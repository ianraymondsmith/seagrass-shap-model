"""
test_validation.py
==================
Unit tests for src/utils/validation.py.

Run with:
    pytest tests/test_validation.py -v

AUTHOR:   [Author placeholder]
CREATED:  2026
"""

import pandas as pd
import numpy as np
import pytest

from src.utils.validation import (
    validate_wq_columns,
    validate_coordinates,
    validate_health_labels,
    validate_no_target_leakage,
    validate_no_constant_features,
    validate_missing_fractions,
    validate_wq_ranges,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_wq_df():
    """Minimal valid WQ DataFrame for testing."""
    return pd.DataFrame({
        "station_id":    ["FB1", "FB2", "FB3"],
        "date":          pd.to_datetime(["2023-06-01", "2023-06-15", "2023-07-01"]),
        "chla_ug_L":     [1.2, 3.4, 0.8],
        "tpo4_mg_L":     [0.01, 0.02, 0.005],
        "totn_mg_L":     [0.5, 0.8, 0.3],
        "do_mg_L":       [7.2, 6.8, 7.5],
        "salinity_ppt":  [35.0, 38.0, 32.5],
        "turbidity_NTU": [2.5, 5.0, 1.8],
        "temp_C":        [28.0, 29.5, 27.0],
        "ph":            [8.1, 8.0, 8.2],
        "latitude":      [25.05, 25.10, 25.15],
        "longitude":     [-80.70, -80.80, -80.60],
        "health_status": ["Healthy", "Intermediate", "Not Healthy"],
    })


# ---------------------------------------------------------------------------
# Tests: validate_wq_columns
# ---------------------------------------------------------------------------

class TestValidateWqColumns:
    def test_passes_when_all_columns_present(self, minimal_wq_df):
        """Should not raise when all required columns are present."""
        required = list(minimal_wq_df.columns)
        validate_wq_columns(minimal_wq_df, required)  # no exception

    def test_raises_on_missing_column(self, minimal_wq_df):
        """Should raise ValueError when a required column is absent."""
        with pytest.raises(ValueError, match="missing required columns"):
            validate_wq_columns(minimal_wq_df, ["chla_ug_L", "nonexistent_col"])


# ---------------------------------------------------------------------------
# Tests: validate_coordinates
# ---------------------------------------------------------------------------

class TestValidateCoordinates:
    def test_valid_coordinates_unchanged(self, minimal_wq_df):
        result = validate_coordinates(minimal_wq_df)
        assert len(result) == len(minimal_wq_df)

    def test_drops_null_coordinates(self, minimal_wq_df):
        df = minimal_wq_df.copy()
        df.loc[0, "latitude"] = np.nan
        result = validate_coordinates(df)
        assert len(result) == len(minimal_wq_df) - 1

    def test_drops_zero_zero_coordinates(self, minimal_wq_df):
        df = minimal_wq_df.copy()
        df.loc[0, "latitude"] = 0.0
        df.loc[0, "longitude"] = 0.0
        result = validate_coordinates(df)
        assert len(result) == len(minimal_wq_df) - 1

    def test_drops_out_of_bounds(self, minimal_wq_df):
        df = minimal_wq_df.copy()
        df.loc[0, "latitude"] = 30.0   # Outside Florida Bay
        df.loc[0, "longitude"] = -85.0
        result = validate_coordinates(df)
        assert len(result) == len(minimal_wq_df) - 1

    def test_raises_on_missing_coordinate_column(self, minimal_wq_df):
        df = minimal_wq_df.drop(columns=["latitude"])
        with pytest.raises(ValueError, match="not found"):
            validate_coordinates(df)


# ---------------------------------------------------------------------------
# Tests: validate_health_labels
# ---------------------------------------------------------------------------

class TestValidateHealthLabels:
    def test_valid_labels_pass(self, minimal_wq_df):
        validate_health_labels(minimal_wq_df["health_status"])

    def test_unexpected_label_raises(self, minimal_wq_df):
        df = minimal_wq_df.copy()
        df.loc[0, "health_status"] = "Unknown"
        with pytest.raises(ValueError, match="Unexpected health labels"):
            validate_health_labels(df["health_status"])


# ---------------------------------------------------------------------------
# Tests: validate_no_target_leakage
# ---------------------------------------------------------------------------

class TestValidateNoTargetLeakage:
    def test_no_target_col_passes(self):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        validate_no_target_leakage(X, target_col="health_status")

    def test_target_present_raises(self):
        X = pd.DataFrame({"a": [1, 2], "health_status": [0, 1]})
        with pytest.raises(ValueError, match="leakage"):
            validate_no_target_leakage(X, target_col="health_status")


# ---------------------------------------------------------------------------
# Tests: validate_no_constant_features
# ---------------------------------------------------------------------------

class TestValidateNoConstantFeatures:
    def test_returns_empty_list_for_varied_features(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = validate_no_constant_features(X)
        assert result == []

    def test_identifies_constant_column(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [7, 7, 7]})
        result = validate_no_constant_features(X)
        assert "b" in result


# ---------------------------------------------------------------------------
# Tests: validate_missing_fractions
# ---------------------------------------------------------------------------

class TestValidateMissingFractions:
    def test_no_missing_returns_empty(self, minimal_wq_df):
        result = validate_missing_fractions(minimal_wq_df, max_missing_fraction=0.4)
        assert result == []

    def test_identifies_high_missing_column(self):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, np.nan],
            "b": [1, 2, 3, 4],
        })
        result = validate_missing_fractions(df, max_missing_fraction=0.4)
        assert "a" in result


# ---------------------------------------------------------------------------
# Tests: validate_wq_ranges
# ---------------------------------------------------------------------------

class TestValidateWqRanges:
    def test_valid_ranges_pass_without_error(self, minimal_wq_df):
        """Should return DataFrame unchanged when all values are in range."""
        result = validate_wq_ranges(minimal_wq_df)
        assert len(result) == len(minimal_wq_df)

    def test_accepts_out_of_range_without_dropping(self, minimal_wq_df):
        """Out-of-range values are warned but NOT dropped (analyst decision)."""
        df = minimal_wq_df.copy()
        df.loc[0, "temp_C"] = 99.9   # Physically impossible but not auto-dropped
        result = validate_wq_ranges(df)
        assert len(result) == len(minimal_wq_df)
