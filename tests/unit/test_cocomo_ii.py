"""
COCOMO II Tests - Unit tests for COCOMO II estimation model.

This module provides tests for the COCOMO II software cost estimation
model, validating effort multipliers, scale factors, and cost calculations.

Author: Skydesk International Dev Team.
"""

from COCOMO_II.cocomo_ii import (
    LOW_VALUATION_EFFORT_MULTIPLIERS,
    LOW_VALUATION_SCALE_FACTORS,
    cocomo_ii,
)


def test_low_valuation_profile_returns_consistent_values():
    result = cocomo_ii(kloc=15.5, cost_per_person_month=5500.0)

    # Sanity checks for baseline scenario.
    assert result.effort_pm > 0
    assert result.duration_months > 0
    assert result.total_cost > 0
    assert result.b_exponent > 0.91
    assert result.eaf < 1.0  # Lower valuation profile uses strong ACAP/PCAP


def test_low_valuation_profile_is_lower_than_experimental_like_profile():
    baseline = cocomo_ii(kloc=15.5, cost_per_person_month=5500.0)

    higher_em = dict(LOW_VALUATION_EFFORT_MULTIPLIERS)
    higher_sf = dict(LOW_VALUATION_SCALE_FACTORS)
    higher_em.update({"DATA": 1.14, "CPLX": 1.49, "RELY": 1.10, "TIME": 1.11, "PVOL": 1.15})
    higher_sf.update({"PREC": 4.96, "FLEX": 3.04, "RESL": 4.22, "TEAM": 3.29, "PMAT": 4.68})

    higher = cocomo_ii(
        kloc=15.5,
        cost_per_person_month=5500.0,
        scale_factors=higher_sf,
        effort_multipliers=higher_em,
    )

    assert baseline.effort_pm < higher.effort_pm
    assert baseline.duration_months < higher.duration_months
    assert baseline.total_cost < higher.total_cost

