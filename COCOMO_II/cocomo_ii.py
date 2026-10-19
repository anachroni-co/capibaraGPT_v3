#!/usr/bin/env python3
"""COCOMO II estimator with a conservative (low valuation) profile."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CocomoResult:
    effort_pm: float
    duration_months: float
    b_exponent: float
    eaf: float
    total_cost: float


LOW_VALUATION_SCALE_FACTORS: Dict[str, float] = {
    "PREC": 1.24,
    "FLEX": 1.01,
    "RESL": 2.83,
    "TEAM": 2.19,
    "PMAT": 3.12,
}

LOW_VALUATION_EFFORT_MULTIPLIERS: Dict[str, float] = {
    "RELY": 1.00,
    "DATA": 1.00,
    "CPLX": 1.00,
    "RUSE": 1.00,
    "TIME": 1.00,
    "PVOL": 1.00,
    "ACAP": 0.85,
    "PCAP": 0.88,
}


def cocomo_ii(
    kloc: float,
    cost_per_person_month: float,
    a: float = 2.94,
    scale_factors: Dict[str, float] | None = None,
    effort_multipliers: Dict[str, float] | None = None,
) -> CocomoResult:
    """Compute COCOMO II effort/duration/cost."""
    if kloc <= 0:
        raise ValueError("kloc must be > 0")
    if cost_per_person_month <= 0:
        raise ValueError("cost_per_person_month must be > 0")

    sf = scale_factors or LOW_VALUATION_SCALE_FACTORS
    em = effort_multipliers or LOW_VALUATION_EFFORT_MULTIPLIERS

    b = 0.91 + 0.01 * sum(sf.values())
    eaf = math.prod(em.values())
    effort = a * (kloc**b) * eaf

    c = 3.67
    d = 0.28 + 0.2 * (b - 0.91)
    duration = c * (effort**d)
    total_cost = effort * cost_per_person_month

    return CocomoResult(
        effort_pm=effort,
        duration_months=duration,
        b_exponent=b,
        eaf=eaf,
        total_cost=total_cost,
    )


def parse_args() -> Tuple[float, float, float]:
    parser = argparse.ArgumentParser(description="COCOMO II low valuation estimator")
    parser.add_argument(
        "--kloc",
        type=float,
        default=15.5,
        help="KLOC (thousands of effective source lines). Default: 15.5",
    )
    parser.add_argument(
        "--cost-per-person-month",
        type=float,
        default=5500.0,
        help="Internal cost per person-month. Default: 5500",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=2.94,
        help="COCOMO II calibration constant A. Default: 2.94",
    )
    args = parser.parse_args()
    return args.kloc, args.cost_per_person_month, args.a


def main() -> int:
    kloc, cost_per_person_month, a = parse_args()
    result = cocomo_ii(kloc=kloc, cost_per_person_month=cost_per_person_month, a=a)

    print("--- COCOMO II (Low Valuation Profile) ---")
    print(f"A constant: {a:.4f}")
    print(f"KLOC: {kloc:.2f}")
    print(f"B exponent: {result.b_exponent:.4f}")
    print(f"EAF: {result.eaf:.4f}")
    print(f"Estimated effort: {result.effort_pm:.2f} person-months")
    print(f"Estimated schedule: {result.duration_months:.2f} months")
    print(f"Estimated total cost: ${result.total_cost:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
