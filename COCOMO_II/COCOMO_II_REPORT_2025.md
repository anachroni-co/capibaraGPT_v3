# COCOMO II Technical Valuation Report (2025)

## Scope
- Repository: `capibaraGPT_v3`
- Objective: Internal technical valuation (non-accounting) using a calibrated COCOMO II scenario.
- Date: 2026-02-07

## Baseline Inputs
- `KLOC`: `128.28`
- `A`: `0.35` (internally calibrated for high AI-assisted productivity)
- `Cost per person-month`: `5,500` (same currency used in the calculation)

## Scale Factors (SF)
- `PREC=1.24`
- `FLEX=1.01`
- `RESL=2.83`
- `TEAM=2.19`
- `PMAT=3.12`

Derived:
- `B = 0.91 + 0.01 * sum(SF) = 1.0139`

## Effort Multipliers (EM)
- `RELY=1.00`
- `DATA=1.00`
- `CPLX=1.00`
- `RUSE=1.00`
- `TIME=1.00`
- `PVOL=1.00`
- `ACAP=0.85`
- `PCAP=0.88`

Derived:
- `EAF = product(EM) = 0.7480`

## COCOMO II Calculation
Formulas:
- `Effort (PM) = A * KLOC^B * EAF`
- `D = 0.28 + 0.2 * (B - 0.91)`
- `Schedule (months) = 3.67 * Effort^D`
- `Total Cost = Effort * Cost per PM`

Computed values:
- `Effort`: `35.927914` person-months
- `Schedule`: `10.777273` months
- `Total Cost`: `197,603.527182`

## Interpretation
- This is a **technical valuation scenario** calibrated to a low-effort assumption (`A=0.35`).
- It is not a canonical default COCOMO II calibration.
- Use this report as internal reference for technical valuation alignment.

## 2024 vs 2025 Comparison
Because repository history does not contain commits from 2024, 2024 LOC cannot be measured directly from `git`.
The 2024 LOC shown below is an inferred estimate using the 2025 ratio:

- `2025 EUR/LOC = 197,603.53 / 128,280 = 1.5406 EUR/LOC`
- `2024 inferred LOC = 115,000 / 1.5406 = 74,646 LOC` (approx.)

| Year | LOC | KLOC | Technical valuation (EUR) | Source quality |
|---|---:|---:|---:|---|
| 2024 | 74,646 (inferred) | 74.65 | 115,000.00 | Estimated (no 2024 git snapshot) |
| 2025 | 128,280 (measured) | 128.28 | 197,603.53 | Measured from current repo + COCOMO II (`A=0.35`) |

## Reproducibility
Command used:

```bash
python COCOMO_II/cocomo_ii.py --kloc 128.28 --cost-per-person-month 5500 --a 0.35
```

## Licensing Model
This software is distributed under a **dual-license model**:

1. **Open-source / non-commercial license**:
- Allowed for research, education, and non-profit/non-commercial use.

2. **Commercial license**:
- Required for companies and any for-profit/commercial use.

This dual model allows open collaboration in academic and public-interest contexts, while preserving a commercial licensing path for enterprise exploitation.
