# Systematic ETF Momentum Strategy

## Overview

This project investigates a systematic momentum-based asset allocation strategy using a diversified ETF universe.

The strategy ranks ETFs based on trailing 12-month momentum, selects the strongest assets, and constructs a monthly rebalanced portfolio. Performance is evaluated relative to a monthly equal-weight benchmark while accounting for portfolio drift, turnover, and transaction costs.

The project was developed as an end-to-end quantitative research workflow using Python, pandas, Jupyter notebooks, and Git.

## Research Question

Can a simple cross-sectional momentum strategy outperform a monthly rebalanced equal-weight portfolio across a diversified ETF universe?

Specifically:

- Does momentum improve risk-adjusted performance?
- How much turnover does the strategy generate?
- What is the impact of transaction costs?

## ETF Universe

| ETF | Asset Class |
|------|------|
| SPY | US Equities |
| EFA | Developed Markets Equities |
| EEM | Emerging Markets Equities |
| TLT | Long-Term US Treasuries |
| GLD | Gold |
| VNQ | Real Estate |

## Methodology

### Benchmark

The benchmark portfolio allocates equally across all ETFs in the universe.

- Monthly rebalancing
- Equal target weights
- Intra-month weight drift

### Momentum Strategy

The momentum strategy uses a 12-month trailing return signal.

At each month-end:

1. Rank all ETFs by momentum.
2. Select the top 3 assets.
3. Allocate equally across selected assets.
4. Hold positions during the following month.
5. Allow weights to drift intra-month.

### Transaction Costs

Transaction costs are modeled as:

Cost = Turnover × Cost Rate

where:

- Turnover = 0.5 × Σ|new weight − old weight|
- Cost Rate = 10 basis points (0.10%)

This allows comparison of gross and net strategy performance.

## Results

The momentum strategy was evaluated against a monthly rebalanced equal-weight benchmark.

Performance was analyzed using:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Rolling Risk Metrics
- Turnover Analysis
- Transaction Cost Analysis

A detailed discussion of results can be found in:

```text
notebooks/10_final_research_report.ipynb
```

## Key Findings

- Momentum successfully identified changing leadership across asset classes.
- Portfolio turnover was generally low and increased primarily during market regime shifts.
- Transaction cost drag remained modest under a monthly rebalancing framework.
- The strategy rotated across equities, real estate, gold, and fixed income rather than maintaining static exposures.
- Risk and performance characteristics varied across market environments, highlighting the importance of rolling analytics and drawdown analysis.

## Equity Curve

The figure below compares cumulative returns of the monthly equal-weight benchmark and the momentum strategy.

![Equity Curve](/reports/images/Equity%20Curves.png)

## Repository Structure

```text
data/
│
├── raw/
│
notebooks/
│
├── 01_data_download.ipynb
├── 02_return_calculation.ipynb
├── ...
├── 10_final_research_report.ipynb
│
src/
│
├── data.py
├── metrics.py
├── signals.py
├── portfolio.py
├── costs.py
├── diagnostics.py
│
README.md
```

## Future Improvements

Potential extensions include:

- Larger ETF universe
- Alternative momentum definitions
- Volatility targeting
- Portfolio optimization
- More realistic transaction cost models
- Performance attribution and factor analysis

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd systematic-etf-portfolio-lab
```

Install dependencies:

```bash
pip install -r requirements.txt
```