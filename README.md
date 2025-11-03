
# Volatility and Tail Risk in Gold Prices (2015–2025)

This notebook explores time-varying volatility and downside risk of gold prices using historical Value-at-Risk (VaR) and Conditional VaR (CVaR).

## Methods
- Daily log returns from Yahoo Finance.
- Historical quantile (nonparametric) VaR and CVaR.
- Rolling 252-day (1-year) VaR and volatility.

## Key Findings
- Gold’s 1-day 95% VaR ≈ −1.56%, 99% VaR ≈ −2.63%.
- Risk surged during 2020–2021 and 2022.
- Volatility clustering evident throughout the decade.

## Tools
- Python: pandas, numpy, matplotlib, yfinance
- Techniques: Quantile estimation, rolling windows

## Next Steps
- Compare Gold vs S&P 500 VaR.
- Add GARCH or Parametric VaR.
- Integrate into a portfolio-risk dashboard.
