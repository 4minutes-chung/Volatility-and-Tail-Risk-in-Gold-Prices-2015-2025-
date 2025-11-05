
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

## Extending the Analysis with Macroeconomic Drivers

### What was added?

To make it easier to combine the gold-risk notebook with the macro indicators listed
in the brief, the last update introduced two supporting pieces:

1. **`config/macro_series.yaml`** – a catalog that enumerates every data series, the
   data provider to query (Statistics Canada, Bank of Canada, FRED, or external Excel),
   and which transformation (e.g., log, log difference, per-capita scaling) should be
   applied after download. YAML is just a human-readable configuration format, so you
   can tweak items like the start date, series codes, or transformations without
   editing Python code. A simplified excerpt looks like:

   ```yaml
   series:
     - id: canada_cpi_inflation            # unique file name when saved to disk
       source: statscan                    # which API module to use
       table: 18-10-0004-01                # Statistics Canada table identifier
       extra_dimensions:
         Products and product groups: All-items
       transformation: log_diff            # convert to monthly inflation
   ```

   Each top-level key mirrors a field on the `SeriesConfig` dataclass inside
   `scripts/download_macro_data.py`, so the script can loop through the list and issue
   the correct API requests.

2. **`scripts/download_macro_data.py`** – a command-line helper that reads the YAML
   catalog, downloads every requested series, applies the indicated transformation,
   and saves one CSV file per indicator (by default under `data/raw/`). Think of it as
   the automated downloader you can run before estimating a VAR or joining the macro
   data to the gold returns.

### Getting the helper files

If you cloned this project before the helper was added, pull the latest
changes from your Git remote (for example `git pull origin main`). The
new configuration and downloader live inside the repository, so once you
update your local checkout you will see:

```
config/macro_series.yaml
scripts/download_macro_data.py
```

You do **not** need to reorganize your folders or start over—just make
sure your working tree is up to date. A quick sanity check is to run
`ls config` and `ls scripts` and confirm the files are present.

### Running the downloader

Example usage:

```bash
python scripts/download_macro_data.py \
  --config config/macro_series.yaml \
  --output data/raw
```

> **Tip:** Export your FRED API key as `FRED_API_KEY` to avoid rate limits and pass
`--force` if you need to refresh previously-downloaded CSVs.

### Results Summary
- 1-day 95% VaR ≈ -1.6%
- 1-day 99% VaR ≈ -2.6%
- Rolling volatility peaked during 2020–2021, aligning with global market stress.
