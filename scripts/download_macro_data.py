"""Utility to download macroeconomic indicators used alongside the gold VaR study.

The script supports three main providers:
- Statistics Canada (REST API)
- Bank of Canada Valet API
- Federal Reserve Economic Data (FRED)

Usage
-----
python scripts/download_macro_data.py --config config/macro_series.yaml --output data/raw

Notes
-----
* A FRED API key can be supplied through the ``FRED_API_KEY`` environment variable.
* Some Statistics Canada tables are large; the helper functions filter to the requested
  geography and series after download to keep the client code simple.
* The script does not overwrite existing CSV files unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - configuration error
    raise SystemExit(
        "PyYAML is required. Install it with `pip install pyyaml` before running this script."
    ) from exc

try:  # Prefer requests when available for cleaner HTTPS handling.
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

import numpy as np
import pandas as pd
from pandas import DataFrame
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

STATSCAN_BASE = "https://proxy-apicdn.statcan.gc.ca/rest"
BOC_BASE = "https://www.bankofcanada.ca/valet"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


@dataclass
class SeriesConfig:
    """Structured metadata for a single macro series entry from the YAML catalog."""
    id: str
    source: str
    description: str
    frequency: str
    start_date: str
    table: Optional[str] = None
    series: Optional[str] = None
    value_column: Optional[str] = None
    date_column: Optional[str] = None
    geography: Optional[str] = None
    extra_dimensions: Dict[str, str] = field(default_factory=dict)
    transformation: Optional[str] = None
    aggregation: Optional[str] = None
    population_table: Optional[str] = None
    cpi_table: Optional[str] = None
    price_index_table: Optional[str] = None
    price_index_item: Optional[str] = None
    sheet_name: Optional[str] = None
    url: Optional[str] = None

    def output_name(self) -> str:
        return f"{self.id}.csv"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fetch JSON either via requests (when available) or urllib."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; data-collection-script/1.0)"}
    if requests is not None:  # pragma: no cover - requests unavailable in CI
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as resp:  # type: ignore[arg-type]
        return json.load(resp)


def http_get_csv(url: str, params: Optional[Dict[str, Any]] = None) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; data-collection-script/1.0)"}
    if requests is not None:  # pragma: no cover
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        return response.content
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url, headers=headers)
    with urlopen(req, timeout=60) as resp:  # type: ignore[arg-type]
        return resp.read()


# ---------------------------------------------------------------------------
# Statistics Canada helpers
# ---------------------------------------------------------------------------

def fetch_statscan_data(cfg: SeriesConfig) -> DataFrame:
    if not cfg.table:
        raise ValueError(f"Statistics Canada series '{cfg.id}' is missing a table id")

    params = {"startDate": cfg.start_date}
    url = f"{STATSCAN_BASE}/getDataTable/v1/en/{cfg.table}"
    try:
        payload = http_get_json(url, params=params)
    except HTTPError as err:  # pragma: no cover - network failure
        raise RuntimeError(f"Failed to download table {cfg.table} from Statistics Canada: {err}")

    data = payload.get("data", [])
    if not data:
        raise RuntimeError(f"No data returned for Statistics Canada table {cfg.table}")

    structure = payload.get("structure", {})
    obs_dims = structure.get("dimensions", {}).get("observation", [])
    dim_labels: List[str] = [dim.get("name", f"dim_{i}") for i, dim in enumerate(obs_dims)]

    records: List[Dict[str, Any]] = []
    categories = {dim.get("name"): dim.get("values", []) for dim in obs_dims}

    for entry in data:
        record = {
            "value": entry.get("value"),
            "REF_DATE": entry.get("REF_DATE"),
        }
        coords = entry.get("coordinate")
        if coords and categories:
            indices = [int(x) for x in coords.split(":")]
            for idx, dim_name in zip(indices, dim_labels):
                value_meta = categories.get(dim_name)
                if value_meta and idx < len(value_meta):
                    record[dim_name] = value_meta[idx].get("name")
        for attr in ("UOM", "SCALAR_FACTOR", "SCALAR_ID", "STATUS", "SYMBOL"):
            if attr in entry:
                record[attr] = entry[attr]
        records.append(record)

    df = pd.DataFrame.from_records(records)

    # Filter by requested dimension labels if provided
    for dim_name, dim_value in (cfg.extra_dimensions or {}).items():
        if dim_name in df.columns:
            df = df[df[dim_name] == dim_value]
        else:
            # Allow partial matches when StatsCan uses slightly different labels.
            matching_cols = [col for col in df.columns if dim_name.lower() in col.lower()]
            if matching_cols:
                col = matching_cols[0]
                df = df[df[col] == dim_value]

    if cfg.geography and "Geography" in df.columns:
        df = df[df["Geography"] == cfg.geography]

    if cfg.value_column and cfg.value_column in df.columns:
        df.rename(columns={cfg.value_column: "value"}, inplace=True)

    df = df[[col for col in df.columns if col in {"REF_DATE", "value"}]].copy()
    df.dropna(subset=["value"], inplace=True)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df.sort_values("REF_DATE", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Bank of Canada helpers
# ---------------------------------------------------------------------------

def fetch_boc_series(cfg: SeriesConfig) -> DataFrame:
    if not cfg.series:
        raise ValueError(f"Bank of Canada series '{cfg.id}' requires a series code")

    params = {"start_date": cfg.start_date}
    url = f"{BOC_BASE}/observations/{cfg.series}"
    payload = http_get_json(url, params=params)
    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(f"No data returned for Bank of Canada series {cfg.series}")
    df = pd.DataFrame(observations)
    df.rename(columns={"d": "REF_DATE", cfg.series: "value"}, inplace=True)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["value"], inplace=True)
    df.sort_values("REF_DATE", inplace=True)
    if cfg.aggregation == "mean":
        df = (
            df.assign(month=lambda d: d["REF_DATE"].dt.to_period("M"))
              .groupby("month", as_index=False)["value"].mean()
              .rename(columns={"month": "REF_DATE"})
        )
        df["REF_DATE"] = df["REF_DATE"].dt.to_timestamp()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def fetch_fred_series(cfg: SeriesConfig, api_key: Optional[str]) -> DataFrame:
    if not cfg.series:
        raise ValueError(f"FRED series '{cfg.id}' requires a series code")
    params = {
        "series_id": cfg.series,
        "file_type": "json",
        "observation_start": cfg.start_date,
        "frequency": cfg.frequency.lower(),
    }
    if api_key:
        params["api_key"] = api_key
    payload = http_get_json(FRED_BASE, params=params)
    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(f"No observations returned for FRED series {cfg.series}")
    df = pd.DataFrame(observations)
    df.rename(columns={"date": "REF_DATE", "value": "value"}, inplace=True)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(subset=["value"], inplace=True)
    df.sort_values("REF_DATE", inplace=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def apply_transformation(df: DataFrame, cfg: SeriesConfig) -> DataFrame:
    if not cfg.transformation:
        return df

    transformed = df.copy()
    transformed.set_index("REF_DATE", inplace=True)

    transformed["value"] = transformed["value"].astype(float)

    if cfg.transformation == "log":
        transformed = transformed[transformed["value"] > 0]
        transformed["value"] = np.log(transformed["value"])
    elif cfg.transformation == "log_diff":
        transformed = transformed[transformed["value"] > 0]
        transformed["value"] = np.log(transformed["value"])
        transformed["value"] = transformed["value"].diff()
        transformed.dropna(inplace=True)
    elif cfg.transformation == "log_per_capita":
        raise NotImplementedError(
            "Population-adjusted transformations require join logic."
        )
    elif cfg.transformation in {"real_log", "log_real"}:
        raise NotImplementedError(
            "Price deflation is not yet implemented in this script."
        )
    elif cfg.transformation == "log_real_index":
        transformed = transformed[transformed["value"] > 0]
        transformed["value"] = np.log(transformed["value"])
    elif cfg.transformation == "level":
        pass
    else:
        raise ValueError(f"Unsupported transformation '{cfg.transformation}' for series {cfg.id}")

    transformed.reset_index(inplace=True)
    return transformed


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def load_config(path: Path) -> List[SeriesConfig]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return [SeriesConfig(**entry) for entry in payload.get("series", [])]


def download_series(cfg: SeriesConfig, output_dir: Path, fred_key: Optional[str], force: bool) -> Path:
    output_path = output_dir / cfg.output_name()
    if output_path.exists() and not force:
        print(f"Skipping {cfg.id}: {output_path.name} already exists")
        return output_path

    if cfg.source == "statscan":
        df = fetch_statscan_data(cfg)
    elif cfg.source == "boc":
        df = fetch_boc_series(cfg)
    elif cfg.source == "fred":
        df = fetch_fred_series(cfg, api_key=fred_key)
    elif cfg.source == "external":
        if not cfg.url:
            raise ValueError(f"External series '{cfg.id}' requires a download URL")
        content = http_get_csv(cfg.url)
        df = pd.read_excel(BytesIO(content), sheet_name=cfg.sheet_name)
        df.rename(columns={df.columns[0]: "REF_DATE", df.columns[1]: "value"}, inplace=True)
        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
        df.sort_values("REF_DATE", inplace=True)
    else:
        raise ValueError(f"Unknown data source '{cfg.source}' for series {cfg.id}")

    df = apply_transformation(df, cfg)
    df.to_csv(output_path, index=False)
    print(f"Saved {cfg.id} → {output_path.relative_to(output_dir.parent)}")
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/macro_series.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV files")
    args = parser.parse_args(argv)

    cfg_path: Path = args.config
    if not cfg_path.exists():
        parser.error(f"Configuration file not found: {cfg_path}")

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    fred_key = os.getenv("FRED_API_KEY")
    configs = load_config(cfg_path)
    if not configs:
        parser.error(f"No series definitions found in {cfg_path}")

    for cfg in configs:
        try:
            download_series(cfg, output_dir=output_dir, fred_key=fred_key, force=args.force)
        except Exception as exc:  # pragma: no cover - runtime diagnostic aid
            print(f"Failed to download {cfg.id}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
