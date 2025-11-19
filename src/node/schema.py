from __future__ import annotations

from typing import Dict, Iterable, List

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is expected in runtime env
    pd = None  # type: ignore

# Canonical column names for node-level signals
NODE_SIGNAL_ALIASES: Dict[str, List[str]] = {
    "rhoH_2020": ["rhoH_2020", "rho_H_2020", "rhoH"],
    "rhoH_2021": ["rhoH_2021", "rho_H_2021"],
    "rhoW_2020": ["rhoW_2020", "rho_W_2020", "WAC_2020", "WAC"],
    "rhoW_2021": ["rhoW_2021", "rho_W_2021", "WAC_2021"],
    "In_2020": ["In_2020", "In"],
    "In_2021": ["In_2021"],
    "lat": ["lat", "LAT", "latitude"],
    "lon": ["lon", "LON", "longitude"],
    "county": ["county", "COUNTY", "county_fips"],
}

REQUIRED_NODE_COLUMNS = (
    "rhoH_2020",
    "rhoH_2021",
    "rhoW_2020",
    "rhoW_2021",
    "lat",
    "lon",
    "county",
)


def _build_rename_map(columns: Iterable[str]) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    for target, candidates in NODE_SIGNAL_ALIASES.items():
        for cand in candidates:
            if cand in columns:
                rename[cand] = target
                break
    return rename


def normalize_node_signals(df):
    """Rename node signal columns to canonical names and validate presence."""
    if pd is None:
        raise ImportError("pandas is required to normalize node signals.")
    rename = _build_rename_map(df.columns)
    df = df.rename(columns=rename)
    missing = [col for col in REQUIRED_NODE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required node columns: {missing}")
    df["county"] = df["county"].astype(str)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df
