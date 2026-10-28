"""Readers for the REFIT CSV files.

Timestamps are taken from the `Unix` column (seconds) and kept in UTC;
conversion to local time is done where it matters (hour-of-day analysis).
"""
import pandas as pd

from src import config

CHANNELS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]


def raw_house_files(house):
    """RAW files for a house, in Part order. Houses 13+ use a `House_N` prefix."""
    name = f"RAW_House{house}" if house < 13 else f"RAW_House_{house}"
    files = sorted(config.RAW_DIR.glob(f"{name}_Part*.csv"))
    if not files:
        raise FileNotFoundError(f"No RAW files for House {house} in {config.RAW_DIR}")
    return files


def read_raw_part(path):
    """One RAW part file, exactly as recorded (no sorting or de-duplication).

    `Time` is read as text so that parsing problems can be inspected rather
    than silently coerced.
    """
    dtypes = {"Time": "string", "Unix": "int64", **{c: "float64" for c in CHANNELS}}
    return pd.read_csv(path, dtype=dtypes, engine="pyarrow")


def read_clean_house(house, channels):
    """Selected channels of a CLEAN file, indexed by UTC timestamp."""
    path = config.CLEAN_DIR / f"CLEAN_House{house}.csv"
    df = pd.read_csv(path, usecols=["Unix", *channels, "Issues"], engine="pyarrow")
    df.index = pd.to_datetime(df.pop("Unix"), unit="s", utc=True)
    df.index.name = "time"
    return df
