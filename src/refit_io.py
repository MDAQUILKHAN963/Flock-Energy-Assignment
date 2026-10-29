"""Readers for the REFIT CSV files.

Timestamps are taken from the `Unix` column (seconds) and kept in UTC;
conversion to local time is done where it matters (hour-of-day analysis).
"""
import pandas as pd

from src import config

CHANNELS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]


def _find(directory, pattern):
    """Files matching `pattern` in `directory`, or in a nested folder if the
    archive was extracted with an extra level."""
    return sorted(directory.glob(pattern)) or sorted(directory.rglob(pattern))


def raw_house_files(house):
    """RAW files for a house, in Part order. Houses 13+ use a `House_N` prefix."""
    name = f"RAW_House{house}" if house < 13 else f"RAW_House_{house}"
    files = _find(config.RAW_DIR, f"{name}_Part*.csv")
    if not files:
        raise FileNotFoundError(f"No RAW files for House {house} in {config.RAW_DIR}")
    return files


def clean_house_file(house):
    files = _find(config.CLEAN_DIR, f"CLEAN_House{house}.csv")
    if not files:
        raise FileNotFoundError(f"No CLEAN file for House {house} in {config.CLEAN_DIR}")
    return files[0]


def read_raw_part(path):
    """One RAW part file, exactly as recorded (no sorting or de-duplication).

    `Time` is read as text so that parsing problems can be inspected rather
    than silently coerced.
    """
    dtypes = {"Time": "string", "Unix": "int64", **{c: "float64" for c in CHANNELS}}
    return pd.read_csv(path, dtype=dtypes, engine="pyarrow")


def read_clean_house(house, channels):
    """Selected channels of a CLEAN file, indexed by UTC timestamp."""
    df = pd.read_csv(clean_house_file(house), usecols=["Unix", *channels, "Issues"], engine="pyarrow")
    df.index = pd.to_datetime(df.pop("Unix"), unit="s", utc=True)
    df.index.name = "time"
    return df
