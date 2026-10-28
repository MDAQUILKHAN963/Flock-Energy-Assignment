"""1-minute resampling and gap handling shared by every section."""
import numpy as np
import pandas as pd

# Gaps up to this length (minutes) are forward-filled; longer ones are flagged
# as outages. In the CLEAN houses, 50-80 % of gaps are <= 5 min but they hold
# < 0.2 % of minutes, while gaps > 60 min hold ~95 % of all missing time.
# 5 min is also short relative to the phases of a wash cycle (heating, wash,
# rinse, spin each last several minutes), so filling cannot invent a cycle.
MAX_FILL_GAP_MIN = 5


def to_1min(df):
    """Mean of each column per minute, plus `n_samples` (readings in that minute).

    Minutes with no readings come out as NaN, so outages stay visible.
    The index is regularised to a complete 1-minute grid.
    """
    out = df.resample("1min").mean()
    out["n_samples"] = df.iloc[:, 0].resample("1min").count()
    return out


def nan_runs(mask):
    """Start, end and length (in rows) of each run of True values in a boolean Series."""
    m = mask.to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    return pd.DataFrame({
        "start": mask.index[starts],
        "end": mask.index[ends - 1],
        "minutes": ends - starts,
    })


def fill_short_gaps(df, columns, max_gap_min):
    """Forward-fill gaps of at most `max_gap_min` minutes; leave longer gaps as NaN.

    A gap is a run of minutes where the first listed column is missing. Forward
    fill matches how REFIT logs data (a reading is only written when the load
    changes), so carrying the last value over a short gap is the least-invented
    choice. Adds `filled` (bool) and `outage` (bool) columns.
    """
    out = df.copy()
    missing = out[columns[0]].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    fill_mask = missing & (run_len <= max_gap_min)

    ffilled = out[columns].ffill()
    out.loc[fill_mask, columns] = ffilled.loc[fill_mask, columns]
    out["filled"] = fill_mask & out[columns[0]].notna()
    out["outage"] = out[columns[0]].isna()
    return out
