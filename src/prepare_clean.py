"""Convert the CLEAN washing-machine houses to 1-minute parquet files.

For each house: read the aggregate, washing-machine channel and Issues flag,
sort and de-duplicate timestamps, average to 1 minute, forward-fill gaps of
<= MAX_FILL_GAP_MIN minutes and flag the rest as outages. Writes
data/processed/clean_house{N}_1min.parquet and a one-row-per-house screening
table (results/clean_screening.csv) used to choose houses for Sections 2-3.

Run: python -m src.prepare_clean
"""
import pandas as pd

from src import config
from src.refit_io import CHANNELS, read_clean_house
from src.resample import MAX_FILL_GAP_MIN, fill_short_gaps, nan_runs, to_1min

SCREEN_ON_W = 20  # screening only; Section 2 derives the cycle rule properly


def processed_path(house):
    return config.PROCESSED / f"clean_house{house}_1min.parquet"


def prepare_house(house):
    ch = config.WASHING_MACHINE_CHANNEL[house]
    df = read_clean_house(house, CHANNELS)
    checks = {
        "sorted": bool(df.index.is_monotonic_increasing),
        "duplicate_ts": int(df.index.duplicated().sum()),
    }
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Sum of all nine IAMs: long runs of exactly 0 reveal the official
    # pipeline's zero-filling of gaps > 2 min (Murray et al., 2017).
    iams = [c for c in CHANNELS if c != "Aggregate"]
    df = pd.DataFrame({
        "aggregate": df["Aggregate"],
        "wm": df[ch],
        "issues": df["Issues"],
        "iam_sum": df[iams].sum(axis=1),
    })

    m = fill_short_gaps(to_1min(df), ["aggregate", "wm", "issues", "iam_sum"], MAX_FILL_GAP_MIN)
    m.to_parquet(processed_path(house))
    return m, checks


def screen(house, m, checks):
    ok = m[~m["outage"]]
    outages = nan_runs(m["outage"])
    return {
        "house": house,
        "wm_channel": config.WASHING_MACHINE_CHANNEL[house],
        "start": m.index[0].date(),
        "end": m.index[-1].date(),
        "span_days": round(len(m) / 1440, 1),
        **checks,
        "observed_pct": round(100 * (m["n_samples"] > 0).mean(), 2),
        "filled_pct": round(100 * m["filled"].mean(), 3),
        "outage_pct": round(100 * m["outage"].mean(), 2),
        "n_outages": len(outages),
        "longest_outage_days": round(outages["minutes"].max() / 1440, 1) if len(outages) else 0,
        "issues_pct": round(100 * (ok["issues"] > 0).mean(), 2),
        "agg_p99_w": round(ok["aggregate"].quantile(0.99)),
        "agg_max_w": round(ok["aggregate"].max()),
        "wm_max_w": round(ok["wm"].max()),
        "wm_on_pct": round(100 * (ok["wm"] > SCREEN_ON_W).mean(), 2),
        "wm_kwh_per_day": round(ok["wm"].mean() * 24 / 1000, 3),
        "wm_share_of_agg_pct": round(100 * ok["wm"].sum() / ok["aggregate"].sum(), 2),
    }


def main():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for house in config.WASHING_MACHINE_CHANNEL:
        m, checks = prepare_house(house)
        rows.append(screen(house, m, checks))
        print(f"House {house:>2}: {len(m):,} minutes")
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "clean_screening.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
