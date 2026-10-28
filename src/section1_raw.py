"""Section 1: inspect and clean RAW House 1, resampled to 1 minute.

Pipeline (each step is logged with row counts in results/s1_cleaning_log.csv):
  1. Put every timestamp on UTC. Part1 `Unix` encodes UK wall-clock time
     (BST in summer); Part2 is UTC from Oct 2014, which is the only Part2 data
     kept (clock_check / DATA.md). The repeated clock-change hour is dropped.
  2. Resolve the Part1/Part2 overlap (Jun-Oct 2014, recorded in both): keep
     Part1 until it ends, then Part2. Part1 is the version the official CLEAN
     data was built from in this period (overlap_check / DATA.md).
  3. Sort by Unix timestamp (stable sort).
  4. Set physically impossible readings to NaN.
  5. Remove duplicates: exact duplicate rows are dropped; for a timestamp that
     still has several rows, keep the last non-missing reading per channel.
  6. Average to 1-minute intervals (minutes without readings stay NaN).
  7. Per channel, forward-fill gaps <= MAX_FILL_GAP_MIN minutes; flag the rest.

Run: python -m src.section1_raw
"""
import numpy as np
import pandas as pd

from src import config, plotting
from src.plotting import BLUE, MUTED, TEXT_2, plt
from src.refit_io import CHANNELS, raw_house_files, read_clean_house, read_raw_part
from src.resample import MAX_FILL_GAP_MIN, fill_short_gaps, nan_runs, to_1min

HOUSE = 1
AGG_MAX_W = 23_000  # 100 A main fuse x 230 V: the most a UK home can draw
IAM_MAX_W = 4_000   # official REFIT spike threshold for appliance monitors
IAMS = [c for c in CHANNELS if c != "Aggregate"]
PLOT_CHANNELS = ["Aggregate", "Appliance5", "Appliance6"]  # + washing machine, dishwasher


def inspect_part(df, name):
    """Quality checks on one part file, before any cleaning."""
    unix = df["Unix"]
    step = unix.diff()
    parsed = pd.to_datetime(df["Time"], errors="coerce")
    text_vs_unix = (parsed - pd.to_datetime(unix, unit="s")).dt.total_seconds().abs()
    no_text = df.drop(columns="Time")
    distinct = no_text.drop_duplicates()
    readings = no_text.drop_duplicates(subset="Unix")
    interval = readings["Unix"].diff()

    row = {
        "part": name,
        "rows": len(df),
        "first": pd.to_datetime(unix.min(), unit="s"),
        "last": pd.to_datetime(unix.max(), unit="s"),
        "time_unparseable": int(parsed.isna().sum()),
        "time_text_differs_from_unix_gt1s": int((text_vs_unix > 1).sum()),
        "backward_steps": int((step < 0).sum()),
        "exact_duplicate_rows": int(no_text.duplicated().sum()),
        "duplicate_timestamps": int(unix.duplicated().sum()),
        "duplicate_timestamps_conflicting": int(distinct["Unix"].duplicated().sum()),
        "median_interval_s": float(interval.median()),
        "p99_interval_s": float(interval.quantile(0.99)),
        "gaps_gt_5min": int((interval > 300).sum()),
        "gaps_gt_1h": int((interval > 3600).sum()),
        "negative_values": int((df[CHANNELS] < 0).sum().sum()),
        "aggregate_gt_23kW": int((df["Aggregate"] > AGG_MAX_W).sum()),
        "appliance_gt_4kW": int((df[IAMS] > IAM_MAX_W).sum().sum()),
    }
    for c in CHANNELS:
        row[f"nan_pct_{c}"] = round(100 * df[c].isna().mean(), 2)
        row[f"zero_pct_{c}"] = round(100 * (df[c] == 0).mean(), 2)
        row[f"max_{c}"] = df[c].max()
    return row


def as_series_frame(df):
    """Rows indexed by naive (wall-clock) timestamp, channels only."""
    out = df[CHANNELS].copy()
    out.index = pd.to_datetime(df["Unix"], unit="s")
    out.index.name = "time"
    return out


def part1_to_utc(part1):
    """Part1 `Unix` encodes UK wall-clock time; convert it to true UTC seconds.

    The hour repeated when clocks go back (27 Oct 2013 01:00-02:00) cannot be
    assigned to BST or GMT reliably, so those readings are dropped.
    """
    local = pd.to_datetime(part1["Unix"], unit="s").dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="NaT")
    keep = local.notna()
    out = part1[keep].copy()
    out["Unix"] = (local[keep].dt.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
    return out, int((~keep).sum())


def clock_check(part1, part2):
    """Per month and part: does RAW line up with the official (UTC) data as
    UTC or as UK local time? Evidence for step 1 of the pipeline."""
    official = read_clean_house(HOUSE, ["Aggregate"])["Aggregate"]
    as_utc = official.tz_localize(None).resample("1min").mean()
    as_local = official.tz_convert("Europe/London").tz_localize(None).resample("1min").mean()
    rows = []
    for name, part in [("Part1", part1), ("Part2", part2)]:
        s = as_series_frame(part.drop_duplicates("Unix", keep="last"))["Aggregate"]
        s = s.where(s <= AGG_MAX_W).resample("1min").mean()
        for month, o in s.groupby(s.index.to_period("M")):
            if o.notna().sum() < 1440:
                continue
            utc, local = o.corr(as_utc.reindex(o.index)), o.corr(as_local.reindex(o.index))
            verdict = "same (GMT)" if abs(utc - local) < 0.05 else ("UTC" if utc > local else "UK local")
            rows.append({"part": name, "month": str(month), "corr_if_utc": round(utc, 3),
                         "corr_if_uk_local": round(local, 3), "clock": verdict})
    return pd.DataFrame(rows)


def clean(part1, part2):
    log = []

    def note(step, rows, detail=""):
        log.append({"step": step, "rows_after": rows, "detail": detail})

    note("load", len(part1) + len(part2), f"Part1 {len(part1):,} + Part2 {len(part2):,} rows")

    part1, ambiguous = part1_to_utc(part1)
    note("timestamps to UTC", len(part1) + len(part2),
         f"Part1 converted from UK local time; {ambiguous:,} readings in the repeated "
         f"clock-change hour dropped")

    cut = part1["Unix"].max()
    dropped = int((part2["Unix"] <= cut).sum())
    part2 = part2[part2["Unix"] > cut]
    df = pd.concat([part1, part2], ignore_index=True)
    note("resolve overlap", len(df),
         f"dropped {dropped:,} Part2 rows up to {pd.to_datetime(cut, unit='s')} (Part1 kept there)")

    df = df.sort_values("Unix", kind="stable", ignore_index=True)
    note("sort", len(df), "stable sort on Unix timestamp")

    agg_bad = df["Aggregate"].gt(AGG_MAX_W) | df["Aggregate"].lt(0)
    iam_bad = df[IAMS].gt(IAM_MAX_W) | df[IAMS].lt(0)
    df.loc[agg_bad, "Aggregate"] = np.nan
    df[IAMS] = df[IAMS].mask(iam_bad)
    note("impossible values -> NaN", len(df),
         f"{int(agg_bad.sum()):,} aggregate > {AGG_MAX_W:,} W; "
         f"{int(iam_bad.sum().sum()):,} appliance > {IAM_MAX_W:,} W; negatives none")

    n = len(df)
    df = df.drop(columns="Time").drop_duplicates()
    note("drop exact duplicates", len(df), f"{n - len(df):,} identical rows removed")

    n = len(df)
    df = df.groupby("Unix", sort=True)[CHANNELS].last().reset_index()
    note("one row per timestamp", len(df),
         f"{n - len(df):,} rows merged: last non-missing reading per channel kept")

    m = to_1min(as_series_frame(df))
    note("resample to 1 min", len(m), f"{m.index[0]} to {m.index[-1]}")

    filled = {}
    for c in CHANNELS:
        f = fill_short_gaps(m[[c]], [c], MAX_FILL_GAP_MIN)
        m[c] = f[c]
        filled[c] = int(f["filled"].sum())
        if c == "Aggregate":
            m["filled"], m["outage"] = f["filled"], f["outage"]
    note("fill short gaps", len(m),
         f"forward-filled gaps <= {MAX_FILL_GAP_MIN} min; minutes filled: Aggregate "
         f"{filled['Aggregate']:,}, Appliance5 {filled['Appliance5']:,}; "
         f"aggregate outage minutes left as NaN: {int(m['outage'].sum()):,}")
    return m, pd.DataFrame(log)


def overlap_check(part1, part2):
    """Which part did the official CLEAN data use where both parts overlap?

    CLEAN timestamps are UTC; RAW ones are UK wall-clock, so CLEAN is converted
    to Europe/London before comparing 1-minute means.
    """
    lo, hi = part2["Unix"].min(), part1["Unix"].max()
    official = read_clean_house(HOUSE, ["Aggregate", "Appliance5"])
    official.index = official.index.tz_convert("Europe/London").tz_localize(None)

    def one_min(part, c):
        p = part[part["Unix"].between(lo, hi)].drop_duplicates("Unix", keep="last")
        s = as_series_frame(p)[c]
        return s.where(s <= (AGG_MAX_W if c == "Aggregate" else IAM_MAX_W)).resample("1min").mean()

    rows = []
    for c in ["Aggregate", "Appliance5"]:
        j = pd.concat([one_min(part1, c), one_min(part2, c), official[c].resample("1min").mean()],
                      axis=1, keys=["Part1", "Part2", "official"]).dropna()
        for k in ["Part1", "Part2"]:
            rows.append({
                "channel": config.HOUSE1_CHANNELS[c],
                "part": k,
                "minutes_compared": len(j),
                "mae_vs_official_w": round((j[k] - j["official"]).abs().mean(), 1),
                "corr_vs_official": round(j[k].corr(j["official"]), 3),
                "minutes_within_1w_pct": round(100 * np.isclose(j[k], j["official"], atol=1).mean(), 1),
            })
    return pd.DataFrame(rows)


def agreement_with_official(after):
    """End-to-end check: our cleaned 1-min series vs the official CLEAN file."""
    channels = ["Aggregate", "Appliance1", "Appliance5", "Appliance6"]
    official = read_clean_house(HOUSE, channels)
    official.index = official.index.tz_localize(None)  # both are UTC now
    official = official[channels].resample("1min").mean()
    rows = []
    for c in channels:
        j = pd.concat([after[c], official[c]], axis=1, keys=["ours", "official"]).dropna()
        rows.append({
            "channel": config.HOUSE1_CHANNELS[c],
            "minutes_compared": len(j),
            "mae_w": round((j["ours"] - j["official"]).abs().mean(), 2),
            "corr": round(j["ours"].corr(j["official"]), 4),
            "minutes_within_1w_pct": round(100 * np.isclose(j["ours"], j["official"], atol=1).mean(), 1),
        })
    return pd.DataFrame(rows)


def minute_summary(m):
    rows = []
    for c in CHANNELS:
        s = m[c]
        rows.append({
            "channel": c,
            "label": config.HOUSE1_CHANNELS[c],
            "missing_pct": round(100 * s.isna().mean(), 2),
            "zero_pct": round(100 * (s == 0).mean(), 2),
            "mean_w": round(s.mean(), 1),
            "p99_w": round(s.quantile(0.99), 1),
            "max_w": round(s.max(), 1),
        })
    return pd.DataFrame(rows)


def plot_overview(before, after, outages):
    """Hourly maximum over the whole period: spikes and outages are visible at this scale."""
    fig, axes = plt.subplots(len(PLOT_CHANNELS), 2, figsize=(13, 7.5), sharex=True)
    long_outages = outages[outages["minutes"] >= 60]
    spans = list(zip(long_outages["start"], long_outages["end"]))
    for row, c in enumerate(PLOT_CHANNELS):
        b = before[c].resample("1h").max()
        a = after[c].resample("1h").max()
        for col, (s, colour, title) in enumerate([(b, MUTED, "before"), (a, BLUE, "after")]):
            ax = axes[row, col]
            if col == 1:
                plotting.shade_spans(ax, spans, "outage > 1 h (left missing)")
            ax.plot(s.index, s.to_numpy(), color=colour, lw=0.6)
            ax.set_title(f"{config.HOUSE1_CHANNELS[c]} - {title} cleaning", loc="left")
            ax.set_ylabel("hourly max (W)")
            ax.ticklabel_format(axis="y", style="plain")
    axes[0, 1].legend(loc="upper right")
    fig.suptitle("Raw House 1: hourly maximum power, as recorded (left) vs cleaned 1-min (right)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT)
    fig.tight_layout()
    return plotting.save(fig, "s1_overview_before_after.png")


def plot_zoom(before, after, start, end, name, title):
    """A short window at native resolution (before) vs 1-minute cleaned (after)."""
    fig, axes = plt.subplots(len(PLOT_CHANNELS), 1, figsize=(13, 7.5), sharex=True)
    for ax, c in zip(axes, PLOT_CHANNELS):
        b = before[c].loc[start:end]
        a = after[c].loc[start:end]
        out = after["outage"].loc[start:end]
        plotting.shade_spans(ax, list(zip(*[nan_runs(out)[k] for k in ("start", "end")])), "outage (not filled)")
        filled = after["filled"].loc[start:end]
        # Cap the axis so spikes do not flatten the real signal; mark them at the top edge.
        top = max(a.max(), 100) * 1.3
        off = b[b > top]
        ax.plot(b.index, b.clip(upper=top).to_numpy(), ".", color=MUTED, ms=2.5,
                label="raw readings (both parts, as recorded)")
        ax.plot(a.index, a.to_numpy(), color=BLUE, lw=1.4, label="cleaned, 1-min mean")
        if filled.any():
            ax.plot(a[filled].index, a[filled].to_numpy(), "o", color=plotting.ORANGE, ms=4,
                    label=f"filled gap (<= {MAX_FILL_GAP_MIN} min)")
        if len(off):
            ax.plot(off.index, np.full(len(off), top), "^", color=TEXT_2, ms=6, clip_on=False)
            ax.annotate(f"{len(off)} raw reading(s) off-scale, up to {off.max():,.0f} W (removed)",
                        (off.index[0], top), xytext=(6, -12), textcoords="offset points",
                        fontsize=8, color=TEXT_2)
        ax.set_ylim(0, top)
        ax.set_title(config.HOUSE1_CHANNELS[c], loc="left")
        ax.set_ylabel("W")
    axes[0].legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("time (UTC)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT)
    fig.tight_layout()
    return plotting.save(fig, name)


def main():
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    f1, f2 = raw_house_files(HOUSE)
    part1, part2 = read_raw_part(f1), read_raw_part(f2)

    inspection = pd.DataFrame([inspect_part(part1, f1.stem), inspect_part(part2, f2.stem)])
    inspection.T.to_csv(config.RESULTS / "s1_inspection.csv", header=False)
    overlap = overlap_check(part1, part2)
    overlap.to_csv(config.RESULTS / "s1_overlap_check.csv", index=False)
    clocks = clock_check(part1, part2)
    clocks.to_csv(config.RESULTS / "s1_clock_check.csv", index=False)

    after, log = clean(part1, part2)
    log.to_csv(config.RESULTS / "s1_cleaning_log.csv", index=False)
    after.to_parquet(config.PROCESSED / "raw_house1_clean_1min.parquet")

    outages = nan_runs(after["outage"])
    outages = outages[outages["minutes"] > MAX_FILL_GAP_MIN].reset_index(drop=True)
    outages.to_csv(config.RESULTS / "s1_outages.csv", index=False)
    minute_summary(after).to_csv(config.RESULTS / "s1_cleaned_summary.csv", index=False)
    agreement = agreement_with_official(after)
    agreement.to_csv(config.RESULTS / "s1_agreement_with_official.csv", index=False)

    # "Before" = every reading as recorded, only placed on the same UTC clock.
    before = as_series_frame(pd.concat([part1_to_utc(part1)[0], part2]).sort_values("Unix", kind="stable"))
    plot_overview(before, after, outages)
    for start, end, name, title in ZOOMS:
        plot_zoom(before, after, start, end, name, title)

    print(inspection.T.to_string(header=False))
    print(overlap.to_string(index=False))
    print(clocks.groupby(["part", "clock"])["month"].agg(["count", "min", "max"]).to_string())
    print(log.to_string(index=False))
    print(agreement.to_string(index=False))
    print(f"outages > {MAX_FILL_GAP_MIN} min: {len(outages)}, "
          f"{outages['minutes'].sum() / 1440:.1f} days in total")


# Windows chosen to show each cleaning rule at work (see DATA.md).
ZOOMS = [
    ("2013-12-31", "2013-12-31 23:59", "s1_zoom_gaps_spikes.png",
     "Raw House 1, 31 Dec 2013: spikes removed, short gaps filled, outage left missing"),
    ("2014-07-27", "2014-07-27 23:59", "s1_zoom_overlap.png",
     "Raw House 1, 27 Jul 2014: overlap period - Part2's copy (UK local time) shows 1 h late; "
     "cleaned keeps Part1"),
]

if __name__ == "__main__":
    main()
