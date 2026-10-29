"""Section 2: washing-machine use in the official CLEAN data (1-minute, 14 houses).

Cycle rule, in plain words:
  * the machine is ON in a minute when it draws more than ON_W (20 W); standby is
    below 10 W in every house, so 20 W separates "doing something" from idle;
  * ON minutes separated by pauses of at most MERGE_GAP_MIN (15 min) belong to the
    same cycle: a wash pauses while filling, soaking and draining;
  * an event counts as a wash cycle if it lasts >= MIN_CYCLE_MIN (10 min) and uses
    >= MIN_CYCLE_KWH (0.05 kWh); shorter/smaller events are display or door-lock blips.
Outages and frozen readings are missing, never "off". A cycle that touches missing
data is counted but flagged `truncated` and left out of duration/energy comparisons.

Run: python -m src.section2_eda   (after python -m src.prepare_clean)
"""
import numpy as np
import pandas as pd

from src import config, plotting
from src.plotting import BLUE, LIGHT_BLUE, MUTED, ORANGE, SEQ_CMAP, TEXT_2, plt
from src.prepare_clean import processed_path
from src.resample import MAX_FILL_GAP_MIN, nan_runs

ON_W = 20
MERGE_GAP_MIN = 15
MIN_CYCLE_MIN = 10
MIN_CYCLE_KWH = 0.05
STUCK_MIN = 30      # identical reading above ON_W for this long = frozen sensor
LOCAL_TZ = "Europe/London"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def frozen(w):
    """True where a 1-minute appliance reading above ON_W repeats exactly for
    STUCK_MIN+ minutes: a monitor that stopped reporting, forward-filled by CLEAN."""
    same = w.diff().eq(0)
    run_id = (~same).cumsum()
    run_len = w.groupby(run_id).transform("size")
    return (run_len >= STUCK_MIN) & (w > ON_W)


def load_house(house):
    """1-minute data with the washing-machine channel's missing minutes applied.

    `wm` is NaN during outages and frozen-sensor runs (`stuck`).
    """
    d = pd.read_parquet(processed_path(house))
    w = d["wm"].where(~d["outage"])
    d["stuck"] = frozen(w)
    d["wm"] = w.where(~d["stuck"])
    return d


def detect_events(w, merge_gap=MERGE_GAP_MIN, min_min=MIN_CYCLE_MIN, min_kwh=MIN_CYCLE_KWH):
    """Group ON minutes into events and classify them as cycles or blips."""
    v = w.to_numpy(dtype=float)
    missing = np.isnan(v)
    pos = np.flatnonzero(np.nan_to_num(v, nan=0.0) > ON_W)
    cols = ["start", "end", "minutes", "kwh", "peak_w", "truncated", "is_cycle"]
    if len(pos) == 0:
        return pd.DataFrame(columns=cols)

    missing_before = np.r_[0, np.cumsum(missing)]
    pause = np.diff(pos) - 1
    missing_between = missing_before[pos[1:]] - missing_before[pos[:-1] + 1]
    split = (pause > merge_gap) | (missing_between > 0)   # never merge across missing data
    starts = pos[np.r_[True, split]]
    ends = pos[np.r_[split, True]]

    rows = []
    for s, e in zip(starts, ends):
        seg = v[s:e + 1]
        lo, hi = s - merge_gap, e + merge_gap + 1
        truncated = lo < 0 or hi > len(v) or bool(missing[lo:hi].any())
        kwh = seg.sum() / 60 / 1000
        minutes = e - s + 1
        rows.append((w.index[s], w.index[e], minutes, kwh, seg.max(), truncated,
                     minutes >= min_min and kwh >= min_kwh))
    return pd.DataFrame(rows, columns=cols)


def availability(house, d, events):
    cyc = events[events["is_cycle"]]
    obs = d["wm"].notna()
    w = d["wm"][obs]
    in_cycle = pd.Series(False, index=d.index)
    for s, e in zip(cyc["start"], cyc["end"]):
        in_cycle.loc[s:e] = True
    outages = nan_runs(d["outage"])
    observed_weeks = obs.sum() / 1440 / 7
    local = d.index.tz_convert(LOCAL_TZ)
    return {
        "house": house,
        "first_day": local[0].date(),
        "last_day": local[-1].date(),
        "span_days": round(len(d) / 1440, 1),
        "observed_days": round(obs.sum() / 1440, 1),
        "outage_pct": round(100 * d["outage"].mean(), 2),
        "outages_gt_5min": int((outages["minutes"] > MAX_FILL_GAP_MIN).sum()),
        "outages_gt_1day": int((outages["minutes"] > 1440).sum()),
        "frozen_pct": round(100 * d["stuck"].mean(), 2),
        "zero_pct": round(100 * (w == 0).mean(), 2),
        "standby_pct": round(100 * ((w > 0) & (w <= ON_W)).mean(), 2),
        "in_cycle_pct": round(100 * in_cycle[obs].mean(), 2),
        "on_outside_cycles_pct": round(100 * ((w > ON_W) & ~in_cycle[obs]).mean(), 2),
        "cycles": len(cyc),
        "cycles_complete": int((~cyc["truncated"]).sum()),
        "cycles_truncated": int(cyc["truncated"].sum()),
        "blips_rejected": int((~events["is_cycle"]).sum()),
        "cycles_per_week": round(len(cyc) / observed_weeks, 2),
        "issues_pct": round(100 * (d["issues"][obs] > 0).mean(), 2),
    }


def cycle_summary(cycles, data):
    rows = []
    for house, c in cycles[~cycles["truncated"]].groupby("house"):
        d = data[house]
        obs = d["wm"].notna() & d["aggregate"].notna()
        q = c[["minutes", "kwh", "peak_w"]].quantile([0.25, 0.5, 0.75])
        rows.append({
            "house": house,
            "complete_cycles": len(c),
            "duration_min_median": round(q.loc[0.5, "minutes"]),
            "duration_min_iqr": f"{q.loc[0.25, 'minutes']:.0f}-{q.loc[0.75, 'minutes']:.0f}",
            "kwh_median": round(q.loc[0.5, "kwh"], 2),
            "kwh_iqr": f"{q.loc[0.25, 'kwh']:.2f}-{q.loc[0.75, 'kwh']:.2f}",
            "peak_w_median": round(q.loc[0.5, "peak_w"]),
            "wm_share_of_house_energy_pct": round(
                100 * d.loc[obs, "wm"].sum() / d.loc[obs, "aggregate"].sum(), 2),
        })
    return pd.DataFrame(rows)


def rule_sensitivity(data):
    """How the cycle count responds to the two judgement calls in the rule."""
    rows = []
    for gap in (5, 10, 15, 30, 60):
        for min_kwh in (0.02, 0.05, 0.1):
            n, durations = 0, []
            for d in data.values():
                ev = detect_events(d["wm"], merge_gap=gap, min_kwh=min_kwh)
                c = ev[ev["is_cycle"]]
                n += len(c)
                durations.append(c.loc[~c["truncated"], "minutes"])
            rows.append({"merge_gap_min": gap, "min_kwh": min_kwh, "cycles": n,
                         "median_duration_min": float(pd.concat(durations).median())})
    return pd.DataFrame(rows)


def usage_shares(cycles):
    """Share of each house's cycles starting in each local hour / weekday."""
    start = pd.DatetimeIndex(cycles["start"]).tz_convert(LOCAL_TZ)
    c = cycles.assign(hour=start.hour, weekday=start.dayofweek, weekend=start.dayofweek >= 5)
    hour = pd.crosstab(c["house"], c["hour"], normalize="index").reindex(columns=range(24), fill_value=0)
    day = pd.crosstab(c["house"], c["weekday"], normalize="index").reindex(columns=range(7), fill_value=0)
    by_type = (c.groupby(["house", "weekend"])["hour"].value_counts(normalize=True)
               .unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
               .groupby(level="weekend").mean())
    return hour, day, by_type


# ---------------------------------------------------------------- figures

def plot_rule(pauses, events):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.logspace(0, np.log10(pauses.max() + 1), 50)
    axes[0].hist(pauses, bins=bins, color=BLUE, rwidth=0.85)
    axes[0].axvline(MERGE_GAP_MIN, color=ORANGE, lw=2)
    axes[0].annotate(f"merge pauses <= {MERGE_GAP_MIN} min", (MERGE_GAP_MIN, axes[0].get_ylim()[1] * 0.9),
                     xytext=(6, 0), textcoords="offset points", color=TEXT_2, fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("pause between ON (> 20 W) stretches, minutes (log scale)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Pauses between ON stretches", loc="left")

    kwh = events["kwh"].clip(lower=1e-3)
    bins = np.logspace(-3, np.log10(kwh.max()), 50)
    axes[1].hist(kwh, bins=bins, color=BLUE, rwidth=0.85)
    axes[1].axvline(MIN_CYCLE_KWH, color=ORANGE, lw=2)
    axes[1].annotate(f"cycle if >= {MIN_CYCLE_KWH} kWh\n(and >= {MIN_CYCLE_MIN} min)",
                     (MIN_CYCLE_KWH, axes[1].get_ylim()[1] * 0.8), xytext=(6, 0),
                     textcoords="offset points", color=TEXT_2, fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("energy per merged event, kWh (log scale)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Energy of merged events", loc="left")
    fig.tight_layout()
    return plotting.save(fig, "s2_cycle_rule.png")


def pick_typical(cycles, house):
    """The complete cycle closest to the house's median duration and energy."""
    c = cycles[(cycles["house"] == house) & ~cycles["truncated"]]
    z = ((c["minutes"] - c["minutes"].median()) / c["minutes"].std()) ** 2 \
        + ((c["kwh"] - c["kwh"].median()) / c["kwh"].std()) ** 2
    return c.loc[z.sort_values().index]


def plot_day(data, cycles, summary):
    """(i) A representative 24 h: the house whose median cycle energy is closest to the
    cross-house median, on a fully observed local day whose only cycle is the most
    typical one available."""
    target = summary["kwh_median"].median()
    house = int(summary.loc[(summary["kwh_median"] - target).abs().idxmin(), "house"])
    d = data[house]
    house_cycles = cycles[cycles["house"] == house]
    for _, cyc in pick_typical(cycles, house).iterrows():
        day = cyc["start"].tz_convert(LOCAL_TZ).normalize()
        window = d.loc[day:day + pd.Timedelta("1D") - pd.Timedelta("1min")]
        n_cycles = house_cycles["start"].between(window.index[0], window.index[-1]).sum()
        if (len(window) == 1440 and n_cycles == 1
                and window["wm"].notna().all() and window["aggregate"].notna().all()):
            break
    t = window.index.tz_convert(LOCAL_TZ)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(t, window["aggregate"], color=MUTED, lw=1, label="whole-home aggregate")
    ax.fill_between(t, window["wm"], color=BLUE, alpha=0.85, lw=0, label="washing machine (sub-metered)")
    s = cyc["start"].tz_convert(LOCAL_TZ)
    ax.annotate(f"wash cycle\n{cyc['minutes']} min, {cyc['kwh']:.2f} kWh\n"
                f"= {100 * cyc['kwh'] / (window['aggregate'].sum() / 60 / 1000):.0f} % of the day's use",
                (s, cyc["peak_w"]), xytext=(-10, 10), textcoords="offset points", ha="right",
                fontsize=8, color=TEXT_2)
    ax.set_ylabel("W")
    ax.set_xlabel(f"local time ({day.strftime('%a %d %b %Y')})")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    ax.legend(loc="upper right")
    ax.set_title(f"House {house}: a representative day - aggregate and washing machine", loc="left")
    fig.tight_layout()
    return plotting.save(fig, "s2_day_trace.png"), house, day


def plot_cycles(data, cycles, summary):
    """(ii) The most typical complete cycle from four houses spanning the duration range."""
    order = summary.sort_values("duration_min_median")["house"].to_numpy()
    houses = [int(order[int(round(q * (len(order) - 1)))]) for q in (0, 1 / 3, 2 / 3, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5))
    for ax, house in zip(axes.flat, houses):
        c = pick_typical(cycles, house).iloc[0]
        pad = pd.Timedelta("20min")
        w = data[house]["wm"].loc[c["start"] - pad:c["end"] + pad]
        t = w.index.tz_convert(LOCAL_TZ)
        ax.axvspan(c["start"].tz_convert(LOCAL_TZ), c["end"].tz_convert(LOCAL_TZ),
                   color=LIGHT_BLUE, lw=0, label="detected cycle")
        ax.plot(t, w, color=BLUE, lw=1.3, label="washing machine, 1-min")
        ax.axhline(ON_W, color=ORANGE, lw=1, ls="--", label=f"ON threshold {ON_W} W")
        ax.set_title(f"House {house}: {c['minutes']} min, {c['kwh']:.2f} kWh, peak {c['peak_w']:,.0f} W",
                     loc="left")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M", tz=LOCAL_TZ))
        ax.set_ylabel("W")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Typical detected wash cycles (heating spike, then wash/rinse, then spin)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT)
    fig.tight_layout()
    return plotting.save(fig, "s2_cycle_examples.png"), houses


def plot_cross_house(cycles, avail):
    """(iii) Energy per cycle, duration and frequency across houses."""
    c = cycles[~cycles["truncated"]]
    order = c.groupby("house")["kwh"].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})
    box = dict(vert=False, widths=0.6, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=LIGHT_BLUE, edgecolor=BLUE),
               medianprops=dict(color=plotting.TEXT, lw=1.5),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    labels = [f"House {h}" for h in order]
    axes[0].boxplot([c.loc[c["house"] == h, "kwh"] for h in order], tick_labels=labels, **box)
    axes[0].set_xlabel("energy per cycle (kWh)")
    axes[0].set_title("Energy per cycle", loc="left")
    axes[1].boxplot([c.loc[c["house"] == h, "minutes"] for h in order], tick_labels=labels, **box)
    axes[1].set_xlabel("duration (min)")
    axes[1].set_title("Cycle duration", loc="left")
    per_week = avail.set_index("house").loc[order, "cycles_per_week"]
    axes[2].barh(range(1, len(order) + 1), per_week, color=BLUE, height=0.6)
    for y, v in enumerate(per_week, start=1):
        axes[2].annotate(f"{v:.1f}", (v, y), xytext=(3, 0), textcoords="offset points",
                         va="center", fontsize=8, color=TEXT_2)
    axes[2].set_xlabel("cycles per observed week")
    axes[2].set_title("Frequency", loc="left")
    axes[2].grid(axis="y", visible=False)
    fig.suptitle("Washing-machine cycles across 14 houses (complete cycles; boxes = middle 50 %)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT)
    fig.tight_layout()
    return plotting.save(fig, "s2_cross_house.png")


def plot_patterns(hour, day, by_type):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1, 1.6])
    ax_h, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    hm_h, hm_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for weekend, colour, label in [(False, BLUE, "weekdays"), (True, ORANGE, "weekends")]:
        ax_h.plot(range(24), 100 * by_type.loc[weekend], color=colour, lw=2, marker="o", ms=4, label=label)
    ax_h.set_xticks(range(0, 24, 2))
    ax_h.set_xlabel("local hour of cycle start")
    ax_h.set_ylabel("% of cycles")
    ax_h.set_title("When washes start (houses weighted equally)", loc="left")
    ax_h.legend(loc="upper right")

    pooled_day = 100 * day.mean()
    ax_d.bar(range(7), pooled_day, color=BLUE, width=0.7)
    ax_d.axhline(100 / 7, color=TEXT_2, lw=1, ls="--")
    ax_d.annotate("even split (14.3 %)", (-0.4, 100 / 7), xytext=(0, -11), textcoords="offset points",
                  ha="left", fontsize=8, color=TEXT_2)
    ax_d.set_xticks(range(7), WEEKDAYS)
    ax_d.set_ylabel("% of cycles")
    ax_d.set_title("By day of week", loc="left")

    for ax, table, xt in [(hm_h, hour, list(range(24))), (hm_d, day, WEEKDAYS)]:
        im = ax.imshow(100 * table.to_numpy(), aspect="auto", cmap=SEQ_CMAP, vmin=0)
        ax.set_yticks(range(len(table)), [f"House {h}" for h in table.index])
        ax.set_xticks(range(len(xt)), xt if isinstance(xt[0], str) else [str(x) for x in xt])
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="% of that house's cycles", shrink=0.8)
    hm_h.set_xlabel("local hour of cycle start")
    hm_h.set_title("Each house's own timing", loc="left")
    hm_d.set_title("Each house by weekday", loc="left")
    fig.tight_layout()
    return plotting.save(fig, "s2_usage_patterns.png")


def plot_frozen(data):
    """Data-quality example: the longest frozen-sensor run (House 5)."""
    d = data[5]
    runs = nan_runs(d["stuck"])
    r = runs.loc[runs["minutes"].idxmax()]
    pad = pd.Timedelta("3h")
    raw = pd.read_parquet(processed_path(5))["wm"].loc[r["start"] - pad:r["end"] + pad]
    t = raw.index.tz_convert(LOCAL_TZ)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axvspan(r["start"].tz_convert(LOCAL_TZ), r["end"].tz_convert(LOCAL_TZ), color=plotting.SHADE,
               lw=0, label=f"frozen: same value for {r['minutes'] / 60:.1f} h -> treated as missing")
    ax.plot(t, raw, color=BLUE, lw=1.2, label="washing machine, CLEAN 1-min")
    ax.set_ylabel("W")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b %H:%M", tz=LOCAL_TZ))
    ax.legend(loc="center")
    ax.set_title("House 5: a frozen washing-machine reading in the official CLEAN data", loc="left")
    fig.tight_layout()
    return plotting.save(fig, "s2_frozen_sensor.png")


def main():
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    data = {h: load_house(h) for h in config.STUDY_HOUSES}

    events, avail, pauses = [], [], []
    for house, d in data.items():
        ev = detect_events(d["wm"]).assign(house=house)
        events.append(ev)
        avail.append(availability(house, d, ev))
        on = d["wm"].gt(ON_W)
        r = nan_runs(on)
        gap = (r["start"].shift(-1) - r["end"]).dt.total_seconds() / 60 - 1
        pauses.append(gap.dropna())
    events = pd.concat(events, ignore_index=True)
    cycles = events[events["is_cycle"]].reset_index(drop=True)
    avail = pd.DataFrame(avail)
    summary = cycle_summary(cycles, data)
    hour, day, by_type = usage_shares(cycles)
    sensitivity = rule_sensitivity(data)

    out = cycles.assign(start_local=pd.DatetimeIndex(cycles["start"]).tz_convert(LOCAL_TZ))
    out.to_csv(config.RESULTS / "s2_cycles.csv", index=False)
    avail.to_csv(config.RESULTS / "s2_availability.csv", index=False)
    summary.to_csv(config.RESULTS / "s2_cycle_summary.csv", index=False)
    sensitivity.to_csv(config.RESULTS / "s2_rule_sensitivity.csv", index=False)
    (100 * hour).round(1).to_csv(config.RESULTS / "s2_share_by_hour.csv")
    (100 * day).round(1).set_axis(WEEKDAYS, axis=1).to_csv(config.RESULTS / "s2_share_by_weekday.csv")

    plot_rule(pd.concat(pauses), events)
    _, day_house, day_date = plot_day(data, cycles, summary)
    _, example_houses = plot_cycles(data, cycles, summary)
    plot_cross_house(cycles, avail)
    plot_patterns(hour, day, by_type)
    plot_frozen(data)

    print(avail.to_string(index=False))
    print(summary.to_string(index=False))
    print(sensitivity.pivot(index="merge_gap_min", columns="min_kwh", values="cycles").to_string())
    print("pooled weekday share %:", (100 * day.mean()).round(1).set_axis(WEEKDAYS).to_dict())
    print("pooled hour share % (weekday):", (100 * by_type.loc[False]).round(1).to_dict())
    print("pooled hour share % (weekend):", (100 * by_type.loc[True]).round(1).to_dict())
    print(f"day trace: House {day_house} on {day_date.date()}; cycle examples: houses {example_houses}")


if __name__ == "__main__":
    main()
