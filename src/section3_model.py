"""Section 3: estimate washing-machine power from the household aggregate (1 minute).

Inputs: a window of the aggregate around each minute (the Seq2Point idea,
"window in, midpoint out", with tree models instead of a CNN). All models see
the same 63 features.

Models (all two-stage ones: a classifier gives P(machine running); where P > tau
a regressor trained on running minutes gives the watts, otherwise 0):
  * LightGBM one-stage  - a single regressor (the first version)
  * LightGBM two-stage  - binary classifier + regressor
  * Random forest two-stage - the same design with scikit-learn random forests
  * LightGBM multi-appliance - the classifier also learns the look-alike
    appliances (dishwasher, tumble dryer, washer-dryer) as their own class, so
    they compete with the washing machine for the same signal
  * "+ cycle filter" - any of the above, keeping predicted power only inside
    events that pass the Section 2 cycle rule (removes short false alarms)
Baselines: always-off, and a usage profile (mean washing-machine power by local
hour x weekday/weekend, learned on the training houses).

Split by household (no house in more than one set):
  train TRAIN_HOUSES; validation VAL_HOUSES (early stopping, tau, and the choice
  of final model: lowest validation MAE); test TEST_HOUSES (every model is
  reported). Evaluation uses every minute where the aggregate
  and the washing machine are observed; training also drops minutes where the
  1-minute appliance sum exceeds the aggregate (see house_frame).

Run: python -m src.section3_model   (after python -m src.prepare_clean)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src import config, plotting
from src.plotting import BLUE, MUTED, ORANGE, TEXT_2, plt
from src.refit_io import read_clean_house
from src.section2_eda import LOCAL_TZ, ON_W, detect_events, frozen, load_house

TRAIN_HOUSES = [1, 5, 6, 7, 9, 10, 16, 18, 19]
VAL_HOUSES = [15, 17]
TEST_HOUSES = [2, 8, 20]

TRAIN_STRIDE = 3          # every 3rd minute: neighbours are near-duplicates
OFFSETS = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60)
WINDOWS = (5, 15, 31, 61, 121)
HEATER_W = 1500           # a heating element shows as a step of this size or more
SEED = 0

# Other large (>1 kW) appliances metered in the test houses (from the readme),
# used only to explain false positives.
OTHER_LOADS = {
    2: {"Appliance3": "Dishwasher", "Appliance5": "Microwave", "Appliance6": "Toaster",
        "Appliance8": "Kettle"},
    8: {"Appliance3": "Washer-dryer", "Appliance5": "Toaster", "Appliance8": "Microwave",
        "Appliance9": "Kettle"},
    20: {"Appliance3": "Tumble dryer", "Appliance5": "Dishwasher", "Appliance8": "Microwave",
         "Appliance9": "Kettle"},
}

# Appliances with the same "heater + motor" signature, per training/validation
# house (from the readme): dishwasher, tumble dryer, washer-dryer. They give the
# multi-appliance classifier its third class. Test houses are never used here.
LOOKALIKES = {
    1: ["Appliance4", "Appliance6"], 5: ["Appliance2", "Appliance4"], 6: ["Appliance3"],
    7: ["Appliance4", "Appliance6"], 9: ["Appliance2", "Appliance4"], 10: ["Appliance6"],
    16: ["Appliance6"], 18: ["Appliance4", "Appliance6"], 19: [],
    15: ["Appliance2", "Appliance4"], 17: ["Appliance3"],
}
NONE, WASHER, LOOKALIKE = 0, 1, 2   # classes of the multi-appliance classifier


def make_features(d):
    """Features for each minute from the aggregate only, plus local time of day.

    `excess` is the aggregate above its 2-hour rolling minimum (the house's
    background load), which makes the features comparable across houses.
    Windows are centred: the model sees up to an hour before and after.
    """
    agg = d["aggregate"]
    base = agg.rolling(121, center=True, min_periods=30).min()
    excess = agg - base
    f = {"agg": agg, "excess": excess, "base": base,
         "diff_back": agg.diff(), "diff_fwd": agg.shift(-1) - agg}
    for k in OFFSETS:
        f[f"excess_lag{k}"] = excess.shift(k)
        f[f"excess_lead{k}"] = excess.shift(-k)
    activity = agg.diff().abs()
    heater = (excess > HEATER_W).astype(float).where(agg.notna())
    for w in WINDOWS:
        r = excess.rolling(w, center=True, min_periods=w // 2)
        f[f"mean_{w}"], f[f"std_{w}"] = r.mean(), r.std()
        f[f"max_{w}"], f[f"min_{w}"] = r.max(), r.min()
        f[f"heater_min_{w}"] = heater.rolling(w, center=True, min_periods=w // 2).sum()
        f[f"activity_{w}"] = activity.rolling(w, center=True, min_periods=w // 2).mean()
        f[f"mid_band_{w}"] = excess.between(50, 600).astype(float).where(agg.notna()) \
            .rolling(w, center=True, min_periods=w // 2).mean()   # drum-turning level
    local = d.index.tz_convert(LOCAL_TZ)
    hour = local.hour + local.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["weekend"] = (local.dayofweek >= 5).astype(float)
    return pd.DataFrame(f, index=d.index).astype("float32")


def house_frame(house):
    """Features, target and bookkeeping for one house (all minutes on the grid)."""
    d = load_house(house)
    X = make_features(d)
    valid = d["wm"].notna() & d["aggregate"].notna()
    # CLEAN's per-sample `Issues` flag fires at almost every switch-on (IAMs report
    # ahead of the aggregate), so it marks 11-46 % of washing minutes. For training
    # we drop a minute only if the inconsistency survives 1-minute averaging.
    # Evaluation keeps every observed minute: a deployed model has no sub-meters
    # to run this check with.
    d["train_ok"] = valid & (d["iam_sum"] <= d["aggregate"])

    # 3-class label: washing machine running, else a look-alike running, else none.
    lookalike = pd.Series(False, index=d.index)
    cols = LOOKALIKES.get(house, [])
    if cols:
        app = read_clean_house(house, cols)[cols].resample("1min").mean().reindex(d.index)
        running = [(app[c] > ON_W) & ~frozen(app[c]) for c in cols]
        lookalike = pd.concat(running, axis=1).any(axis=1)
    d["label"] = np.where(d["wm"] > ON_W, WASHER, np.where(lookalike, LOOKALIKE, NONE))
    return d, X, valid


def profile_table(frames):
    """Mean washing-machine power by local hour and weekend flag, over training houses."""
    rows = []
    for d, _, _ in frames:
        local = d.index.tz_convert(LOCAL_TZ)
        rows.append(pd.DataFrame({"hour": local.hour, "weekend": local.dayofweek >= 5,
                                  "wm": d["wm"]})[d["train_ok"].to_numpy()])
    return pd.concat(rows).groupby(["hour", "weekend"])["wm"].mean()


def profile_predict(table, index):
    local = index.tz_convert(LOCAL_TZ)
    keys = pd.MultiIndex.from_arrays([local.hour, local.dayofweek >= 5])
    return table.reindex(keys).to_numpy()


LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, verbose=-1)
TAUS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


# Fixed in advance, not tuned: enough trees for stable probabilities, each tree
# on a 25 % sample, leaves of >= 20 minutes.
RF_PARAMS = dict(n_estimators=100, max_samples=0.25, min_samples_leaf=20, max_features="sqrt",
                 random_state=SEED, n_jobs=-1)


class TwoStage:
    """P(washing machine running) from a classifier; watts from a regressor fitted
    on running minutes. `on_label` is the classifier's class for "running"."""

    def __init__(self, clf, reg, on_label, tau=None):
        self.clf, self.reg, self.on_label, self.tau = clf, reg, on_label, tau

    def parts(self, X):
        col = list(self.clf.classes_).index(self.on_label)
        return self.clf.predict_proba(X)[:, col], np.clip(self.reg.predict(X), 0, 4000)

    def predict(self, X, tau=None):
        prob, watts = self.parts(X)
        return np.where(prob > (self.tau if tau is None else tau), watts, 0.0)


def training_data(train, val):
    Xtr = pd.concat([X[d["train_ok"]][::TRAIN_STRIDE] for d, X, _ in train])
    ytr = pd.concat([d["wm"][d["train_ok"]][::TRAIN_STRIDE] for d, _, _ in train]).to_numpy()
    ctr = np.concatenate([d["label"][d["train_ok"].to_numpy()][::TRAIN_STRIDE] for d, _, _ in train])
    Xva = pd.concat([X[v] for _, X, v in val])
    yva = pd.concat([d["wm"][v] for d, _, v in val]).to_numpy()
    cva = np.concatenate([d["label"][v.to_numpy()] for d, _, v in val])
    return Xtr, ytr, ctr, Xva, yva, cva


def train_models(train, val):
    """Fit every model; choose each two-stage model's tau on validation (lowest MAE)."""
    Xtr, ytr, ctr, Xva, yva, cva = training_data(train, val)
    stop = [lgb.early_stopping(100, verbose=False)]
    on_tr, on_va = ytr > ON_W, yva > ON_W

    one_stage = lgb.LGBMRegressor(**LGB_PARAMS)
    one_stage.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1", callbacks=stop)

    # Watts given running: shared by both LightGBM two-stage models, so the
    # multi-appliance comparison changes only the classifier.
    reg = lgb.LGBMRegressor(**LGB_PARAMS)
    reg.fit(Xtr[on_tr], ytr[on_tr], eval_set=[(Xva[on_va], yva[on_va])], eval_metric="l1",
            callbacks=stop)
    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(Xtr, on_tr, eval_set=[(Xva, on_va)], callbacks=stop)
    multi = lgb.LGBMClassifier(**LGB_PARAMS)
    multi.fit(Xtr, ctr, eval_set=[(Xva, cva)], callbacks=stop)

    rf_clf = RandomForestClassifier(**RF_PARAMS).fit(Xtr, on_tr)
    rf_reg = RandomForestRegressor(**RF_PARAMS).fit(Xtr[on_tr], ytr[on_tr])

    models = {
        "LightGBM two-stage": TwoStage(clf, reg, True),
        "Random forest two-stage": TwoStage(rf_clf, rf_reg, True),
        "LightGBM multi-appliance": TwoStage(multi, reg, WASHER),
    }
    sweep = [{"model": "Always off", **quick_scores(np.zeros_like(yva), yva)},
             {"model": "LightGBM one-stage", **quick_scores(postprocess(one_stage.predict(Xva)), yva)}]
    for name, model in models.items():
        prob, watts = model.parts(Xva)
        rows = [{"model": name, "tau": tau, **quick_scores(np.where(prob > tau, watts, 0.0), yva)}
                for tau in TAUS]
        model.tau = min(rows, key=lambda r: r["mae_w"])["tau"]
        sweep += rows
    info = {"train_rows": len(Xtr), "label_counts": np.bincount(ctr).tolist(),
            "one_stage_iters": one_stage.best_iteration_, "reg_iters": reg.best_iteration_,
            "clf_iters": clf.best_iteration_, "multi_iters": multi.best_iteration_,
            "taus": {n: m.tau for n, m in models.items()}}
    return one_stage, models, pd.DataFrame(sweep), info


def cycle_filter(p, index, valid):
    """Keep predicted power only inside predicted events that pass the Section 2
    cycle rule; shorter or smaller events (likely false alarms) become 0."""
    full = pd.Series(np.nan, index=index)
    full[valid] = p
    ev = detect_events(full)
    keep = pd.Series(False, index=index)
    for s, e in zip(ev.loc[ev["is_cycle"], "start"], ev.loc[ev["is_cycle"], "end"]):
        keep.loc[s:e] = True
    return np.where(keep[valid].to_numpy(), p, 0.0)


def quick_scores(p, y):
    on_t, on_p = y > ON_W, p > ON_W
    tp = (on_t & on_p).sum()
    prec, rec = tp / max(on_p.sum(), 1), tp / on_t.sum()
    return {"mae_w": np.abs(p - y).mean(), "energy_error_pct": 100 * (p.sum() - y.sum()) / y.sum(),
            "on_precision": prec, "on_recall": rec, "on_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def postprocess(p, threshold=ON_W):
    """Clip to physical range; below the threshold the machine is idle."""
    p = np.clip(p, 0, 4000)
    p[p < threshold] = 0
    return p


def cycle_match(y_full, p_full):
    """Event-level recall/precision using the Section 2 cycle rule on both series."""
    def cycles(s):
        ev = detect_events(s)
        return ev[ev["is_cycle"]]
    t, p = cycles(y_full), cycles(p_full)
    if len(t) == 0 or len(p) == 0:
        return len(t), len(p), 0.0, 0.0
    ts, te = t["start"].to_numpy()[:, None], t["end"].to_numpy()[:, None]
    ps, pe = p["start"].to_numpy()[None, :], p["end"].to_numpy()[None, :]
    overlap = (ts <= pe) & (ps <= te)
    return len(t), len(p), overlap.any(axis=1).mean(), overlap.any(axis=0).mean()


def metrics(y, p, index, house):
    """Per-minute, energy, on/off and daily-energy metrics on valid minutes.

    `house` labels each minute so that daily energy is summed per house-day.
    """
    on_t, on_p = y > ON_W, p > ON_W
    tp = (on_t & on_p).sum()
    precision = tp / on_p.sum() if on_p.sum() else 0.0
    recall = tp / on_t.sum() if on_t.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    daily = pd.DataFrame({"y": y, "p": p, "house": house,
                          "day": index.tz_convert(LOCAL_TZ).normalize()}).groupby(["house", "day"])
    full_days = daily.size() >= 0.95 * 1440
    kwh = (daily[["y", "p"]].sum()[full_days] / 60 / 1000)
    return {
        "minutes": len(y),
        "mae_w": np.abs(p - y).mean(),
        "mae_when_on_w": np.abs(p - y)[on_t].mean(),
        "energy_error_pct": 100 * (p.sum() - y.sum()) / y.sum(),
        "daily_kwh_mae": np.abs(kwh["p"] - kwh["y"]).mean(),
        "daily_kwh_mean_actual": kwh["y"].mean(),
        "on_precision": precision,
        "on_recall": recall,
        "on_f1": f1,
    }


def evaluate(houses, frames, preds):
    """Metrics per house and pooled, for every prediction set in `preds`."""
    rows = []
    for name in preds:
        ys, ps, idx, hs = [], [], [], []
        for house, (d, _, valid) in zip(houses, frames):
            y = d["wm"][valid].to_numpy()
            p = preds[name][house]
            m = metrics(y, p, d.index[valid], np.full(len(y), house))
            y_full = d["wm"].where(valid)
            p_full = pd.Series(np.nan, index=d.index)
            p_full[valid] = p
            m["true_cycles"], m["pred_cycles"], m["cycle_recall"], m["cycle_precision"] = \
                cycle_match(y_full, p_full)
            rows.append({"model": name, "house": house, **m})
            ys.append(y), ps.append(p), idx.append(d.index[valid]), hs.append(np.full(len(y), house))
        pooled = metrics(np.concatenate(ys), np.concatenate(ps), idx[0].append(idx[1:]), np.concatenate(hs))
        sub = [r for r in rows if r["model"] == name]
        tc, pc = sum(r["true_cycles"] for r in sub), sum(r["pred_cycles"] for r in sub)
        pooled.update(true_cycles=tc, pred_cycles=pc,
                      cycle_recall=sum(r["cycle_recall"] * r["true_cycles"] for r in sub) / tc,
                      cycle_precision=sum(r["cycle_precision"] * r["pred_cycles"] for r in sub) / pc
                      if pc else 0.0)
        rows.append({"model": name, "house": "all", **pooled})
    return pd.DataFrame(rows)


def false_positive_sources(house, d, valid, p):
    """Which other metered heavy appliance was running when the model wrongly said ON."""
    others = OTHER_LOADS[house]
    app = read_clean_house(house, list(others))[list(others)].resample("1min").mean()
    app = app.reindex(d.index)[valid].fillna(0)   # a missing reading counts as "not running"
    y = d["wm"][valid].to_numpy()
    fp = (p > ON_W) & (y <= ON_W)
    running = app.to_numpy() > 200
    label = np.where(running.any(axis=1),
                     np.array(list(others.values()))[app.to_numpy().argmax(axis=1)],
                     "no other metered appliance")
    s = pd.Series(p[fp] / 60 / 1000, index=label[fp]).groupby(level=0).sum()
    share = (100 * s / s.sum()).round(1).sort_values(ascending=False)
    return share, s.sum(), pd.Series(np.where(fp, label, ""), index=d.index[valid]), app


def missed_cycles(test, preds, final):
    """Compare true cycles the model found with the ones it missed."""
    rows = []
    for house, (d, _, valid) in test.items():
        p_full = pd.Series(np.nan, index=d.index)
        p_full[valid] = preds[final][house]
        t = detect_events(d["wm"].where(valid))
        t = t[t["is_cycle"] & ~t["truncated"]].copy()
        p = detect_events(p_full)
        p = p[p["is_cycle"]]
        hit = (t["start"].to_numpy()[:, None] <= p["end"].to_numpy()[None, :]) \
            & (p["start"].to_numpy()[None, :] <= t["end"].to_numpy()[:, None])
        t["detected"] = hit.any(axis=1)
        other = d["aggregate"] - d["wm"]
        t["other_load_w"] = [other.loc[s:e].median() for s, e in zip(t["start"], t["end"])]
        t["house"] = house
        rows.append(t)
    c = pd.concat(rows)
    summary = c.groupby("detected").agg(
        cycles=("kwh", "size"), kwh_median=("kwh", "median"), minutes_median=("minutes", "median"),
        peak_w_median=("peak_w", "median"), other_load_w_median=("other_load_w", "median"))
    return c, summary


def plot_false_alarm(house, d, valid, p, fp_label, app, final):
    """A failure case: the day with the most false-alarm energy during another
    metered heating appliance, on a day without any real wash."""
    pred = pd.Series(p, index=d.index[valid])
    named = fp_label[(fp_label != "") & (fp_label != "no other metered appliance")]
    local_day = lambda idx: idx.tz_convert(LOCAL_TZ).normalize()
    wm_day = d["wm"].groupby(local_day(d.index)).max()
    fp_day = pred[named.index].groupby(local_day(named.index)).sum()
    fp_day = fp_day[wm_day.reindex(fp_day.index) <= ON_W].sort_values(ascending=False)
    day = fp_day.index[0]
    lo, hi = day.tz_convert("UTC"), (day + pd.Timedelta("1D")).tz_convert("UTC")
    appliance = named.loc[lo:hi].value_counts().index[0]
    column = [c for c, n in OTHER_LOADS[house].items() if n == appliance][0]

    w = d.loc[lo:hi]
    t = w.index.tz_convert(LOCAL_TZ)
    fig, (ax_a, ax_w) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1]})
    ax_a.plot(t, w["aggregate"], color=MUTED, lw=1, label="whole-home aggregate")
    ax_a.plot(t, app[column].reindex(w.index), color=plotting.AQUA, lw=1.3,
              label=f"{appliance.lower()} (sub-metered)")
    ax_a.set_ylabel("W")
    ax_a.legend(loc="upper right")
    ax_a.set_title(f"House {house}, {day.strftime('%a %d %b %Y')}: the washing machine never ran",
                   loc="left")
    ax_w.plot(t, w["wm"], color=BLUE, lw=1.2, label="actual washing machine")
    ax_w.plot(t, pred.reindex(w.index), color=ORANGE, lw=1.4, label=f"predicted ({final})")
    ax_w.set_ylabel("W")
    ax_w.legend(loc="upper right")
    ax_w.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    fig.suptitle(f"Failure case: the {appliance.lower()}'s heater + motor pattern is mistaken "
                 "for a wash", x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT)
    fig.tight_layout()
    return plotting.save(fig, "s3_failure_false_alarm.png"), day, appliance


# ---------------------------------------------------------------- figures

def plot_actual_vs_predicted(test, preds, cycles_csv, final):
    """One typical true cycle per test house: aggregate, true and predicted WM."""
    cyc = pd.read_csv(cycles_csv, parse_dates=["start", "end"])
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(len(test), 1, hspace=0.45, top=0.93, bottom=0.05)
    for row, (house, (d, _, valid)) in enumerate(test.items()):
        c = cyc[(cyc["house"] == house) & ~cyc["truncated"]]
        c = c.iloc[((c["minutes"] - c["minutes"].median()).abs()
                    + 60 * (c["kwh"] - c["kwh"].median()).abs()).argsort()]
        for _, cy in c.iterrows():
            lo, hi = cy["start"] - pd.Timedelta("3h"), cy["end"] + pd.Timedelta("3h")
            if valid.loc[lo:hi].mean() > 0.98:
                break
        sub = gs[row].subgridspec(2, 1, height_ratios=[1, 1.6], hspace=0.08)
        ax_a, ax_w = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        w = d.loc[lo:hi]
        t = w.index.tz_convert(LOCAL_TZ)
        pm = pd.Series(np.nan, index=d.index)
        pm[valid] = preds[final][house]
        ax_a.plot(t, w["aggregate"], color=MUTED, lw=1)
        ax_a.set_ylabel("aggregate\n(W)")
        ax_a.tick_params(labelbottom=False)
        ax_a.set_title(f"Test House {house} (never seen in training)", loc="left")
        ax_w.fill_between(t, w["wm"], color=BLUE, alpha=0.35, lw=0, label="actual washing machine")
        ax_w.plot(t, w["wm"], color=BLUE, lw=1)
        ax_w.plot(t, pm.loc[lo:hi], color=ORANGE, lw=1.4, label=f"predicted ({final})")
        ax_w.set_ylabel("washing\nmachine (W)")
        for ax in (ax_a, ax_w):
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M", tz=LOCAL_TZ))
            ax.set_xlim(t[0], t[-1])
        if row == 0:
            ax_w.legend(loc="upper right")
        ax_w.annotate(t[0].strftime("%a %d %b %Y"), (0.005, 0.85), xycoords="axes fraction",
                      fontsize=8, color=TEXT_2)
    fig.suptitle("Actual vs predicted washing-machine power on the test houses (local time)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT, y=0.995)
    return plotting.save(fig, "s3_actual_vs_predicted.png")


def plot_daily(test, preds, final):
    """Daily washing-machine energy, actual vs predicted, on full test days."""
    fig, axes = plt.subplots(1, len(test), figsize=(13, 4.3), sharex=True, sharey=True)
    hi = 0
    for ax, (house, (d, _, valid)) in zip(axes, test.items()):
        idx = d.index[valid]
        day = idx.tz_convert(LOCAL_TZ).normalize()
        df = pd.DataFrame({"y": d["wm"][valid].to_numpy(), "p": preds[final][house],
                           "prof": preds["Usage profile"][house]}, index=day)
        g = df.groupby(level=0)
        full = g.size() >= 0.95 * 1440
        kwh = g.sum()[full] / 60 / 1000
        hi = max(hi, kwh[["y", "p"]].max().max())
        ax.scatter(kwh["y"], kwh["prof"], s=10, color=MUTED, label="usage-profile baseline", zorder=2)
        ax.scatter(kwh["y"], kwh["p"], s=12, color=ORANGE, label=final, zorder=3)
        r = np.corrcoef(kwh["y"], kwh["p"])[0, 1]
        ax.set_title(f"House {house}: {full.sum()} days, r = {r:.2f}", loc="left")
        ax.set_xlabel("actual kWh / day")
    for ax in axes:
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color=TEXT_2, lw=1, ls="--", zorder=1)
    axes[0].set_ylabel("predicted kWh / day")
    axes[0].legend(loc="upper left")
    fig.suptitle("Daily washing-machine energy on the test houses (dashed = perfect)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=plotting.TEXT)
    fig.tight_layout()
    return plotting.save(fig, "s3_daily_energy.png")


FILTERED = " + cycle filter"
BASELINES = ("Always off", "Usage profile")


def importance(model, columns):
    """Share of the classifier's importance per feature (LightGBM: gain; forest: impurity)."""
    clf = model.clf
    vals = clf.booster_.feature_importance("gain") if hasattr(clf, "booster_") else clf.feature_importances_
    s = pd.Series(vals, index=columns, dtype=float)
    return (100 * s / s.sum()).sort_values(ascending=False)


def main():
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    train = [house_frame(h) for h in TRAIN_HOUSES]
    val = [house_frame(h) for h in VAL_HOUSES]
    one_stage, models, sweep, info = train_models(train, val)
    profile = profile_table(train)
    print(info)
    sweep.round(4).to_csv(config.RESULTS / "s3_validation_model_choice.csv", index=False)

    def predict_all(frames, houses):
        out = {name: {} for name in BASELINES}
        for house, (d, X, valid) in zip(houses, frames):
            out["Always off"][house] = np.zeros(int(valid.sum()))
            out["Usage profile"][house] = profile_predict(profile, d.index[valid])
            learned = {"LightGBM one-stage": postprocess(one_stage.predict(X[valid]))}
            learned.update({name: m.predict(X[valid]) for name, m in models.items()})
            for name, p in learned.items():
                out.setdefault(name, {})[house] = p
                out.setdefault(name + FILTERED, {})[house] = cycle_filter(p, d.index, valid)
        return out

    val_results = evaluate(VAL_HOUSES, val, predict_all(val, VAL_HOUSES))
    val_results.round(4).to_csv(config.RESULTS / "s3_metrics_validation.csv", index=False)

    # The final model is chosen here, on validation only: lowest pooled MAE among learned models.
    pooled = val_results[(val_results["house"] == "all") & ~val_results["model"].isin(BASELINES)]
    final = pooled.loc[pooled["mae_w"].idxmin(), "model"]
    print(f"final model (lowest validation MAE): {final}")

    test = {h: house_frame(h) for h in TEST_HOUSES}
    preds = predict_all(list(test.values()), TEST_HOUSES)
    results = evaluate(TEST_HOUSES, list(test.values()), preds)
    results.round(4).to_csv(config.RESULTS / "s3_metrics_test.csv", index=False)

    base_name = final.replace(FILTERED, "")
    final_model = models.get(base_name)
    imp = importance(final_model if final_model else models["LightGBM two-stage"], train[0][1].columns)
    imp.round(2).to_csv(config.RESULTS / "s3_feature_importance.csv", header=["importance_pct"])

    # False alarms of the final model, and of the binary two-stage model for comparison.
    fp_rows, fp_detail = [], {}
    for name in dict.fromkeys([final, "LightGBM two-stage"]):
        for house, (d, _, valid) in test.items():
            share, kwh, labels, app = false_positive_sources(house, d, valid, preds[name][house])
            if name == final:
                fp_detail[house] = (labels, app)
            fp_rows += [{"model": name, "house": house, "source": src, "share_pct": pct,
                         "false_alarm_kwh_total": round(kwh, 1)} for src, pct in share.items()]
    fp_long = pd.DataFrame(fp_rows)
    fp_long.to_csv(config.RESULTS / "s3_false_positive_sources.csv", index=False)
    fp = fp_long[fp_long["model"] == final].pivot(index="source", columns="house", values="share_pct").fillna(0)
    print(fp_long.pivot_table(index=["model", "source"], columns="house", values="share_pct").round(1))

    cycles, missed = missed_cycles(test, preds, final)
    missed.round(3).to_csv(config.RESULTS / "s3_missed_vs_detected_cycles.csv")
    print(missed.round(2).to_string())

    plot_actual_vs_predicted(test, preds, config.RESULTS / "s2_cycles.csv", final)
    plot_daily(test, preds, final)
    named_share = 100 - fp.loc["no other metered appliance"]
    house = int(named_share.idxmax())
    d, _, valid = test[house]
    _, day, appliance = plot_false_alarm(house, d, valid, preds[final][house], *fp_detail[house], final)
    print(f"failure example: House {house}, {day.date()}, {appliance}")

    cols = ["model", "house", "mae_w", "mae_when_on_w", "energy_error_pct", "daily_kwh_mae",
            "on_precision", "on_recall", "on_f1", "cycle_recall", "cycle_precision",
            "true_cycles", "pred_cycles"]
    print(val_results.loc[val_results["house"] == "all", cols].round(3).to_string(index=False))
    print(results.loc[results["house"] == "all", cols].round(3).to_string(index=False))
    print(results[cols].round(3).to_string(index=False))
    print(imp.head(12).round(1).to_string())


if __name__ == "__main__":
    main()
