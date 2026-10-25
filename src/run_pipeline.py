"""Complaints forecast pipeline: new data in -> validate -> clean -> retrain -> candidate forecast out.

Usage (from the repo root):
    python src/run_pipeline.py            # to build a candidate model and forecast
    python src/run_pipeline.py --promote  # after revision, to move the candidate to production

The modelling choices are explained in lab/analysis_walkthrough.ipynb.
"""
import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed: plots go straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

# Paths are built from this file's location, so the script runs wherever the repo is cloned
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "raw_data.csv"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

H = 90               # forecast horizon, and length of the holdout used to check each new model
K = 2                # annual Fourier pairs
ORDER = (1, 0, 0)    # AR(1) errors around a linear trend (see the DHR sanity check in the notebook)
RECENT_DAYS = 365    # interval widths reflect the residual spread of the last full year
EPOCH = pd.Timestamp("2000-01-01")  # fixed origin, so annual terms line up between training and forecast dates
REQUIRED = ["date", "complaints"]   # the only columns the model uses; others may come and go


def fail(message):
    sys.exit(f"ERROR: {message}")


# ---------------------------------------------------------------- data

def load_data(path):
    """Read the CSV, check it, and return complaints on a complete daily calendar (missing days as NaN)."""
    if not path.exists():
        fail(f"data file not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        fail(f"missing required columns: {missing}")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        fail(f"{dates.isna().sum()} dates could not be read")
    if dates.duplicated().any():
        fail(f"duplicate dates: {dates[dates.duplicated()].dt.date.tolist()[:5]}")

    y = pd.to_numeric(df["complaints"], errors="coerce")
    if (df["complaints"].notna() & y.isna()).any():
        fail("complaints contains non-numeric values")
    if (y < 0).any() or (y.dropna() % 1 != 0).any():
        fail("complaints must be whole, non-negative counts")

    y = pd.Series(y.to_numpy(), index=dates, name="complaints").sort_index()
    y = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq="D"))

    # Two annual cycles before the holdout, or the annual shape can't be separated from the trend
    if len(y) < 2 * 365 + H:
        fail(f"need at least {2 * 365 + H} days of history, got {len(y)}")
    return y


def clean_target(y):
    """Fill missing and extreme days using only the data passed in, so it never sees past its last date.
    Same rules as the notebook: same-weekday baseline, rolling 13-week robust scale, |z| > 5."""
    logy = np.log(y)
    baseline = logy.groupby(logy.index.dayofweek).transform(lambda s: s.rolling(5, center=True, min_periods=3).median())
    resid = logy - baseline
    w = 91
    centre = resid.rolling(w, center=True, min_periods=w // 2).median()
    scale = 1.4826 * (resid - centre).abs().rolling(w, center=True, min_periods=w // 2).median()
    extreme = ((resid - centre) / scale).abs() > 5
    to_fill = y.isna() | extreme
    clean = y.where(~to_fill, np.exp(baseline).round())
    if clean.isna().any():
        fail(f"gaps too long to fill (e.g. {clean.index[clean.isna()][0].date()}); the cleaning needs nearby same-weekday values")
    return clean


# ---------------------------------------------------------------- model

def calendar_features(dates, k=K):
    """Weekday dummies (Monday as reference) plus k sin/cos pairs for the annual cycle."""
    X = pd.DataFrame({f"dow_{d}": (dates.dayofweek == d).astype(float) for d in range(1, 7)}, index=dates)
    t = (dates - EPOCH).days.to_numpy()
    for j in range(1, k + 1):
        X[f"sin{j}"] = np.sin(2 * np.pi * j * t / 365.25)
        X[f"cos{j}"] = np.cos(2 * np.pi * j * t / 365.25)
    return X


def fit_dhr(y, k=K):
    # innovations_mle estimates the regression part by GLS; the default optimiser failed to converge on some windows
    return ARIMA(np.log(y), exog=calendar_features(y.index, k), order=ORDER, trend="ct").fit(method="innovations_mle")


def forecast_dhr(res, h, k=K):
    """Forecast h days after the model's last training date, with intervals widened to the recent residual spread."""
    future = pd.date_range(res.fittedvalues.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    fc = res.get_forecast(steps=h, exog=calendar_features(future, k))
    log_mean = fc.predicted_mean.to_numpy()
    widen = res.resid.iloc[-RECENT_DAYS:].std() / np.sqrt(res.params["sigma2"])
    out = pd.DataFrame({"forecast": np.exp(log_mean)}, index=future)
    for lvl in (80, 95):
        half = fc.conf_int(alpha=1 - lvl / 100).to_numpy()[:, 1] - log_mean
        out[f"lo{lvl}"] = np.exp(log_mean - widen * half)
        out[f"hi{lvl}"] = np.exp(log_mean + widen * half)
    out.index.name = "date"
    return out


def seasonal_naive(train, h):
    """Repeat the last observed week; standard seasonal-naive intervals (sd grows with each extra week ahead)."""
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    mean = np.resize(train.iloc[-7:].to_numpy(), h)
    sd = (train - train.shift(7)).std() * np.sqrt(np.arange(h) // 7 + 1)
    out = pd.DataFrame({"forecast": mean}, index=idx)
    for lvl in (80, 95):
        zq = stats.norm.ppf(0.5 + lvl / 200)
        out[f"lo{lvl}"], out[f"hi{lvl}"] = mean - zq * sd, mean + zq * sd
    return out


def score(actual, fc, train):
    """Same metrics as the notebook backtest. Days without a recorded target are never scored."""
    a = actual.dropna()
    f = fc.loc[a.index, :]
    err = a - f["forecast"]
    days = pd.Series(1, index=actual.index).resample("W").sum()
    weekly = pd.DataFrame({"a": a, "f": f["forecast"]}).resample("W").sum().loc[days[days == 7].index]
    metrics = {
        "MAE": err.abs().mean(),
        "MASE": err.abs().mean() / (train - train.shift(7)).abs().mean(),
        "weekly_APE_%": 100 * ((weekly["a"] - weekly["f"]).abs() / weekly["a"]).mean(),
        "bias_%": 100 * (f["forecast"].sum() - a.sum()) / a.sum(),
        "cover_80": ((a >= f["lo80"]) & (a <= f["hi80"])).mean(),
        "cover_95": ((a >= f["lo95"]) & (a <= f["hi95"])).mean(),
        "width_80_%": 100 * (f["hi80"] - f["lo80"]).mean() / a.mean(),
    }
    return {name: round(float(v), 3) for name, v in metrics.items()}


# ---------------------------------------------------------------- outputs

def write_outputs(folder, forecast, y_raw, y_clean):
    folder.mkdir(parents=True, exist_ok=True)
    forecast = forecast.round().astype(int)
    forecast.to_csv(folder / "forecast.csv")

    # Daily interval bounds don't add up to a weekly interval, so weekly totals are point forecasts only
    weekly = forecast["forecast"].resample("W-MON", label="left", closed="left").agg(["sum", "count"])
    weekly.columns = ["total", "days"]
    weekly.index.name = "week_starting"
    weekly.to_csv(folder / "weekly_totals.csv")

    start = y_raw.index[-1] - pd.DateOffset(years=2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    hist = y_raw.loc[start:]
    ax1.plot(hist.index, hist, lw=0.6, color="grey", alpha=0.6, label="actual")
    ax1.plot(hist.index, hist.rolling(28, center=True, min_periods=14).mean(), color="black", lw=1.5, label="actual, 28-day mean")
    ax1.fill_between(forecast.index, forecast["lo95"], forecast["hi95"], color="C0", alpha=0.15, label="95% interval")
    ax1.fill_between(forecast.index, forecast["lo80"], forecast["hi80"], color="C0", alpha=0.3, label="80% interval")
    ax1.plot(forecast.index, forecast["forecast"], color="C0", lw=1.2, label="forecast (median)")
    ax1.set_title(f"Daily complaints and the forecast for {forecast.index[0]:%d %b %Y} – {forecast.index[-1]:%d %b %Y}")
    ax1.legend(loc="upper left")

    wk = y_clean.loc[start:].resample("W-MON", label="left", closed="left").agg(["sum", "count"])
    wk_hist = wk.loc[wk["count"] == 7, "sum"]
    wk_fc = weekly.loc[weekly["days"] == 7, "total"]
    ax2.plot(wk_hist.index, wk_hist, color="grey", marker=".", label="actual weekly total (gaps filled)")
    ax2.plot(wk_fc.index, wk_fc, color="C0", marker=".", label="forecast weekly total")
    ax2.set_title("Weekly totals (complete weeks)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- steps

def build_candidate():
    y_raw = load_data(DATA)
    y_clean = clean_target(y_raw)

    # Holdout check: a model fitted without the newest H days, scored on them.
    # Cleaning uses the training part only, so the holdout never leaks into training
    holdout_start = y_raw.index[-H]
    train = clean_target(y_raw.loc[: holdout_start - pd.Timedelta(days=1)])
    actual = y_raw.loc[holdout_start:]
    scores = {
        "candidate": score(actual, forecast_dhr(fit_dhr(train), H), train),
        "seasonal_naive": score(actual, seasonal_naive(train, H), train),
    }

    # Production is only a fair comparison if it never saw the holdout days
    production_note = "no production model yet"
    prod_model = MODELS / "production" / "model.pkl"
    if prod_model.exists():
        with open(prod_model, "rb") as f:
            prod = pickle.load(f)
        prod_k = json.loads((MODELS / "production" / "metrics.json").read_text())["config"]["fourier_pairs"]
        prod_end = prod.fittedvalues.index[-1]
        if prod_end < holdout_start:
            steps = (y_raw.index[-1] - prod_end).days
            scores["production"] = score(actual, forecast_dhr(prod, steps, prod_k).loc[holdout_start:], train)
            production_note = f"production (trained to {prod_end.date()}) scored on the same holdout"
        else:
            production_note = (f"not compared: production was trained to {prod_end.date()}, which overlaps the holdout "
                               f"(needs {H} days of new data since then)")

    beats_naive = scores["candidate"]["MASE"] < scores["seasonal_naive"]["MASE"]
    vs_prod = None
    if "production" in scores:
        # A tie is expected when production was trained exactly up to the holdout: same recipe, same data
        diff = scores["candidate"]["MASE"] - scores["production"]["MASE"]
        vs_prod = "tie" if diff == 0 else ("better" if diff < 0 else "worse")

    # Final candidate: refit on everything
    res = fit_dhr(y_clean)
    forecast = forecast_dhr(res, H)

    cand_models = MODELS / "candidate"
    cand_models.mkdir(parents=True, exist_ok=True)
    with open(cand_models / "model.pkl", "wb") as f:
        pickle.dump(res, f)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"start": str(y_raw.index[0].date()), "end": str(y_raw.index[-1].date()),
                 "days": len(y_raw), "days_without_target": int(y_raw.isna().sum())},
        "config": {"model": "DHR", "fourier_pairs": K, "order": list(ORDER), "recent_days": RECENT_DAYS, "horizon": H},
        "trained_until": str(y_clean.index[-1].date()),
        "forecast_window": [str(forecast.index[0].date()), str(forecast.index[-1].date())],
        "holdout": {"window": [str(actual.index[0].date()), str(actual.index[-1].date())],
                    "scores": scores, "production_comparison": production_note},
        "beats_seasonal_naive": beats_naive,
        "candidate_vs_production": vs_prod,
    }
    (cand_models / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_outputs(OUTPUTS / "candidate", forecast, y_raw, y_clean)

    print(f"Data: {metrics['data']['start']} to {metrics['data']['end']} ({len(y_raw)} days)")
    print(f"Holdout {metrics['holdout']['window'][0]} to {metrics['holdout']['window'][1]}:")
    print(pd.DataFrame(scores).T.to_string())
    print(f"Production: {production_note}")
    print(f"Beats seasonal naive: {beats_naive} | candidate vs production: {vs_prod}")
    print(f"Candidate written to {cand_models.relative_to(ROOT)}/ and {(OUTPUTS / 'candidate').relative_to(ROOT)}/")
    print("Review the forecast plot and weekly totals, then run with --promote to move it to production."
          if beats_naive else "Candidate does NOT beat seasonal naive: do not promote. Investigate before the next run.")


def promote():
    metrics_file = MODELS / "candidate" / "metrics.json"
    if not metrics_file.exists():
        fail("no candidate to promote; run the pipeline first")
    metrics = json.loads(metrics_file.read_text())
    if not metrics["beats_seasonal_naive"]:
        fail("the candidate did not beat seasonal naive on the holdout; refusing to promote")

    for base in (MODELS, OUTPUTS):
        if (base / "production").exists():
            shutil.rmtree(base / "production")
        shutil.copytree(base / "candidate", base / "production")

    metrics["promoted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (MODELS / "production" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Promoted the candidate trained to {metrics['trained_until']} "
          f"(forecast {metrics['forecast_window'][0]} to {metrics['forecast_window'][1]}) to production.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", action="store_true", help="copy the reviewed candidate to production")
    args = parser.parse_args()
    promote() if args.promote else build_candidate()


if __name__ == "__main__":
    main()