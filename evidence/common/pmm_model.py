"""The model and data of PyMC-Marketing's `mmm_prior_sensitivity` notebook (PR #3011), with overrides.

Two models, as in that notebook:
  * "business": observational MMM + business prior on ROAS (observed Normal on the implied ROAS).
  * "lift":     the same + two lift tests on x1.

`build_and_fit` takes the changes a scenario makes (priors, sales likelihood, business prior,
lift tests) so that the reference fit and every full refit share one code path.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pymc as pm
from pymc_extras.prior import Prior

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

DATA_CSV = os.environ.get(  # set PMM_DATA_CSV to read a local copy
    "PMM_DATA_CSV", "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_roas_data.csv")

CHANNELS = ["x1", "x2"]
TARGET, DATE = "y", "date"

BUSINESS_PRIOR = {"channel": ["x1", "x2"], "roas": [100.0, 150.0], "sigma": [20.0, 50.0]}

# Priors of the notebook (`baseline_model_config`); the keys of `prior_overrides` replace entries.
BASE_CONFIG = {
    "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.5)),
    "gamma_fourier": Prior("Normal", mu=0, sigma=2, dims="fourier_mode"),
    "intercept_tvp_config": HSGPKwargs(m=50, L=None, eta_lam=1.0, ls_mu=5.0, ls_sigma=10.0, cov_func=None),
    "adstock_alpha": Prior("Beta", alpha=2, beta=3, dims="channel"),
    "saturation_lam": Prior("Gamma", alpha=2, beta=2, dims="channel"),
    "saturation_beta": Prior("HalfNormal", sigma=1, dims="channel"),
}

SAMPLER = {"tune": 1_000, "chains": 6, "cores": 6, "draws": 2_000,
           "nuts": {"adaptation": "draw_diag", "target_accept": 0.94}}
LIFT_SAMPLER = {**SAMPLER, "nuts": {"adaptation": "draw_diag", "target_accept": 0.96}}


def lift_table(X: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
    """The two synthetic lift tests of the notebook (x1, spend changes 0.25 and 0.8)."""
    return pd.DataFrame({
        "channel": ["x1", "x1"],
        "x": [0.25, 0.8],
        "delta_x": [0.25, 0.8],
        "delta_y": [23.34703279570842, 74.71050494626694],
        "sigma": [sigma, sigma],
        "date": pd.to_datetime([X["date"].max() - pd.Timedelta(weeks=50),
                                X["date"].max() - pd.Timedelta(weeks=14)]),
    })


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_CSV, parse_dates=[DATE])[[DATE, *CHANNELS, TARGET]]
    return df.drop(columns=[TARGET]), df[TARGET]


def build_mmm(model_config: dict) -> MMM:
    return MMM(
        adstock=GeometricAdstock(l_max=4), saturation=LogisticSaturation(),
        date_column=DATE, channel_columns=CHANNELS, target_column=TARGET,
        time_varying_intercept=True, time_varying_media=False, yearly_seasonality=5,
        model_config=model_config,
    )


def build_model(
    model: str = "business",
    prior_overrides: dict | None = None,
    sales_likelihood: Prior | None = None,
    business_prior: dict | None = None,
    lift_sigma: float = 3.0,
    lift_likelihood: Prior | None = None,
    lift_rows: list[int] | None = None,
) -> tuple[MMM, pd.DataFrame, pd.Series]:
    """Build (not fit) the model.

    Args:
      model: "business" or "lift".
      prior_overrides: {config key: Prior} replacing entries of BASE_CONFIG.
      sales_likelihood: replaces the sales likelihood, e.g.
        Prior("StudentT", nu=5, sigma=Prior("HalfNormal", sigma=0.5)).
      business_prior: replaces BUSINESS_PRIOR ({"channel", "roas", "sigma"}); None keeps it.
      lift_sigma, lift_likelihood, lift_rows: lift-test sigma, family (default Normal) and the
        rows of `lift_table` to keep (default both). Ignored for model="business".
    """
    X, y = load_data()
    config = {**BASE_CONFIG, **(prior_overrides or {})}
    if sales_likelihood is not None:
        config["likelihood"] = sales_likelihood
    mmm = build_mmm(config)
    mmm.build_model(X, y)
    if model == "lift":
        lt = lift_table(X, lift_sigma)
        if lift_rows is not None:
            lt = lt.iloc[lift_rows].reset_index(drop=True)
        if len(lt):
            mmm.add_lift_test_measurements(df_lift_test=lt, likelihood=lift_likelihood)
    elif model != "business":
        raise ValueError(model)
    mmm.add_original_scale_contribution_variable(
        var=["channel_contribution", "fourier_contribution", "intercept_contribution"])
    if business_prior != "none":
        mmm.add_cost_per_target_calibration(
            data=X, calibration_data=pd.DataFrame(business_prior or BUSINESS_PRIOR),
            name_prefix="business_prior", target_column="roas", target_per_cost=True)
    return mmm, X, y


def add_derived_groups(mmm: MMM) -> MMM:
    """Log-likelihood and log-prior groups, and the ROAS variable, as in the notebook.

    The business prior is moved from `log_likelihood` to `log_prior` (it is a prior in meaning).
    """
    with mmm.model:
        pm.compute_log_likelihood(mmm.idata)
        pm.stats.compute_log_prior(mmm.idata)
    if "business_prior" in mmm.idata["log_likelihood"].data_vars:
        mmm.idata["log_prior"]["business_prior"] = mmm.idata["log_likelihood"]["business_prior"]
        del mmm.idata["log_likelihood"]["business_prior"]
    mmm.idata["posterior"]["ROAS"] = (
        mmm.incrementality.compute_incremental_contribution("all_time")
        / mmm.idata["constant_data"]["channel_data"].sum("date"))
    return mmm


def build_and_fit(seed: int, model: str = "business", **scenario) -> MMM:
    """Build, fit and add the derived groups. `scenario` is passed to `build_model`."""
    mmm, X, y = build_model(model, **scenario)
    sampler = LIFT_SAMPLER if model == "lift" else SAMPLER
    mmm.fit(X, y, random_seed=np.random.default_rng(seed), **sampler)
    return add_derived_groups(mmm)
