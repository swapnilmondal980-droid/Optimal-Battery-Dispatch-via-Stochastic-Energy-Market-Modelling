from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(r"D:/ISI/SEM2/Optimisation/Project/Finalised opt dataset.csv")
OUTPUT_DIR = Path(__file__).resolve().parent

RNG_SEED = 2025
HORIZON_DAYS = 30
HORIZON_HOURS = 24 * HORIZON_DAYS
N_PRICE_PATHS = 500
TARGET_SPREAD_QUANTILE = 0.95

E_MAX = 2000.0
E_MIN = 0.0
E_0 = 0.5 * E_MAX
C_MAX = 500.0
D_MAX = 500.0
ETA_C = 0.97
ETA_D = 0.97

STATE_STEP_MWH = 10.0
POWER_STEP_MW = 50.0
TOL = 1e-9


def read_and_clean_data(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]

    needed = [
        "utc_timestamp",
        "DE_load_actual_entsoe_transparency",
        "DE_LU_price_day_ahead",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual",
        "DE_solar_capacity",
        "DE_wind_capacity",
    ]
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = raw.rename(
        columns={
            "utc_timestamp": "timestamp",
            "DE_load_actual_entsoe_transparency": "load_mw",
            "DE_LU_price_day_ahead": "price_eur_mwh",
            "DE_solar_generation_actual": "solar_mw",
            "DE_wind_generation_actual": "wind_mw",
            "DE_solar_capacity": "solar_cap",
            "DE_wind_capacity": "wind_cap",
        }
    )[
        ["timestamp", "load_mw", "price_eur_mwh", "solar_mw", "wind_mw", "solar_cap", "wind_cap"]
    ].copy()

    for col in ["load_mw", "price_eur_mwh", "solar_mw", "wind_mw", "solar_cap", "wind_cap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    except TypeError:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df = df.loc[
        df["timestamp"].notna()
        & df["load_mw"].notna()
        & df["price_eur_mwh"].notna()
        & df["solar_mw"].notna()
        & df["wind_mw"].notna()
        & df["solar_cap"].notna()
        & df["wind_cap"].notna()
        & (df["solar_cap"] > 0)
        & (df["wind_cap"] > 0)
    ].copy()

    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["month_label"] = df["timestamp"].dt.strftime("%Y-%m")
    df["days_in_month"] = df["timestamp"].dt.days_in_month
    df["solar_cf"] = np.clip(df["solar_mw"] / df["solar_cap"], 0.0, 1.0)
    df["wind_cf"] = np.clip(df["wind_mw"] / df["wind_cap"], 0.0, 1.0)

    return df.sort_values("timestamp").reset_index(drop=True)


def fit_weibull_mle(x: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[(x > 0.0) & np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Weibull fit received no positive observations.")

    logs = np.log(x)
    std_logs = float(np.std(logs))
    mean_logs = float(np.mean(logs))
    k = max(1.0 / max(std_logs, 1e-6), 0.5)

    for _ in range(max_iter):
        xk = x**k
        s0 = np.sum(xk)
        s1 = np.sum(xk * logs)
        s2 = np.sum(xk * logs * logs)

        f = 1.0 / k + mean_logs - s1 / s0
        fp = -1.0 / (k * k) - (s2 / s0 - (s1 / s0) ** 2)
        step = f / fp
        new_k = k - step

        if not np.isfinite(new_k) or new_k <= 0.0:
            new_k = k / 2.0

        if abs(new_k - k) < tol:
            k = new_k
            break
        k = new_k

    scale = float(np.mean(x**k) ** (1.0 / k))
    return float(k), scale


def normal_cdf(x: float, mean: float, sd: float) -> float:
    if sd <= 0.0:
        return 0.5 if x == mean else float(x > mean)
    z = (x - mean) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def weibull_ppf(p: float, shape: float, scale: float) -> float:
    p = min(max(p, 1e-8), 1.0 - 1e-8)
    return scale * (-math.log(1.0 - p)) ** (1.0 / shape)


def fit_models(df: pd.DataFrame) -> dict[str, object]:
    price_x = df["price_eur_mwh"].to_numpy()[:-1]
    price_y = df["price_eur_mwh"].to_numpy()[1:]
    X_price = np.column_stack([np.ones_like(price_x), price_x])
    alpha_p, beta_p_raw = np.linalg.lstsq(X_price, price_y, rcond=None)[0]
    beta_p = float(np.clip(beta_p_raw, 1e-8, 0.999999))
    price_resid = price_y - (alpha_p + beta_p * price_x)
    sigma_eta_p = float(np.sqrt(np.mean(price_resid**2)))
    mu_p = float(alpha_p / (1.0 - beta_p))
    stationary_sd_p = sigma_eta_p / math.sqrt(max(1.0 - beta_p**2, 1e-8))

    price_hourly_profile = (
        df.groupby("hour")["price_eur_mwh"]
        .mean()
        .reindex(range(24))
        .interpolate(limit_direction="both")
    )

    wind_fit_df = df.loc[(df["wind_cf"] > 0.001) & (df["wind_cf"] < 0.999), ["wind_cf"]].copy()
    wind_x = wind_fit_df["wind_cf"].to_numpy()[:-1]
    wind_y = wind_fit_df["wind_cf"].to_numpy()[1:]
    X_wind = np.column_stack([np.ones_like(wind_x), wind_x])
    alpha_w, beta_w_raw = np.linalg.lstsq(X_wind, wind_y, rcond=None)[0]
    beta_w = float(np.clip(beta_w_raw, 1e-8, 0.999999))
    wind_resid = wind_y - (alpha_w + beta_w * wind_x)
    sigma_eta_w = float(np.sqrt(np.mean(wind_resid**2)))
    mu_w = float(alpha_w / (1.0 - beta_w))
    stationary_sd_w = sigma_eta_w / math.sqrt(max(1.0 - beta_w**2, 1e-8))

    weibull_shape, weibull_scale = fit_weibull_mle(wind_fit_df["wind_cf"].to_numpy())
    wind_mean_cf = weibull_scale * math.gamma(1.0 + 1.0 / weibull_shape)

    load_params = (
        df.groupby("hour")
        .agg(mu_load=("load_mw", "mean"), sigma_load=("load_mw", "std"))
        .reindex(range(24))
        .reset_index()
    )
    load_params["sigma_load"] = load_params["sigma_load"].fillna(0.0)

    solar_params = (
        df.groupby("hour")
        .agg(mu_solar=("solar_cf", "mean"), var_solar=("solar_cf", "var"))
        .reindex(range(24))
        .reset_index()
    )
    solar_params["mu_solar"] = solar_params["mu_solar"].fillna(0.0)
    solar_params["var_solar"] = solar_params["var_solar"].fillna(0.0)

    mu_beta = np.clip(solar_params["mu_solar"].to_numpy(dtype=float), 0.001, 0.999)
    var_beta = np.maximum(solar_params["var_solar"].to_numpy(dtype=float), 1e-6)
    common = np.maximum(mu_beta * (1.0 - mu_beta) / var_beta - 1.0, 0.1)
    alpha_s = mu_beta * common
    beta_s = (1.0 - mu_beta) * common

    night_mask = solar_params["mu_solar"].to_numpy(dtype=float) < 0.01
    alpha_s = np.where(night_mask, 0.1, alpha_s)
    beta_s = np.where(night_mask, 5.0, beta_s)

    solar_params["alpha_s"] = alpha_s
    solar_params["beta_s"] = beta_s

    return {
        "price_alpha": float(alpha_p),
        "price_beta": beta_p,
        "price_mu": mu_p,
        "price_sigma_eta": sigma_eta_p,
        "price_stationary_sd": stationary_sd_p,
        "price_hourly_profile": price_hourly_profile,
        "wind_alpha": float(alpha_w),
        "wind_beta": beta_w,
        "wind_mu": mu_w,
        "wind_sigma_eta": sigma_eta_w,
        "wind_stationary_sd": stationary_sd_w,
        "wind_weibull_shape": weibull_shape,
        "wind_weibull_scale": weibull_scale,
        "wind_mean_cf": wind_mean_cf,
        "load_params": load_params,
        "solar_params": solar_params,
        "solar_cap_sim": float(df["solar_cap"].median()),
        "wind_cap_sim": float(df["wind_cap"].median()),
    }


def build_simulated_scenario(models: dict[str, object], rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, float]]:
    alpha_p = float(models["price_alpha"])
    beta_p = float(models["price_beta"])
    mu_p = float(models["price_mu"])
    sigma_eta_p = float(models["price_sigma_eta"])
    stationary_sd_p = float(models["price_stationary_sd"])

    price_bank: list[np.ndarray] = []
    price_spreads: list[float] = []

    for _ in range(N_PRICE_PATHS):
        path = np.empty(HORIZON_HOURS, dtype=float)
        path[0] = rng.normal(mu_p, stationary_sd_p)
        for t in range(1, HORIZON_HOURS):
            path[t] = alpha_p + beta_p * path[t - 1] + rng.normal(0.0, sigma_eta_p)
        price_bank.append(path)
        price_spreads.append(float(np.max(path) - np.min(path)))

    price_spreads_arr = np.asarray(price_spreads, dtype=float)
    target_spread = float(np.quantile(price_spreads_arr, TARGET_SPREAD_QUANTILE))
    selected_price_idx = int(np.argmin(np.abs(price_spreads_arr - target_spread)))
    sim_price = price_bank[selected_price_idx]

    alpha_w = float(models["wind_alpha"])
    beta_w = float(models["wind_beta"])
    mu_w = float(models["wind_mu"])
    sigma_eta_w = float(models["wind_sigma_eta"])
    stationary_sd_w = float(models["wind_stationary_sd"])
    weibull_shape = float(models["wind_weibull_shape"])
    weibull_scale = float(models["wind_weibull_scale"])

    sim_wind_cf = np.empty(HORIZON_HOURS, dtype=float)
    sim_wind_cf[0] = np.clip(mu_w, 1e-4, 0.999)
    for t in range(1, HORIZON_HOURS):
        w_raw = alpha_w + beta_w * sim_wind_cf[t - 1] + rng.normal(0.0, sigma_eta_w)
        p_u = normal_cdf(w_raw, mu_w, stationary_sd_w)
        sim_wind_cf[t] = weibull_ppf(min(max(p_u, 1e-4), 1.0 - 1e-4), weibull_shape, weibull_scale)

    load_params = models["load_params"].set_index("hour")
    solar_params = models["solar_params"].set_index("hour")
    solar_cap_sim = float(models["solar_cap_sim"])
    wind_cap_sim = float(models["wind_cap_sim"])

    sim_load = np.empty(HORIZON_HOURS, dtype=float)
    sim_solar_cf = np.empty(HORIZON_HOURS, dtype=float)

    for t in range(HORIZON_HOURS):
        hr = t % 24
        mu_load = float(load_params.loc[hr, "mu_load"])
        sigma_load = float(load_params.loc[hr, "sigma_load"])
        sim_load[t] = max(0.0, rng.normal(mu_load, sigma_load))

        alpha_s = float(solar_params.loc[hr, "alpha_s"])
        beta_s = float(solar_params.loc[hr, "beta_s"])
        sim_solar_cf[t] = rng.beta(alpha_s, beta_s)

    sim_solar = sim_solar_cf * solar_cap_sim
    sim_wind = sim_wind_cf * wind_cap_sim

    sim_df = pd.DataFrame(
        {
            "hour": np.arange(1, HORIZON_HOURS + 1),
            "hour_of_day": np.arange(HORIZON_HOURS) % 24,
            "date": "Simulated Month",
            "price_eur_mwh": sim_price,
            "load_mw": sim_load,
            "solar_mw": sim_solar,
            "wind_mw": sim_wind,
            "solar_cap": solar_cap_sim,
            "wind_cap": wind_cap_sim,
        }
    )
    sim_df["net_load"] = sim_df["load_mw"] - sim_df["solar_mw"] - sim_df["wind_mw"]

    sim_meta = {
        "selected_price_path": float(selected_price_idx + 1),
        "price_spread": float(price_spreads_arr[selected_price_idx]),
        "target_spread": target_spread,
    }
    return sim_df, sim_meta


def build_real_historical_scenario(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    historical_months = (
        df.groupby(["month_label", "days_in_month"], as_index=False)
        .agg(
            n_hours=("price_eur_mwh", "size"),
            price_min=("price_eur_mwh", "min"),
            price_max=("price_eur_mwh", "max"),
            avg_price=("price_eur_mwh", "mean"),
        )
    )
    historical_months["price_spread"] = historical_months["price_max"] - historical_months["price_min"]
    historical_months = historical_months.loc[
        (historical_months["days_in_month"] == HORIZON_DAYS)
        & (historical_months["n_hours"] >= HORIZON_HOURS)
    ].copy()
    if historical_months.empty:
        raise ValueError("No complete 30-day historical month found in the dataset.")

    selected = historical_months.sort_values("price_spread", ascending=False).iloc[0]
    validation_month = str(selected["month_label"])

    real_df = (
        df.loc[df["month_label"] == validation_month]
        .sort_values("timestamp")
        .head(HORIZON_HOURS)
        .reset_index(drop=True)
    )
    if len(real_df) != HORIZON_HOURS:
        raise ValueError("Selected historical scenario does not contain the full 720-hour monthly horizon.")

    real_df = real_df[
        ["timestamp", "date", "price_eur_mwh", "load_mw", "solar_mw", "wind_mw", "solar_cap", "wind_cap", "hour"]
    ].copy()
    real_df["hour_of_day"] = real_df["timestamp"].dt.hour
    real_df["hour"] = np.arange(1, HORIZON_HOURS + 1)
    real_df["date"] = validation_month
    real_df["net_load"] = real_df["load_mw"] - real_df["solar_mw"] - real_df["wind_mw"]

    real_meta = {
        "reference_date": validation_month,
        "price_spread": float(selected["price_spread"]),
    }
    return real_df.drop(columns=["timestamp"]), real_meta


def build_expected_scenario(scenario_df: pd.DataFrame, models: dict[str, object]) -> pd.DataFrame:
    scenario_df = scenario_df.sort_values("hour").reset_index(drop=True).copy()
    if len(scenario_df) != HORIZON_HOURS:
        raise ValueError(f"Expected-value scenario must contain exactly {HORIZON_HOURS} hours.")

    beta_p = float(models["price_beta"])
    baseline = models["price_hourly_profile"].loc[scenario_df["hour_of_day"]].to_numpy(dtype=float)
    expected_price = baseline + (beta_p ** np.arange(HORIZON_HOURS)) * (
        float(scenario_df["price_eur_mwh"].iloc[0]) - float(baseline[0])
    )

    load_params = models["load_params"].set_index("hour")
    solar_params = models["solar_params"].set_index("hour")
    expected_load = load_params.loc[scenario_df["hour_of_day"], "mu_load"].to_numpy(dtype=float)
    expected_solar_cf = solar_params.loc[scenario_df["hour_of_day"], "mu_solar"].to_numpy(dtype=float)
    expected_wind_cf = np.full(HORIZON_HOURS, float(models["wind_mean_cf"]), dtype=float)

    expected_df = scenario_df.copy()
    expected_df["price_eur_mwh"] = expected_price
    expected_df["load_mw"] = expected_load
    expected_df["solar_mw"] = expected_solar_cf * expected_df["solar_cap"].to_numpy(dtype=float)
    expected_df["wind_mw"] = expected_wind_cf * expected_df["wind_cap"].to_numpy(dtype=float)
    expected_df["net_load"] = expected_df["load_mw"] - expected_df["solar_mw"] - expected_df["wind_mw"]
    return expected_df


def solve_dispatch_dp(
    scenario_df: pd.DataFrame,
    scenario_group: str,
    objective_basis: str,
    reference_date: str,
    state_step_mwh: float = STATE_STEP_MWH,
    power_step_mw: float = POWER_STEP_MW,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    scenario_df = scenario_df.sort_values("hour").reset_index(drop=True).copy()
    prices = scenario_df["price_eur_mwh"].to_numpy(dtype=float)
    net_load = scenario_df["net_load"].to_numpy(dtype=float)
    n_hours = len(scenario_df)

    states = np.arange(E_MIN, E_MAX + state_step_mwh / 2.0, state_step_mwh)
    start_state_idx = int(np.argmin(np.abs(states - E_0)))
    actions = np.concatenate(
        [
            -np.arange(D_MAX, 0.0, -power_step_mw),
            np.array([0.0]),
            np.arange(power_step_mw, C_MAX + power_step_mw / 2.0, power_step_mw),
        ]
    )

    value = np.full((n_hours + 1, len(states)), np.inf, dtype=float)
    next_state_choice = np.full((n_hours, len(states)), -1, dtype=int)
    action_choice = np.full((n_hours, len(states)), 0.0, dtype=float)

    value[n_hours, start_state_idx] = 0.0

    for t in range(n_hours - 1, -1, -1):
        price = prices[t]
        for s_idx, state in enumerate(states):
            best_value = np.inf
            best_next_idx = -1
            best_action = 0.0

            for action in actions:
                if action > 0.0:
                    max_charge = min(C_MAX, (E_MAX - state) / ETA_C)
                    if action > max_charge + TOL:
                        continue
                    next_state = state + ETA_C * action
                    immediate_cost = price * action
                elif action < 0.0:
                    discharge = -action
                    max_discharge = min(D_MAX, (state - E_MIN) * ETA_D)
                    if discharge > max_discharge + TOL:
                        continue
                    next_state = state - discharge / ETA_D
                    immediate_cost = -price * discharge
                else:
                    next_state = state
                    immediate_cost = 0.0

                next_idx = int(np.argmin(np.abs(states - next_state)))
                candidate = immediate_cost + value[t + 1, next_idx]
                if candidate < best_value:
                    best_value = candidate
                    best_next_idx = next_idx
                    best_action = action

            value[t, s_idx] = best_value
            next_state_choice[t, s_idx] = best_next_idx
            action_choice[t, s_idx] = best_action

    charge = np.zeros(n_hours, dtype=float)
    discharge = np.zeros(n_hours, dtype=float)
    soc = np.zeros(n_hours, dtype=float)

    current_idx = start_state_idx
    for t in range(n_hours):
        action = action_choice[t, current_idx]
        if action > 0.0:
            charge[t] = action
        elif action < 0.0:
            discharge[t] = -action

        next_idx = next_state_choice[t, current_idx]
        if next_idx < 0:
            raise RuntimeError(
                f"Dynamic-programming solver failed to reconstruct a feasible schedule for {scenario_group} / {objective_basis}."
            )
        soc[t] = states[next_idx]
        current_idx = next_idx

    schedule = scenario_df.copy()
    schedule["scenario"] = scenario_group
    schedule["objective_basis"] = objective_basis
    schedule["reference_date"] = reference_date
    schedule["schedule_label"] = f"{scenario_group} | {objective_basis}"
    schedule["charge_mw"] = charge
    schedule["discharge_mw"] = discharge
    schedule["soc_mwh"] = soc
    schedule["operating_mode"] = np.where(
        charge > TOL,
        "Charge",
        np.where(discharge > TOL, "Discharge", "Idle"),
    )
    schedule["simultaneous_flag"] = (schedule["charge_mw"] > TOL) & (schedule["discharge_mw"] > TOL)
    schedule["battery_profit"] = schedule["price_eur_mwh"] * (schedule["discharge_mw"] - schedule["charge_mw"])
    schedule["cost_no_batt"] = schedule["price_eur_mwh"] * schedule["net_load"]
    schedule["cost_w_batt"] = schedule["price_eur_mwh"] * (
        schedule["net_load"] + schedule["charge_mw"] - schedule["discharge_mw"]
    )

    simultaneous_hours = int(schedule["simultaneous_flag"].sum())
    if simultaneous_hours > 0:
        raise ValueError(
            f"Charge and discharge happened together in {simultaneous_hours} hours for {scenario_group} / {objective_basis}."
        )

    total_no_batt = float(schedule["cost_no_batt"].sum())
    total_w_batt = float(schedule["cost_w_batt"].sum())
    absolute_saving = total_no_batt - total_w_batt
    pct_savings = 100.0 * absolute_saving / abs(total_no_batt) if abs(total_no_batt) > TOL else np.nan

    summary = {
        "scenario": scenario_group,
        "objective_basis": objective_basis,
        "reference_date": reference_date,
        "total_no_batt": total_no_batt,
        "total_w_batt": total_w_batt,
        "absolute_saving": absolute_saving,
        "pct_savings": pct_savings,
        "battery_arbitrage_profit": float(schedule["battery_profit"].sum()),
        "profit_per_mwh_capacity": float(schedule["battery_profit"].sum()) / E_MAX,
        "price_spread": float(schedule["price_eur_mwh"].max() - schedule["price_eur_mwh"].min()),
        "charge_hours": int((schedule["charge_mw"] > TOL).sum()),
        "discharge_hours": int((schedule["discharge_mw"] > TOL).sum()),
        "idle_hours": int((schedule["operating_mode"] == "Idle").sum()),
        "simultaneous_hours": simultaneous_hours,
    }
    return schedule, summary


def build_display_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    display_df = summary_df.copy()
    display_df["sort_key"] = pd.Categorical(
        display_df["scenario"] + " | " + display_df["objective_basis"],
        categories=[
            "Simulated | Expected Value",
            "Simulated | Realised Path",
            "Real Historical | Expected Value",
            "Real Historical | Realised Path",
        ],
        ordered=True,
    )
    display_df = display_df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    return pd.DataFrame(
        {
            "Scenario": display_df["scenario"],
            "Objective Basis": display_df["objective_basis"],
            "Reference Date": display_df["reference_date"],
            "No Battery Cost (EUR)": display_df["total_no_batt"].map(lambda x: f"EUR {x:.2f}"),
            "With Battery Cost (EUR)": display_df["total_w_batt"].map(lambda x: f"EUR {x:.2f}"),
            "Absolute Saving (EUR)": display_df["absolute_saving"].map(lambda x: f"EUR {x:.2f}"),
            "Percentage Saving": display_df["pct_savings"].map(lambda x: f"{x:.2f} %"),
            "Battery Arbitrage Profit (EUR)": display_df["battery_arbitrage_profit"].map(lambda x: f"EUR {x:.2f}"),
            "Profit per MWh Capacity": display_df["profit_per_mwh_capacity"].map(lambda x: f"EUR {x:.2f} / MWh"),
            "Price Spread": display_df["price_spread"].map(lambda x: f"{x:.2f} EUR/MWh"),
            "Charge Hours": display_df["charge_hours"].astype(int),
            "Discharge Hours": display_df["discharge_hours"].astype(int),
            "Idle Hours": display_df["idle_hours"].astype(int),
        }
    )


def scale_value(value: float, data_min: float, data_max: float, pixel_min: float, pixel_max: float) -> float:
    if math.isclose(data_max, data_min):
        return (pixel_min + pixel_max) / 2.0
    ratio = (value - data_min) / (data_max - data_min)
    return pixel_max - ratio * (pixel_max - pixel_min)


def build_svg_plot(schedule_df: pd.DataFrame, title: str, output_path: Path) -> None:
    width = 1100
    height = 620
    left = 90
    right = 70
    plot_width = width - left - right

    top_top = 80
    top_bottom = 290
    bottom_top = 380
    bottom_bottom = 545

    hours = schedule_df["hour"].to_numpy(dtype=float)
    charge = schedule_df["charge_mw"].to_numpy(dtype=float)
    discharge = schedule_df["discharge_mw"].to_numpy(dtype=float)
    price = schedule_df["price_eur_mwh"].to_numpy(dtype=float)
    soc = schedule_df["soc_mwh"].to_numpy(dtype=float)

    power_limit = max(C_MAX, D_MAX) * 1.1
    power_min = -power_limit
    power_max = power_limit
    power_ticks = [-500.0, -250.0, 0.0, 250.0, 500.0]

    price_min = float(np.min(price))
    price_max = float(np.max(price))
    if math.isclose(price_min, price_max):
        price_min -= 1.0
        price_max += 1.0
    price_pad = 0.08 * (price_max - price_min)
    price_min -= price_pad
    price_max += price_pad

    soc_min = 0.0
    soc_max = E_MAX
    soc_ticks = [0.0, 500.0, 1000.0, 1500.0, 2000.0]

    x_centers = np.linspace(left + 18, left + plot_width - 18, len(hours))
    bar_half_width = max(0.35, min(6.0, plot_width / max(len(hours) * 1.6, 1)))

    zero_y = scale_value(0.0, power_min, power_max, top_top, top_bottom)
    price_points = []
    soc_points = []
    soc_fill_points = [(x_centers[0], bottom_bottom)]

    for x, p, s in zip(x_centers, price, soc):
        price_y = scale_value(float(p), price_min, price_max, top_top, top_bottom)
        soc_y = scale_value(float(s), soc_min, soc_max, bottom_top, bottom_bottom)
        price_points.append((x, price_y))
        soc_points.append((x, soc_y))
        soc_fill_points.append((x, soc_y))
    soc_fill_points.append((x_centers[-1], bottom_bottom))

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="38" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#1f2937">{html.escape(title)}</text>',
        f'<text x="{left}" y="62" font-family="Arial, sans-serif" font-size="14" fill="#6b7280">Blue = charge, red = discharge, purple = price</text>',
    ]

    for plot_top, plot_bottom in [(top_top, top_bottom), (bottom_top, bottom_bottom)]:
        parts.append(
            f'<rect x="{left}" y="{plot_top}" width="{plot_width}" height="{plot_bottom - plot_top}" fill="none" stroke="#d1d5db" stroke-width="1"/>'
        )

    for tick in power_ticks:
        y = scale_value(tick, power_min, power_max, top_top, top_bottom)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{int(tick)}</text>'
        )

    for tick in soc_ticks:
        y = scale_value(tick, soc_min, soc_max, bottom_top, bottom_bottom)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{int(tick)}</text>'
        )

    if len(hours) > 48:
        tick_indices = list(range(0, len(hours), 24))
        tick_labels = [f"D{idx + 1}" for idx in range(len(tick_indices))]
        tick_font_size = 9
    else:
        tick_indices = list(range(len(hours)))
        tick_labels = [str(idx + 1) for idx in range(len(hours))]
        tick_font_size = 11

    for idx, label in zip(tick_indices, tick_labels):
        x = x_centers[idx]
        parts.append(
            f'<line x1="{x:.2f}" y1="{top_top}" x2="{x:.2f}" y2="{top_bottom}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{x:.2f}" y1="{bottom_top}" x2="{x:.2f}" y2="{bottom_bottom}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top_bottom + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{tick_font_size}" fill="#4b5563">{label}</text>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{bottom_bottom + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{tick_font_size}" fill="#4b5563">{label}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" stroke="#9ca3af" stroke-width="1.2"/>'
    )

    for x, c, d in zip(x_centers, charge, discharge):
        if c > TOL:
            y = scale_value(c, power_min, power_max, top_top, top_bottom)
            height_rect = zero_y - y
            parts.append(
                f'<rect x="{x - bar_half_width:.2f}" y="{y:.2f}" width="{2 * bar_half_width:.2f}" height="{height_rect:.2f}" fill="#3498db" opacity="0.85"/>'
            )
        if d > TOL:
            y = scale_value(-d, power_min, power_max, top_top, top_bottom)
            height_rect = y - zero_y
            parts.append(
                f'<rect x="{x - bar_half_width:.2f}" y="{zero_y:.2f}" width="{2 * bar_half_width:.2f}" height="{height_rect:.2f}" fill="#e74c3c" opacity="0.85"/>'
            )

    parts.append(
        f'<polyline fill="none" stroke="#8e44ad" stroke-width="2.4" points="{polyline(price_points)}"/>'
    )
    if len(price_points) <= 72:
        for x, y in price_points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#8e44ad"/>')

    parts.append(
        f'<polygon points="{polyline(soc_fill_points)}" fill="#3498db" opacity="0.18"/>'
    )
    parts.append(
        f'<polyline fill="none" stroke="#2980b9" stroke-width="2.2" points="{polyline(soc_points)}"/>'
    )
    if len(soc_points) <= 72:
        for x, y in soc_points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#2980b9"/>')

    parts.extend(
        [
            f'<text x="{width - 18}" y="{top_top + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{price_max:.1f}</text>',
            f'<text x="{width - 18}" y="{top_bottom + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{price_min:.1f}</text>',
            f'<text x="{28}" y="{(top_top + top_bottom) / 2:.2f}" transform="rotate(-90 28 {(top_top + top_bottom) / 2:.2f})" font-family="Arial, sans-serif" font-size="13" fill="#374151">Battery Power (MW)</text>',
            f'<text x="{width - 24}" y="{(top_top + top_bottom) / 2:.2f}" transform="rotate(90 {width - 24} {(top_top + top_bottom) / 2:.2f})" font-family="Arial, sans-serif" font-size="13" fill="#374151">Price (EUR/MWh)</text>',
            f'<text x="{28}" y="{(bottom_top + bottom_bottom) / 2:.2f}" transform="rotate(-90 28 {(bottom_top + bottom_bottom) / 2:.2f})" font-family="Arial, sans-serif" font-size="13" fill="#374151">SoC (MWh)</text>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">{"Day" if len(hours) > 48 else "Hour"}</text>',
        ]
    )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_html(display_df: pd.DataFrame, graph_files: list[Path], output_path: Path) -> None:
    rows = []
    for _, row in display_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")

    table_header = "".join(f"<th>{html.escape(col)}</th>" for col in display_df.columns)
    image_blocks = "\n".join(
        f'<div class="card"><h3>{html.escape(path.stem.replace("_", " ").title())}</h3><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></div>'
        for path in graph_files
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Simulated vs Real Battery Dispatch</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ color: #4b5563; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 30px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .card {{ border: 1px solid #e5e7eb; padding: 14px; }}
    img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
  <h1>Simulated vs Real Comparison</h1>
  <p>Battery dispatch comparison across expected-value and realised-path formulations.</p>
  <table>
    <thead><tr>{table_header}</tr></thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <div class="grid">
    {image_blocks}
  </div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    df = read_and_clean_data(DATA_PATH)
    models = fit_models(df)

    sim_realised_df, sim_meta = build_simulated_scenario(models, rng)
    real_realised_df, real_meta = build_real_historical_scenario(df)

    sim_expected_df = build_expected_scenario(sim_realised_df, models)
    real_expected_df = build_expected_scenario(real_realised_df, models)

    sim_expected_schedule, sim_expected_summary = solve_dispatch_dp(
        sim_expected_df,
        "Simulated",
        "Expected Value",
        "Simulated Month",
    )
    sim_realised_schedule, sim_realised_summary = solve_dispatch_dp(
        sim_realised_df,
        "Simulated",
        "Realised Path",
        "Simulated Month",
    )
    real_expected_schedule, real_expected_summary = solve_dispatch_dp(
        real_expected_df,
        "Real Historical",
        "Expected Value",
        str(real_meta["reference_date"]),
    )
    real_realised_schedule, real_realised_summary = solve_dispatch_dp(
        real_realised_df,
        "Real Historical",
        "Realised Path",
        str(real_meta["reference_date"]),
    )

    summary_df = pd.DataFrame(
        [
            sim_expected_summary,
            sim_realised_summary,
            real_expected_summary,
            real_realised_summary,
        ]
    )
    display_df = build_display_table(summary_df)

    schedules = [
        (sim_expected_schedule, "Simulated Expected Value", OUTPUT_DIR / "graph_simulated_expected_value.svg"),
        (sim_realised_schedule, "Simulated Realised Path", OUTPUT_DIR / "graph_simulated_realised_path.svg"),
        (real_expected_schedule, "Real Historical Expected Value", OUTPUT_DIR / "graph_real_historical_expected_value.svg"),
        (real_realised_schedule, "Real Historical Realised Path", OUTPUT_DIR / "graph_real_historical_realised_path.svg"),
    ]

    graph_paths: list[Path] = []
    for schedule_df, title, output_path in schedules:
        build_svg_plot(schedule_df, title, output_path)
        graph_paths.append(output_path)

    display_df.to_csv(OUTPUT_DIR / "dispatch_comparison_summary_python.csv", index=False)
    sim_realised_df.to_csv(OUTPUT_DIR / "simulated_realised_inputs_python.csv", index=False)
    sim_expected_df.to_csv(OUTPUT_DIR / "simulated_expected_inputs_python.csv", index=False)
    real_realised_df.to_csv(OUTPUT_DIR / "real_realised_inputs_python.csv", index=False)
    real_expected_df.to_csv(OUTPUT_DIR / "real_expected_inputs_python.csv", index=False)
    sim_expected_schedule.to_csv(OUTPUT_DIR / "schedule_simulated_expected_value.csv", index=False)
    sim_realised_schedule.to_csv(OUTPUT_DIR / "schedule_simulated_realised_path.csv", index=False)
    real_expected_schedule.to_csv(OUTPUT_DIR / "schedule_real_historical_expected_value.csv", index=False)
    real_realised_schedule.to_csv(OUTPUT_DIR / "schedule_real_historical_realised_path.csv", index=False)
    write_summary_html(display_df, graph_paths, OUTPUT_DIR / "dispatch_comparison_python.html")

    print("7.7 Simulated vs Real Comparison")
    print("Battery Dispatch Comparison Across Expected-Value and Realised-Path Formulations")
    print()
    print(display_df.to_string(index=False))
    print()
    print(
        f"Selected simulated month: path {int(sim_meta['selected_price_path'])}/{N_PRICE_PATHS} | "
        f"price spread {sim_meta['price_spread']:.2f} EUR/MWh"
    )
    print(f"Selected real historical month: {real_meta['reference_date']} | price spread {real_meta['price_spread']:.2f} EUR/MWh")
    print()
    print("Saved graph files:")
    for graph_path in graph_paths:
        print(f" - {graph_path}")


if __name__ == "__main__":
    main()
