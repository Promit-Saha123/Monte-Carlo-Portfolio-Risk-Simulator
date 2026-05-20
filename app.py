# app.py
import datetime as dt

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from utils import (
    parse_tickers,
    normalize_weights,
    weights_from_holdings,
    clean_prices,
    compute_log_returns,
    portfolio_returns,
    portfolio_return_path,
    max_drawdown,
    time_to_failure,
)
from sim import mu_cov, sim_log_returns, sim_log_returns_bootstrap
from stats import var_cvar

#  Page setup
st.set_page_config(page_title="Monte Carlo Portfolio Risk Simulator", layout="wide")

# Smaller header (less vertical space than st.title)
st.markdown(
    """
    <h2 style="margin-bottom: 0.2rem;"> Monte Carlo Portfolio Risk Simulator</h2>
    <p style="margin-top: 0; font-size: 0.9rem; color: #9CA3AF;">
        Historical + Monte Carlo portfolio risk. Path-based mode unlocks drawdown + failure metrics.
    </p>
    """,
    unsafe_allow_html=True,
)

#  Cached download 
@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date_str, end_date_str, interval):
    data = yf.download(
        tickers,
        start=start_date_str,
        end=end_date_str,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    else:
        prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return clean_prices(prices)

# UI helpers
def metric_card(label: str, value: str):
    """Neutral metric card (no semantic coloring)."""
    st.markdown(
        f"""
        <div style="
            padding: 0.75rem;
            border-radius: 12px;
            background-color: #111827;
            border: 1px solid rgba(255,255,255,0.08);
            text-align: center;
        ">
            <div style="font-size: 0.80rem; color: #9CA3AF; margin-bottom: 0.25rem;">
                {label}
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; color: inherit;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar 
st.sidebar.header("Controls")

tickers_raw = st.sidebar.text_input("Tickers (comma separated)", value="SPY,AAPL,MSFT")
tickers_list = parse_tickers(tickers_raw)

st.sidebar.subheader("Date Range")
date_range = st.sidebar.date_input(
    "Pick a date range",
    value=(dt.date(2015, 1, 1), dt.date.today()),
    min_value=dt.date(1990, 1, 1),
    max_value=dt.date.today(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, dt.date.today()

st.sidebar.caption(f"Selected: {start_date} → {end_date}")

interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
portfolio_value = st.sidebar.number_input("Portfolio value ($)", value=100_000, step=1_000)

alpha = st.sidebar.slider("Alpha level:", 0.90, 0.99, 0.95)

mode = st.sidebar.selectbox("Input mode", ["Weights", "Dollar Holdings"], index=0)

use_mc = st.sidebar.checkbox("Use Monte Carlo simulation", value=True)

# Defaults
path_based = False
n_sims = 10_000
years = 1.0
horizon_steps = 252
seed = 0
mc_method = "Bootstrap (historical)"

if use_mc:
    n_sims = st.sidebar.slider("Simulations", 1_000, 100_000, 10_000, step=1_000)

    # Horizon in YEARS 
    years = st.sidebar.number_input(
        "Horizon (years)",
        min_value=0.05,
        max_value=50.0,
        value=1.0,
        step=0.25
    )

    # Convert years to steps based on interval
    if interval == "1d":
        horizon_steps = int(round(years * 252))
        st.sidebar.caption(f"Horizon steps: {horizon_steps} (~{years:.2f} trading years)")
    elif interval == "1wk":
        horizon_steps = int(round(years * 52))
        st.sidebar.caption(f"Horizon steps: {horizon_steps} (~{years:.2f} years)")
    elif interval == "1mo":
        horizon_steps = int(round(years * 12))
        st.sidebar.caption(f"Horizon steps: {horizon_steps} (~{years:.2f} years)")
    else:
        horizon_steps = int(round(years * 252))
        st.sidebar.caption(f"Horizon steps: {horizon_steps} (~{years:.2f} years)")

    if horizon_steps > 10_000:
        st.sidebar.error("Horizon too large — reduce years or use a shorter interval.")
        horizon_steps = 10_000

    path_based = st.sidebar.checkbox("Path-based simulation (enables paths/drawdowns)", value=True)
    seed = st.sidebar.number_input("Random seed (0 = none)", value=0)

    mc_method = st.sidebar.selectbox(
        "MC method",
        ["Bootstrap (historical)", "Normal (mu/cov)"],
        index=0,  # Bootstrap default
    )
    st.sidebar.caption("Bootstrap uses a fixed internal block size (not user-configurable).")

# Allocation editor: tickers as index, single column (weight OR holding) 
st.sidebar.subheader("Allocation")

if len(tickers_list) == 0:
    st.sidebar.info("Enter tickers to configure allocation.")
    can_run = False
    edited_alloc = None
else:
    if mode == "Weights":
        col_name = "Weight"
        default_val = 1.0 / len(tickers_list)
    else:
        col_name = "Holding ($)"
        default_val = 10_000.0

    alloc_df = pd.DataFrame({col_name: [default_val] * len(tickers_list)}, index=tickers_list)

    edited_alloc = st.sidebar.data_editor(
        alloc_df,
        use_container_width=True,
        num_rows="fixed",
        key="alloc_editor",
    )

    # Validate
    vals = pd.to_numeric(edited_alloc[col_name], errors="coerce").values
    if len(vals) != len(tickers_list) or np.any(pd.isna(vals)):
        st.sidebar.error("Fill in all allocation values.")
        can_run = False
    elif np.any(vals < 0):
        st.sidebar.error("Allocation values must be non-negative.")
        can_run = False
    elif float(np.sum(vals)) == 0:
        st.sidebar.error("Allocation values can’t all be zero.")
        can_run = False
    else:
        can_run = True

    # Helpful summary
    if mode == "Weights" and can_run:
        st.sidebar.caption(f"Weight sum: {float(np.sum(vals)):.4f} (auto-normalized on run)")
    if mode == "Dollar Holdings" and can_run:
        st.sidebar.caption(f"Total holdings: ${float(np.sum(vals)):,.0f} (portfolio value updates on run)")

run = st.sidebar.button("Run", disabled=not can_run)

# Main run
if run:
    try:
        # Extract weights
        vals = pd.to_numeric(edited_alloc[col_name], errors="coerce").values.astype(float)

        if mode == "Weights":
            weights = normalize_weights(vals)
        else:
            holdings = vals
            weights, portfolio_value = weights_from_holdings(holdings)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        with st.spinner("Downloading prices + computing returns..."):
            try:
                prices = download_prices(tickers_list, start_date_str, end_date_str, interval)
                if prices.empty or prices.shape[0] == 0:
                    st.error(f"⚠️ No data available for {', '.join(tickers_list)}. This can happen if:\n- Ticker is delisted or invalid\n- Date range has no trading data\n- yfinance has temporary issues\n\nTry different tickers or a wider date range.")
                    st.stop()
                returns = compute_log_returns(prices)
            except Exception as e:
                st.error(f"⚠️ Failed to download price data: {str(e)}\n\nCommon causes:\n- Invalid ticker symbols\n- Network issues\n- Ticker delisted or restricted by yfinance\n\nTry other tickers like: AAPL, SPY, TSLA, JNJ, MSFT")
                st.stop()

        if returns.shape[0] < 5:
            st.error("Not enough return data in that date range. Try a wider range.")
            st.stop()

        V_path = None
        V_T = None

        # ---------------- Scenarios ----------------
        if not use_mc:
            # Historical: distribution of 1-step portfolio log returns (not compounded)
            pr_log = returns.values @ weights
            net = portfolio_value * pr_log
            losses = -net
        else:
            if mc_method.startswith("Bootstrap"):
                sim_r = sim_log_returns_bootstrap(
                    returns,
                    sim=int(n_sims),
                    horizon_days=int(horizon_steps),  # horizon is steps
                    seed=None if seed == 0 else int(seed),
                    path_based=path_based,
                )
            else:
                mu, cov = mu_cov(returns)
                sim_r = sim_log_returns(
                    mu,
                    cov,
                    sim=int(n_sims),
                    horizon_days=int(horizon_steps),
                    seed=None if seed == 0 else int(seed),
                    path_based=path_based,
                )

            if not path_based:
                port_lr = portfolio_returns(sim_r, weights)  # (sim,)
                V_T = portfolio_value * np.exp(port_lr)
                net = V_T - portfolio_value
                losses = -net
            else:
                port_lr_path = portfolio_return_path(sim_r, weights)  # (sim, steps)
                V_path = portfolio_value * np.exp(np.cumsum(port_lr_path, axis=1))
                V_T = V_path[:, -1]
                net = V_T - portfolio_value
                losses = -net

        # Metrics 
        VaR, CVaR = var_cvar(losses, alpha)
        expected_net = float(np.mean(net))
        median_net = float(np.median(net))
        prob_profit = float(np.mean(net > 0))

        st.markdown("### Summary Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Mean P/L", f"${expected_net:,.0f}")
        with c2:
            metric_card("Median P/L", f"${median_net:,.0f}")
        with c3:
            metric_card("P(profit)", f"{prob_profit:.2%}")
        with c4:
            metric_card(f"VaR ({int(alpha*100)}%)", f"${VaR:,.0f}")
        with c5:
            metric_card(f"CVaR ({int(alpha*100)}%)", f"${CVaR:,.0f}")

        st.divider()

        # ---------------- Graph gallery ----------------
        st.subheader("Graphs")
        st.caption("Scroll down and flip tabs to view every chart. All charts are labeled.")

        tab1, tab2, tab3, tab4 = st.tabs([
            "1) Net Return (%)",
            "2) Spaghetti Paths",
            "3) Fan Chart",
            "4) Drawdowns + Failure",
        ])

        # 1) Net Return Distribution (%)
        with tab1:
            st.markdown("### Net Return Distribution (%)")
            st.caption("Histogram of net return over the full horizon, expressed as a percentage of initial value.")

            net_return_pct = 100.0 * net / float(portfolio_value)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(net_return_pct, bins=60)
            ax.axvline(0, linestyle="--", linewidth=2, color="#9CA3AF", label="Break-even (0%)")

            ax.set_title("Portfolio Net Return Distribution")
            ax.set_xlabel("Net Return (%)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)

        # 2) Spaghetti plot (only path-based MC)
        with tab2:
            st.markdown("### Monte Carlo Portfolio Value Paths (Spaghetti Plot)")
            st.caption("Only available in Monte Carlo + Path-based mode.")

            if not (use_mc and path_based and V_path is not None):
                st.info("Enable **Monte Carlo** + **Path-based simulation** to see this plot.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                max_lines = min(100, V_path.shape[0])
                for i in range(max_lines):
                    ax.plot(V_path[i], alpha=0.15)
                ax.axhline(portfolio_value, linestyle="--", linewidth=1, label="Initial Value")
                ax.set_title("Monte Carlo Portfolio Value Paths")
                ax.set_xlabel("Time (steps)")
                ax.set_ylabel("Portfolio Value ($)")
                ax.legend()
                st.pyplot(fig)

        # 3) Fan chart (only path-based MC)
        with tab3:
            st.markdown("### Monte Carlo Fan Chart (5–95% band + median)")
            st.caption("Only available in Monte Carlo + Path-based mode.")

            if not (use_mc and path_based and V_path is not None):
                st.info("Enable **Monte Carlo** + **Path-based simulation** to see this plot.")
            else:
                p5 = np.percentile(V_path, 5, axis=0)
                p50 = np.percentile(V_path, 50, axis=0)
                p95 = np.percentile(V_path, 95, axis=0)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.fill_between(range(len(p50)), p5, p95, alpha=0.3, label="5–95% band")
                ax.plot(p50, linewidth=2, label="Median path")
                ax.axhline(portfolio_value, linestyle="--", linewidth=1, label="Initial Value")
                ax.set_title("Monte Carlo Fan Chart (Portfolio Value)")
                ax.set_xlabel("Time (steps)")
                ax.set_ylabel("Portfolio Value ($)")
                ax.legend()
                st.pyplot(fig)

        # 4) Drawdowns + failure (only path-based MC)
        with tab4:
            st.markdown("### Max Drawdown + Time-to-Failure")
            st.caption("Only meaningful with path-based simulation (needs full paths).")

            if not (use_mc and path_based and V_path is not None):
                st.info("Enable **Monte Carlo** + **Path-based simulation** to see these plots.")
            else:
                st.markdown("#### Max Drawdown Distribution")
                mdd = np.array([max_drawdown(V_path[i]) for i in range(V_path.shape[0])], dtype=float)

                colA, colB, colC = st.columns(3)
                colA.metric("Median Max Drawdown", f"{np.median(mdd):.2%}")
                colB.metric("5th pct (worse)", f"{np.quantile(mdd, 0.05):.2%}")
                colC.metric("95th pct (better)", f"{np.quantile(mdd, 0.95):.2%}")

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(mdd, bins=60)
                ax.set_title("Max Drawdown Distribution (Monte Carlo)")
                ax.set_xlabel("Max Drawdown")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                st.markdown("#### Time to Failure (below threshold of initial)")
                threshold = st.slider("Failure threshold (fraction of initial)", 0.30, 0.95, 0.70, 0.05)

                ttf = np.array([time_to_failure(V_path[i], threshold=float(threshold)) for i in range(V_path.shape[0])], dtype=object)
                fail_rate = float(np.mean(pd.notna(ttf)))
                st.metric(f"P(fall below {int(threshold*100)}% of initial)", f"{fail_rate:.2%}")

                ttf_clean = pd.Series(ttf).dropna().astype(int).values
                if ttf_clean.size > 0:
                    st.metric("Median TTF (steps | conditional on failure)", f"{int(np.median(ttf_clean))}")
                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.hist(ttf_clean, bins=50)
                    ax.set_title("Time-to-Failure Distribution (Conditional on Failure)")
                    ax.set_xlabel("Steps until failure")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                else:
                    st.info("No failures occurred at this threshold in the current simulations.")

        with st.expander("Run details", expanded=False):
            st.write("**Tickers:**", tickers_list)
            st.write("**Date range:**", f"{start_date_str} → {end_date_str}")
            st.write("**Interval:**", interval)
            st.write("**Portfolio value:**", f"${portfolio_value:,.0f}")
            st.write("**Mode:**", "Monte Carlo" if use_mc else "Historical")
            st.write("**Weights:**", np.round(weights, 6))

            if use_mc:
                st.write("**MC method:**", mc_method)
                st.write("**Simulations:**", int(n_sims))
                st.write("**Horizon (years):**", float(years))
                st.write("**Horizon (steps):**", int(horizon_steps))
                st.write("**Path-based:**", path_based)
                st.write("**Seed:**", "None" if seed == 0 else int(seed))

    except Exception as e:
        st.error(f"Error: {e}")