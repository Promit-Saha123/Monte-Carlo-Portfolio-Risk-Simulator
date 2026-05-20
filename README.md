# Monte Carlo Portfolio Risk Simulator

A Python-based portfolio risk simulator that uses Monte Carlo methods to model potential future portfolio outcomes and evaluate downside risk.

This project allows users to input portfolio weights, pull historical market data, simulate thousands of possible portfolio paths, and analyze key risk metrics such as volatility, drawdown, Value at Risk, and Conditional Value at Risk.

---

## Overview

The Monte Carlo Portfolio Risk Simulator is designed to help visualize how a portfolio may perform under uncertain market conditions. By using historical asset returns and randomized simulations, the project estimates a range of possible portfolio outcomes rather than relying on a single expected return.

The goal of this project was to apply data science, financial modeling, and software development skills to build an interactive risk analysis tool.

---

## Features

- Pulls historical stock price data using Yahoo Finance
- Calculates daily returns and portfolio performance
- Runs Monte Carlo simulations across thousands of possible outcomes
- Supports both bootstrap-based and normal distribution-based Monte Carlo simulation
- Computes key portfolio risk metrics:
  - Volatility
  - Maximum drawdown
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
- Visualizes simulated portfolio paths (spaghetti plots, fan charts)
- Provides an interactive Streamlit interface for user inputs

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- yFinance
- Matplotlib

---

## How It Works

1. The user selects assets and portfolio weights.
2. Historical price data is pulled and converted into daily log returns.
3. The simulator estimates return behavior using historical data.
4. Monte Carlo simulation generates thousands of possible future portfolio outcomes.
5. Risk metrics are calculated from the simulated profit/loss distribution.
6. Results are displayed through summary metrics, charts, and path-based risk visualizations.

---

## Risk Metrics

**Volatility**: Measures how much portfolio returns fluctuate over time.

**Maximum Drawdown**: Measures the largest peak-to-trough decline in portfolio value.

**Value at Risk (VaR)**: Estimates the potential loss at a given confidence level.

**Conditional Value at Risk (CVaR)**: Measures the expected loss in the worst-case scenarios beyond the VaR threshold.

---

## What I Learned

Through this project, I strengthened my understanding of:

- Monte Carlo simulation
- Portfolio risk modeling
- Financial data analysis
- Data visualization
- Interactive app development with Streamlit
- Structuring and documenting a technical project
- Unit Testing

---

## Future Improvements

- Add support for more asset classes
- Include portfolio optimization
- Add benchmark comparison against major indexes
- Improve visualizations and dashboard layout
- Deploy the app publicly using Streamlit Cloud

---

## Running the Project

**1. Clone the repository**

```bash
git clone https://github.com/Promit-Saha123/Monte-Carlo-Portfolio-Risk-Simulator.git
cd Monte-Carlo-Portfolio-Risk-Simulator
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

---

## Video Demo

*Coming soon*

---
