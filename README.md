# S&P 500 Market Analysis & Prediction

A data science project exploring the predictability of S&P 500 market behavior across three distinct tasks using classical machine learning approaches.

---

## Project Structure 

sp500-prediction-analysis/
├── data/                   # Raw data fetched via yfinance
│   └── sp500_raw.csv
├── notebooks/              # Analysis and modeling
│   └── sp500_prediction.ipynb
├── src/                    # Reusable modules
│   ├── features.py         # Technical indicator engineering
│   ├── prepare.py          # Data preparation and splitting
│   └── evaluate.py         # Evaluation metrics
├── outputs/                # Generated charts
└── README.md




---

## Data

- **Source:** Yahoo Finance (`yfinance`)
- **Index:** S&P 500 (`^GSPC`)
- **Period:** January 2015 – December 2024
- **Frequency:** Daily OHLCV
- **Train/Test Split:** 80% / 20% (no shuffle — time series integrity preserved)
  - Train: October 2015 – February 2023
  - Test: February 2023 – December 2024

---

## Feature Engineering

Technical indicators were derived from raw OHLCV data:

| Feature | Description |
|---|---|
| `MA_20` | 20-day simple moving average of closing price |
| `MA_50` | 50-day simple moving average of closing price |
| `MA_200` | 200-day simple moving average — long-term trend indicator |
| `RSI` | Relative Strength Index (14-day). Values above 70 indicate overbought, below 30 oversold |
| `MACD` | Difference between 12-day and 26-day exponential moving averages |
| `MACD_Signal` | 9-day EMA of MACD — used to identify trend reversals |
| `Volatility_20` | 20-day rolling standard deviation of daily returns |
| `Daily_Return` | Day-over-day percentage change in closing price |
| `Lag_1` | Closing price 1 day ago |
| `Lag_3` | Closing price 3 days ago |
| `Lag_7` | Closing price 7 days ago |

---

## Evaluation Metrics

Each task uses metrics appropriate to its problem type.

### Classification Metrics (Task 1)

| Metric | Description |
|---|---|
| **Accuracy** | Percentage of correctly predicted directions (up/down) |
| **Precision** | Of all predicted "Up" days, how many were actually "Up" |
| **Recall** | Of all actual "Up" days, how many were correctly predicted |
| **F1-Score** | Harmonic mean of precision and recall |

### Regression Metrics (Tasks 2 & 3)

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — average prediction error in original units. Penalizes large errors more heavily |
| **MAE** | Mean Absolute Error — average absolute difference between predicted and actual values |
| **R²** | Coefficient of determination. 1.0 = perfect prediction, 0.0 = no better than mean, negative = worse than mean |

---

## Tasks & Results

Three prediction tasks were defined and evaluated independently. Each task uses its own baseline and appropriate metrics — cross-task comparison is intentionally avoided as the tasks are fundamentally different problems.

---

### Task 1 — Direction Prediction (Classification)

**Question:** Can we predict whether the S&P 500 will close higher or lower tomorrow?

**Baseline strategy:** Always predict "Up" (majority class = 56.37% of test days)

| Model | Accuracy | Precision (Up) | Recall (Up) | F1 (Up) |
|---|---|---|---|---|
| Baseline (Always Up) | 0.5637 | 0.5637 | 1.0000 | 0.7209 |
| XGBoost | 0.4924 | 0.5800 | 0.3400 | 0.4300 |
| Random Forest | 0.4795 | 0.5500 | 0.3900 | 0.4600 |

**Finding:** No model outperformed the naive baseline. This is a practical demonstration of the **Efficient Market Hypothesis (EMH)** — in an efficient market, price movements are largely random and past data alone cannot reliably predict future direction. More complex models performed worse, likely due to overfitting to noise.

---

### Task 2 — Price Prediction (Regression)

**Question:** Can we predict tomorrow's closing price?

**Baseline strategy:** Use today's closing price as tomorrow's prediction (naive carry-forward)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (Carry-forward) | 39.27 | 29.47 | 0.9961 |
| XGBoost | 629.61 | 430.40 | -0.0115 |
| Random Forest | 588.74 | 398.77 | 0.1155 |

**Why did XGBoost and Random Forest fail?**

Tree-based models **cannot extrapolate** beyond the value range seen during training. The S&P 500 reached all-time highs during the 2023–2024 test period — driven largely by the AI boom — at price levels never observed in training data. As a result:

- **XGBoost** produced erratic predictions capped at training-era price levels, oscillating around a ceiling it could not exceed.
- **Random Forest** averaged its tree outputs, producing a nearly flat line around the mean of training prices — unable to follow the upward trend.

This is a fundamental limitation of tree-based models on non-stationary time series with strong upward trends.

---

### Task 3 — Volatility Prediction (Regression)

**Question:** Can we predict tomorrow's market volatility?

**Baseline strategy:** Use today's volatility as tomorrow's prediction

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Baseline (Carry-forward) | 0.9441 | 0.000472 | 0.000285 |
| Random Forest | 0.9390 | 0.000493 | 0.000318 |
| XGBoost | 0.8958 | 0.000645 | 0.000479 |

**Why does volatility behave differently from price?**

Unlike price, volatility is **mean-reverting** — it fluctuates around a long-term average rather than trending indefinitely upward. This property makes it more predictable from historical data, and explains why all three models performed well. The baseline's strong performance (R² = 0.9441) reflects high autocorrelation in volatility: today's volatility is a very good predictor of tomorrow's. Machine learning models add complexity without meaningfully improving on this simple relationship.

---

## Key Takeaways

**1. The Efficient Market Hypothesis holds in practice.**
Technical indicators and historical prices were insufficient to beat a naive classifier on direction prediction. Increasing model complexity made things worse.

**2. Tree-based models cannot extrapolate.**
XGBoost and Random Forest failed on price prediction because the test period contained price levels outside the training range. This is a known and fundamental limitation — not a tuning problem.

**3. Problem type determines predictability.**
Direction (random walk) → very hard. Price level (trending, non-stationary) → hard for tree models. Volatility (mean-reverting) → relatively predictable.

**4. Complexity is not always better.**
In all three tasks, the simplest model (naive baseline) was competitive or best. The right tool depends on the structure of the problem, not the sophistication of the algorithm.

**5. Metric choice matters.**
Comparing models across tasks using different metrics (accuracy vs RMSE vs R²) is misleading. Each task was evaluated independently with its own appropriate metric and baseline.

---

## What Could Improve Results

- **Sentiment analysis:** News sentiment and social media signals were not included but could significantly improve direction prediction
- **Macroeconomic features:** Interest rates, inflation data, VIX index
- **Regime detection:** Separate models for bull/bear market conditions
- **LSTM / Transformer models:** Better suited for sequential dependencies in time series
- **Ensemble approaches:** Combining predictions from multiple models

---

## Tech Stack

- Python 3.11
- `yfinance` — market data
- `pandas`, `numpy` — data processing
- `scikit-learn` — Random Forest, preprocessing, metrics
- `xgboost` — gradient boosting
- `matplotlib`, `seaborn` — visualization

---

## Author

**Murat Samancı**
Data Analyst @ ING Bank · Aspiring Data Scientist
[LinkedIn](https://www.linkedin.com/in/mursamanci/) · [GitHub](https://github.com/mursam)
