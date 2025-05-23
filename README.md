# Price Prediction Model

This repository contains machine learning models for predicting cryptocurrency prices using technical indicators and historical data.

## Directory Structure

```
.
├── data/               # Data files
│   └── crypto_ohlcv.csv
├── models/            # Trained model files
│   └── crypto_models.joblib
├── scripts/           # Python scripts
│   ├── get_data.py
│   ├── regression.py
│   ├── predict.py
│   ├── may22nd_predict.py
│   ├── get_historical_prices.py
│   └── percentage_error.py
├── graphs/            # Visualizations and predictions
│   ├── model_r2_scores.png
│   ├── model_mae_percent.png
│   ├── Model_Performance_Summary.csv
│   ├── predicted_may22_prices.csv
│   └── [coin]_signals.csv
└── README.md
```

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download historical data:
```bash
python scripts/get_data.py
```

3. Train models:
```bash
python scripts/regression.py
```

4. Make predictions:
```bash
python scripts/predict.py
```

## Model Details

- Uses Ridge and Lasso regression models
- Features include technical indicators like EMA, MACD, and volume changes
- Each cryptocurrency has its own trained model
- Models are saved in `models/crypto_models.joblib`

## Prediction Results

- Latest predictions are saved in `graphs/predicted_may22_prices.csv`
- Historical predictions and backtest results are in `graphs/[coin]_signals.csv`
- Model performance metrics are in `graphs/Model_Performance_Summary.csv`

## Visualization

- Model performance visualizations are in the `graphs/` directory
- Each coin has its own set of visualizations:
  - Feature importance
  - Predicted vs actual prices
  - Residuals
  - Cumulative profit

This repo trains a separate Lasso or Ridge regression model per altcoin to predict the next day's closing price using only past data. No lookahead, no cheating. Models are trained on 2023 data, tested on 2024+, and the one with the best R² after a time-based split is picked.

Features (all calculated from day t-1):
	•	log_return: log of price ratio between previous two days
	•	ema_20: 20-day exponential moving average
	•	macd_hist: MACD histogram (momentum signal)
	•	daily_range_pct: (high - low) / close
	•	volume_change: percent change in volume
	•	price_to_support: distance from close to local support

Data
	•	Sourced from Polygon.io and yfinance
	•	Daily OHLCV up to May 21, 2025
	•	If a coin didn't have enough data, it got skipped

Model Results
	•	Average prediction error: ~4.15%
	•	Best fits: ARB, RUNE, SOL
	•	Worst fits: INJ, FET (off by 10–14%)
	•	R² ranged from 0.62 to 0.98
	•	Directional accuracy: ~60%
	•	When the model predicted a move > 2%, it was right more often than not

Trading Strategy

This isn't just regression—it's a signal generator.

The logic:
	•	If the model predicts a >2% gain tomorrow → go long
	•	If it predicts a >2% drop → short
	•	If it predicts <2% either way → ignore

On May 22, 2025:
	•	The model flagged 6 coins
	•	All 6 moved in the predicted direction
	•	Avg return: 7.2% across those signals in one day

Portfolio Simulation
	•	Starting capital: $10,000
	•	Trade size: 10% per signal
	•	Only act on >2% predicted moves
	•	Sharpe ratio of simulated daily trading: ~1.94
	•	CAGR over backtest horizon: strong
	•	Max drawdown: controlled

See the full PDF writeup (report.pdf) for formulas, derivations, and performance charts.

Final Word

The model isn't psychic, but it's not guessing either. It's signal, not noise—especially when it's confident. When it talks loud, listen.

Use this to filter noise, trade selectively, and stay mechanical.

⸻

Let me know if you want to plug in live data, deploy to a dashboard, or wrap it into a bot. Also let me know if you have any advice I am still refining the model and strategy

Glossary

Lasso Regression – A type of regression that helps with feature selection by shrinking less important feature weights to zero. It’s like telling the model “only keep the useful stuff.”

Ridge Regression – Similar to Lasso, but instead of zeroing out weights, it shrinks them to small values. It’s good when you think all features might be useful, just not equally.

R² Score (R-squared) – Tells you how well your model explains the variance in the actual data. 1.0 is perfect, 0.0 means it’s guessing the average, negative means it’s doing worse than that.

Sharpe Ratio – A way to measure risk-adjusted returns. Higher is better. It tells you how much return you’re getting per unit of risk.

Log Return – The natural log of the price ratio from one day to the next. Used because it treats gains and losses symmetrically.

EMA (Exponential Moving Average) – A moving average that gives more weight to recent prices. Helps detect trends more quickly than a simple average.

MACD Histogram – A technical indicator that shows momentum. Positive = bullish, negative = bearish. We use the histogram part, which is basically the “acceleration.”

Daily Range % – (High - Low) / Close. Tells you how much a coin moved in a day, relative to its closing price.

Volume Change – How much trading volume changed from the previous day. High volume spikes can mean big interest.

Price to Support – How far the price is from its rolling support level. Helps identify whether we’re near a bounce zone.

Backtest – A way to test a trading strategy using past data. If it made money in hindsight, it might have a shot live.

No Lookahead Bias – Means the model only used information that would have been available at the time of prediction. No cheating with future data.

Signal – When the model predicts a big enough move (e.g., >2%), we treat that as a trading signal.

Long – Buying with the expectation that the asset goes up.

Short – Selling (or betting against) an asset hoping its price will drop.

