crypto price prediction model that tries to forecast the next day’s closing price of altcoins using only information available up to today. 

For each coin:
	•	train on 2023 data
	•	test on 2024+ data
	•	currently using Lasso and Ridge regression to predict the closing price
	•	then let the model pick the best alpha (regularization strength)

Pick best model based on R² performance.

this repo trains separate lasso or ridge regression models for each altcoin to predict the next day’s closing price using only past data. no lookahead bias. for each coin it selects whichever model has the better r² after a time-based split. the model uses features like log_return, ema_20, macd_hist, daily_range_pct, volume_change, and price_to_support — all calculated from t-1 to predict close at t.

the split is done chronologically: train on 2023, test on 2024+. once trained, each model is used to predict may 22, 2025 using may 21 data. if there wasn’t enough data for a coin, it was skipped.

performance is solid. average absolute prediction error across coins is around 4.15%. most coins land between 8-12% mae relative to price, with matic hitting 0.98 r². lowest performing coin was sol with ~0.62 r².

directionally, the model got the move right 60% of the time. more interestingly, predictions with more confidence (bigger % moves) had higher directional accuracy. small predicted moves were noisy. but when the model called a strong move, it usually got the direction right.

there’s signal here. the model isn’t perfect, but it’s not guessing. especially when it thinks something is going to move hard. those are the ones to pay attention to.
