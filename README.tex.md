this repo trains a separate lasso or ridge regression model for each altcoin to predict the next day’s closing price using only past data. no lookahead bias. each model uses 2023 data to train and tests on 2024+. it picks whichever model has the best r² score after a time-based split.

data comes from polygon.io and yfinance. daily ohlcv (open, high, low, close, volume) pulled and cleaned for each coin, covering up to may 21, 2025. once trained, each model is used to predict the may 22 close using only may 21 features. if a coin didn’t have enough data, it got skipped.

the features used are all lagged to avoid leakage:
	•	log_return: \log\left(\frac{P_{t-1}}{P_{t-2}}\right)
	•	ema_20: exponential moving average over 20 days
	•	macd_hist: histogram difference from MACD indicator
	•	daily_range_pct: \frac{\text{high} - \text{low}}{\text{close}_{t-1}}
	•	volume_change: \frac{\text{vol}{t-1} - \text{vol}{t-2}}{\text{vol}_{t-2}}
	•	price_to_support: relative gap from a rolling support level

on backtest, average prediction error across all coins was ~4.15%. some coins like matic, arb, and rune had tight predictions. others like inj and fet were off by 10–14%. r² scores ranged from 0.62 to 0.98 depending on the coin.

directionally, the model got the move right 60% of the time. but more interestingly, when the model predicted larger moves (over 2%), it was more likely to be right. the takeaway is: louder predictions are usually more correct.

⸻

does this even matter?

the model’s not going to give you the exact price tomorrow. but it doesn’t need to. when it thinks something’s going to move big, it usually nails the direction.

so instead of treating this like just a regression problem, we use it as a signal generator.

⸻

trading strategy

start with $10,000. only enter trades when the predicted move is above 2% (up or down). if up, we long. if down, we short. flat signals get ignored.

capital allocation: equal split per signal. no compounding.

may 22, 2025 example:
	•	model flagged arb, rune, sol, ada, avax (long) and matic (short)
	•	all six were right in direction
	•	average return: +7.2\%
	•	that’s $720 profit in one day on a simple signal-following strat

⸻

simulated backtest

run the same idea across 2024–2025. only trade signals where |\hat{y}_{t+1} - y_t| / y_t > 0.02

mean daily return: \mu = 0.0034
standard deviation: \sigma = 0.0167
sharpe ratio: S = \frac{\mu}{\sigma} \approx 0.20

filtering for higher-confidence calls (moves over 5%):
	•	\mu = 0.011, \sigma = 0.016, S \approx 0.69

so yeah, there’s signal here.

⸻

summary

it’s not magic. but it’s not noise either. this model captures real movement. it won’t tell you exactly where price will be, but it does flag when something’s about to move. and if you trade only when it’s loud, you get cleaner results.

this setup mimics a real-world strategy: no future peeking, wait for strong signals, act on confidence. and the numbers show it: better sharpe on bigger calls, directionally right more than half the time, and decent profit when it swings hard.

most models whisper. this one yells when it matters.
![Screenshot 2025-05-22 at 5 27 54 PM](https://github.com/user-attachments/assets/287df09a-eb74-4e9e-8b3b-e1e39dc655c0)
![Screenshot 2025-05-22 at 5 30 34 PM](https://github.com/user-attachments/assets/1fbc8c92-1c50-4821-8073-5f810105b0ee)
![Screenshot 2025-05-22 at 5 31 05 PM](https://github.com/user-attachments/assets/e9fcca00-f2c1-476f-94e4-685e2be63891)
![Screenshot 2025-05-22 at 5 31 12 PM](https://github.com/user-attachments/assets/4bd65ddc-305c-4db3-9d91-ed885cf8374f)
![Screenshot 2025-05-22 at 5 31 34 PM](https://github.com/user-attachments/assets/065a530d-469f-4d75-90fc-101e5073746a)
![Screenshot 2025-05-22 at 5 32 33 PM](https://github.com/user-attachments/assets/32d9787c-b68b-4126-ac41-c3b558c981e2)
