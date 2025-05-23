this repo trains a separate lasso or ridge regression model for each altcoin to predict the next day’s closing price using only past data. no lookahead bias. each model uses 2023 data to train and tests on 2024+. it picks whichever model has the best r² score after a time-based split.

the features used are all calculated from t-1:
log_return
ema_20
macd_hist
daily_range_pct
volume_change
price_to_support

data comes from polygon.io and yfinance. daily ohlcv pulled and cleaned for each coin, covering up to may 21, 2025. once trained, each model is used to predict the may 22 close using only may 21 features. if a coin didn’t have enough data, it got skipped.

on backtest, average prediction error across all coins was around 4.15%. some coins like matic, arb, and rune had tight predictions. others like inj and fet were off by 10–14%. model performance ranged from ~0.62 r² up to 0.98 depending on the coin.

directionally, the model got the move right 60% of the time. more importantly, when it predicted larger moves (over 2%), it was more likely to be correct. small changes were noisy. but when the model said something was going to move hard, it usually did — in the right direction.

bottom line: it’s not magic, but it’s not random either. there’s signal here. especially when the model is confident. that’s where this gets interesting.
![Screenshot 2025-05-22 at 5 27 54 PM](https://github.com/user-attachments/assets/287df09a-eb74-4e9e-8b3b-e1e39dc655c0)
![Screenshot 2025-05-22 at 5 30 34 PM](https://github.com/user-attachments/assets/1fbc8c92-1c50-4821-8073-5f810105b0ee)
![Screenshot 2025-05-22 at 5 31 05 PM](https://github.com/user-attachments/assets/e9fcca00-f2c1-476f-94e4-685e2be63891)
![Screenshot 2025-05-22 at 5 31 12 PM](https://github.com/user-attachments/assets/4bd65ddc-305c-4db3-9d91-ed885cf8374f)
![Screenshot 2025-05-22 at 5 31 34 PM](https://github.com/user-attachments/assets/065a530d-469f-4d75-90fc-101e5073746a)
![Screenshot 2025-05-22 at 5 32 33 PM](https://github.com/user-attachments/assets/32d9787c-b68b-4126-ac41-c3b558c981e2)
