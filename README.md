crypto price prediction model that tries to forecast the next day’s closing price of altcoins using only information available up to today. 

For each coin:
	•	train on 2023 data
	•	test on 2024+ data
	•	currently using Lasso and Ridge regression to predict the closing price
	•	then let the model pick the best alpha (regularization strength)

Pick best model based on R² performance.
