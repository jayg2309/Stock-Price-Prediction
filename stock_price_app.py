import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'
stock_data = download_stock_data(ticker, start_date, end_date)


plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'])
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.show()


stock_data['Date'] = stock_data.index
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['Date_ordinal'] = stock_data['Date'].apply(lambda date: date.toordinal())


X = stock_data[['Date_ordinal']]
y = stock_data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


predictions = model.predict(X_test_scaled)

