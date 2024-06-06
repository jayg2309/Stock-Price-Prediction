# Import necessary libraries for data manipulation, visualization, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# Import libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Import Streamlit for creating the web app
import streamlit as st

# Title of the web app
st.title('Stock Price Prediction of Google using Linear Rregression')

# Inputs for stock ticker and date range
ticker_input = st.text_input('Enter Stock Ticker', 'GOOGL')
start_date_input = st.date_input('Start Date', pd.to_datetime('2010-01-01'))
end_date_input = st.date_input('End Date', pd.to_datetime('2023-01-01'))

#Fetch and display stock data
if st.button('Fetch Data'):
    # Download stock data using yfinance
    stock_data = yf.download(ticker_input, start=start_date_input, end=end_date_input)

    # Display the downloaded stock data
    st.write(f"Showing data for {ticker_input} from {start_date_input} to {end_date_input}")
    st.write(stock_data)

    # Plot the closing price of the stock
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'])
    plt.title(f'{ticker_input} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($) ')
    plt.grid()
    st.pyplot(plt)

    #Data Preprocessing
    # Convert the date to an ordinal number for machine learning
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date_ordinal'] = stock_data['Date'].apply(lambda date: date.toordinal())

    # Features (date in ordinal form) and target (closing price)
    X = stock_data[['Date_ordinal']] # ordinal dates data - features
    y = stock_data['Close'] # closing price data - target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalize the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)


    # Predict the closing price of the stock
    predictions = model.predict(X_test_scaled)

    # Plot the actual vs predicted prices
    plt.figure(figsize=(10, 6)) # width and height of the plot in inches
    plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Price') # actual price
    plt.plot(stock_data['Date'].iloc[-len(predictions):], predictions, label='Predicted Price') # predicted prices
    plt.title(f'{ticker_input} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Evaluate the model
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'Mean Absolute Error: {mae}')










