# Forecasting-Airline-Passengers

Goal is to predict the number of airline passengers from historical data with as much accuracy as possible.

The first step in the process was to make the time series stationary. This means removing trend and seasonality from the time series. To achieve this, I tried a few methods such as first order differencing, log transformation and moving average. Dickey fuller test is used to determine stationarity of time series.

After removing trend from time series and plotting Autocorrelation function(ACF) and Partial Autocorrelation function(PACF), I found that time series also had seasonality in it. For accurate predictions, it becomes important to remove seasonality as well. So I used first order seasonal differencing to remove seasonality and trend from the time series and validated my results with Dickey-Fuller test.

Lastly on plotting ACF and PACF again, I was able to determine parameters of my Seasonal Autoregressive Integrated Moving Average(SARIMA) model and implement it to make predictions. I achieved an accuracy of 95% [MAPE: 5%].


