#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
About the project: 
    
Created on Wed Jun 19 10:05:02 2024

@author: Priyanshu
"""
# Download the libraries
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import math
from numpy.polynomial import polynomial
from statsmodels.tsa.ar_model import AutoReg

""" 
Download the data using Yahoo finance
"""
stocks = {"AMZN", "MSFT","META","GOOG", "NVDA", "^GSPC"}
start = dt.datetime.today() - dt.timedelta(3650)
end = dt.datetime.today()

cl_price = pd.DataFrame()

for ticker in stocks:
    cl_price[ticker] = yf.download(ticker, start, end)["Adj Close"]

cl_price.dropna(axis = 0, inplace=True)

cl_price.plot()
cl_price.drop("^GSPC", axis=1).plot()
""" 
Calculate simple return "R" and log return "r" 
"""
simple_return = cl_price.pct_change()
simple_return.dropna(axis=0, inplace=True)

log_return = pd.DataFrame()
for ticker in stocks:
    log_return[ticker] = np.log(cl_price[ticker]) - np.log(cl_price[ticker]).shift(1)

log_return.dropna(axis=0, inplace=True)
"""
Checking the first four moments for both return series R and r
1. Mean return - Sum(Xi)/T where i E [1,T]
2. Sample Variance return - Sum(Xi - X.mean())^2 / (T-1) where i E [1,T] 
3. Sample Skewness - Measure of Asymmetry - Sum(Xi-X.mean())^3 / ((T-1) * X.std()^3) where i E [1,T]
4. Sample Kurtosis - Measure of Tailedness - Sum(Xi-X.mean)^4 / ((T-1) * X.std()^4) where i E [1,T]
"""

simple_return.mean()
simple_return.var()
simple_return.skew()
simple_return.kurt()

log_return.mean()
log_return.var()
log_return.skew()
log_return.kurt()

""" Now we will check for both R and r for the six stocks whether they follow Normality

Null hypothesis; Ho:= Return series follows normality
Alternate hypothesis; Ha:= Return series deviates significantly from normal distribution

We will conduct Jarque-Bera test for this and calculate the 
1. t-stat and 
2. p-value with a alpha = 5% level of significance

"""

jb_results = {"simple_return": {}, "log_return": {}}

for ticker in stocks:
    
    test = st.jarque_bera(simple_return[ticker],nan_policy="omit")
    jb_results["simple_return"][ticker] = [test.statistic, test.pvalue]
    
    test = st.jarque_bera(log_return[ticker],nan_policy="omit")
    jb_results["log_return"][ticker] = [test.statistic, test.pvalue]

"""
JARQUE-BERA TEST RESULTS:
    
    All the return series have a p-value of 0.0 which is lesser than .05 and hence we can 
    reject the null hypothesis and conclude that none of the series follow normal distribution

"""

"""
TESTING INDIVIDUAL ACF

    Since none of the return series are normally distributed it is safe to assume that all of them
    will be weak form stationary. We will check for stationarity using Auto-correlation functions with
    different lags and see which lag values give us the best stationarity using different information
    criteria values.

"""
for ticker in stocks:
    ax = plot_acf(simple_return[ticker],missing="drop",, lags=np.arange(0,40), title= "simple_return" + ticker)
    plot_acf(log_return[ticker],missing="drop", lags=np.arange(1,40), title= "log_return" + ticker)

"""
PORTMANTEAU TEST - LJUNG BOX TEST

    Ho:= No autocorrelation exists
    Ha:= There is autocorrelation

This is test of autocorrelation. 
"""
lb_test = {"simple_return": {}, "log_return": {}}
for ticker in stocks:
    test = acorr_ljungbox(simple_return[ticker].dropna(), lags=np.arange(1,40))
    lb_test["simple_return"][ticker] = test
    
    test = acorr_ljungbox(log_return[ticker].dropna(), lags=np.arange(1,40))
    lb_test["log_return"][ticker] = test
    
    print("Stock Ticker: {}".format(ticker))
    print("Simple Return lb_test: ".format(lb_test["simple_return"]["GOOG"]))
    print("Log Return lb_test: ".format(lb_test["log_return"]["GOOG"])
    

"""
PORTMANTEAU TEST RESULTS
Based on the results we can:
    GOOG:= Both R and r series reject the null hypothesis from lag 1 onwards and hence there is
        autocorrelation that exists for Google and can be modelled from AR models
        
    META:= In case of META we can observe that in both the R and r series autocorrelations
    are significant from lag 8 onwards. This shows that series is not white noise and can be 
    modelled using AR models.
    
    NVDA:= NVDA also has significant autocorrelation at 1-8 lags

    MSFT:= Microsoft also is similar to GOOG

    AMZN:= AMZN as we can see fails to reject the null hypothesis in case of lags until 24 
    but after that there is significant auto-correlation for both R and r similar results appear.
    But we must see if we can treat this as stationary and additional test is needed.

    ^GSPC:= S&P 500 also has significant auto-correlation at all the lags
"""

"""
AUGMENTED DICKY FULLER TEST
Ho:= The series is not stationary
Ha:= The series is stationary


"""
from statsmodels.tsa.stattools import adfuller

adf_results = {"simple_return": {}, "log_return": {}}

for ticker in stocks:
    adf_results["simple_return"][ticker] = adfuller(simple_return[ticker].dropna(), autolag='AIC')
    adf_results["log_return"][ticker] = adfuller(log_return[ticker].dropna(), autolag='AIC')

"""
ADF TEST RESULTS
All the p-values for all the series are below 0.05% and hence we can reject
the null hypothesis and conclude that the return series are stationary

Stationarity allows us to use AR family of models as the entire mathematics
is based on the assumption of stationarity and independence of white noise
"""

"""
Calculate equation of the model using AR(1), AR(2) and AR(3) 
and dissect and calculate decays and stochastic cycles
"""

def stochasticCycle(a,b):
    phi_1 = 2 * a
    phi_2 = -(a**2 + b**2)

    if phi_1**2 + 4*phi_2 >=0:
        return "Stochastic Cycle Does not exist"
    
    return str(round(2*np.pi/math.acos((phi_1/2)/np.sqrt(-phi_2)),2)) + " quarters"


res = AutoReg(simple_return["NVDA"].dropna(), lags=[1,2]).fit()
print(res.summary())

gnp = pd.read_csv("https://faculty.chicagobooth.edu/-/media/faculty/ruey-s-tsay/teaching/fts3/dgnp82.txt", header=None, names=["GNP"])
gnp

"""
Check whether stochastic cycle is there
"""

a, b = -3.7536, -4.2285

stochasticCycle(a, b)

"""
HOW TO FIND WHICH LAG TO USE

AIC
BIC
"""
min_l = min_aic = 0
for lag in range(15):
    res = AutoReg(simple_return["NVDA"].dropna(), lags=lag).fit()
    if min_aic>res.bic:
        min_aic = res.bic
        min_l = lag
print(min_l)
    
