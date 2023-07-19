#!/usr/bin/env python
# coding: utf-8

# # Problem 1

# In[4]:


import sys
#!{sys.executable} -m pip install yfinance # you only need to pip install this once.  Once installed, you can comment out this command


# In[5]:


import yfinance as yf
from datetime import date, datetime
from dateutil import parser
from math import log, sqrt, pi, exp
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# Get all option expiration dates for a particular underlying symbol
Apl = yf.Ticker("AAPL")
Goo = yf.Ticker("GOOG")
Apl_date = Apl.options
Goo_date = Goo.options


# In[7]:


# Get the details of the option chain for calls for a certain expiration date

date_objectA = parser.parse(Apl_date[1])
date_objectG = parser.parse(Goo_date[1])
print(date_objectA.date(), date_objectG.date()) 
optA = Apl.option_chain(str(date_objectA.date()))
optG = Goo.option_chain(str(date_objectG.date()))

call_optA = optA.calls
call_optG = optG.calls
VolA = call_optA[(call_optA['strike'] >= 100) & (call_optA['strike'] <= 250)]
VolG = call_optG[(call_optG['strike'] >= 50) & (call_optG['strike'] <= 250)]
#compare the impliedvol from this table below to your calculated price


# In[8]:


# Get the most recent closing price of the underlying symbol
today = date.today()

pricesG = yf.download("GOOG", 
                   start='2017-01-01', 
                   end=today, 
                   progress=False, auto_adjust=True)
pricesA = yf.download("AAPL", 
                   start='2017-01-01', 
                   end=today, 
                   progress=False, auto_adjust=True)
#display(pricesA)
Sg = pricesG['Close'][-1]
Sa = pricesA['Close'][-1]
SYg =((call_optG['bid'] + call_optG['ask'])/2)
display(Sg,Sa)


# In[9]:


# Set of functions to calculate the implied volatility of a call option
#Price is what Yf tells us



def d1(S,K,T,r,sigma):
    return(log(S/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))

def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

def call_implied_volatility(Price, S, K, T, r):
    sigma = 0.0001
    while sigma < 1:
        Price_implied = S * \
            norm.cdf(d1(S, K, T, r, sigma))-K*exp(-r*T) * \
            norm.cdf(d2(S, K, T, r, sigma))
        if Price-(Price_implied) < 0.0001:
            return sigma
        sigma += 0.0001
    return "Not Found"


# In[10]:


"1a: Risk Free Rate assumed to be very close to 0, arbitrarily made 0.00001"


# In[133]:


# Loop through each strike price and calculate the implied vol
#Strikes chosen because 50-250 always resulted in an index error I couldn't figure out :(
C_IV_list = []
C_K_list = []

df_calls = call_optA[(call_optA['strike'] >= 100) & (call_optA['strike'] <= 250)] #for apple
for i in range(0, len(df_calls)):
    K = df_calls.iloc[i, 2]
    Price = (df_calls.iloc[i, 4] + df_calls.iloc[i, 5]) / 2
    CIV = call_implied_volatility(Price, Sa, K, ((date_objectA.date() - today).days/30), 0.00001)
    days_to_exp = (date_objectA.date() - today).days
    if CIV == 'Not Found':
        CIV = C_IV_list[-1]
        continue
    C_IV_list.append(CIV)
    C_K_list.append(K)
    
#if you want to convert implied vol to annual CIV*np.sqrt(12)    


# In[15]:


#Doing the same for Google
C_IV_listG = []
C_K_listG = []

df_calls_G = call_optG[(call_optG['strike'] >= 50) & (call_optG['strike'] <= 250)] #for google
for i in range(0, len(df_calls_G)):
    Kg = df_calls_G.iloc[i, 2]
    Priceg = (df_calls_G.iloc[i, 4] + df_calls_G.iloc[i, 5]) / 2
    CIVg = call_implied_volatility(Priceg, Sg, Kg, ((date_objectG.date() - today).days/30), 0.00001)
    days_to_expg = (date_objectG.date() - today).days
    if CIVg == 'Not Found':
        CIVg = C_IV_listG[-1]
        continue
    C_IV_listG.append(CIVg)
    C_K_listG.append(Kg)


# In[58]:


Va = []
for i in range(0,len(VolA)):
    x = VolA.iloc[i,10]
    Va.append(x)
plt.plot(C_K_list, C_IV_list, 'o')
plt.plot(C_K_list, Va, 'go')
plt.plot()
plt.title("Apple Implied Vol vs. Strike Price - Expiration " + str(date_objectA.date()))
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.show()


# In[16]:


Ga = []
for i in range(0,len(VolG)):
    y = VolG.iloc[i,10]
    Ga.append(y)
plt.plot(C_K_listG, C_IV_listG, 'o')
plt.plot(C_K_listG, Ga, 'go')
plt.plot()
plt.title("Google Implied Vol vs. Strike Price - Expiration " + str(date_objectA.date()))
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.show()


# In[ ]:


#ANSWER TO PART B: I chose an expiration of one month since most calls only last a max of a few months/half a year.

#ANSWER TO PART C: The implied volatility in both stocks is significantly lower than the impliedVol provided by yahoof.
#This difference may be due to the monthly time period in use whereas yahoof uses annual.

#ANSWER TO PART D: Compared to the recitation answer, this volatility also seems to be slighlty higher.

#All code adapted from uploaded code


# # Problem 2

# In[134]:


Strikes = call_optA[(call_optA['strike'] >= 100) & (call_optA['strike'] <= 250)]
#for YF price
Mprice = []
for i in range(0,len(Strikes)):
    x = (Strikes.iloc[i,4] + Strikes.iloc[i,5])/2
    #y = Strikes.iloc[i,3]     testing if lastPrice produced a model that showed convergence of BS
    Mprice.append(x)
#for binomial 1 period
V_binom = []
S0 = Sa # initial stock price
dt = ((date_objectA.date() - today).days)/30 # expiry
N = 10000 # number of samples
n = 1     # number of steps
u = []
d = []
q = []
r = 0.00001 # interest rate

sigma = C_IV_list # volatility
for i in range(0,len(C_IV_list)):
    u.append(np.exp(sigma[i]*np.sqrt(dt))) # up return
    down = np.exp(sigma[i]*np.sqrt(dt))
    d.append(1/down) # down return
for i in range(0,len(u)):
    q.append((np.exp(r*dt) - d[i])/(u[i]-d[i]))
S =[]
for i in range(0,len(Strikes)):
    U = np.random.binomial(n, q[i], N) # result of flipping a coin n times, with  N samples of size n
    S.append(S0*u[i]**U*d[i]**(n-U))
    V_binom.append(np.exp(-r*dt*n)*np.sum(np.maximum(S[i]-C_K_list[i],0))/N)
plt.plot(C_K_list,Mprice, 'r')
plt.plot(C_K_list,V_binom, 'b')
plt.xlabel("Strike Prices")
plt.ylabel("Option Prices")
plt.title("Option Prices Vs Strikes for 1 Period Binomial")
plt.legend(['Mprice','V_Binom'])
plt.show()
#PLOT BINOM - YF


# # 5 step Binom

# In[136]:


V_binom5 = []
S0 = Sa # initial stock price
dt = ((date_objectA.date() - today).days)/30 # expiry
N = 10000 # number of samples
n = 5     # number of steps
u = []
d = []
q = []
r = 0.00001 # interest rate

sigma = C_IV_list # volatility
for i in range(0,len(C_IV_list)):
    u.append(np.exp(sigma[i]*np.sqrt(dt))) # up return
    down = np.exp(sigma[i]*np.sqrt(dt))
    d.append(1/down) # down return
for i in range(0,len(u)):
    q.append((np.exp(r*dt) - d[i])/(u[i]-d[i]))
S =[]
for i in range(0,len(Strikes)):
    U = np.random.binomial(n, q[i], N) # result of flipping a coin n times, with  N samples of size n
    S.append(S0*u[i]**U*d[i]**(n-U))
    V_binom5.append(np.exp(-r*dt*n)*np.sum(np.maximum(S[i]-C_K_list[i],0))/N)
plt.plot(C_K_list,Mprice, 'r')
plt.plot(C_K_list,V_binom5, 'b')
plt.xlabel("Strike Prices")
plt.ylabel("Option Prices")
plt.title("Option Prices Vs Strikes for 1 Period Binomial")
plt.legend(['Mprice','V_Binom5'])
plt.show()


# # 10 Period

# In[137]:


V_binom10 = []
S0 = Sa # initial stock price
dt = ((date_objectA.date() - today).days)/30 # expiry
N = 10000 # number of samples
n = 10     # number of steps
u = []
d = []
q = []
r = 0.00001 # interest rate

sigma = C_IV_list # volatility
for i in range(0,len(C_IV_list)):
    u.append(np.exp(sigma[i]*np.sqrt(dt))) # up return
    down = np.exp(sigma[i]*np.sqrt(dt))
    d.append(1/down) # down return
for i in range(0,len(u)):
    q.append((np.exp(r*dt) - d[i])/(u[i]-d[i]))
S =[]
for i in range(0,len(Strikes)):
    U = np.random.binomial(n, q[i], N) # result of flipping a coin n times, with  N samples of size n
    S.append(S0*u[i]**U*d[i]**(n-U))
    V_binom10.append(np.exp(-r*dt*n)*np.sum(np.maximum(S[i]-C_K_list[i],0))/N)
plt.plot(C_K_list,Mprice, 'r')
plt.plot(C_K_list,V_binom10, 'b')
plt.xlabel("Strike Prices")
plt.ylabel("Option Prices")
plt.title("Option Prices Vs Strikes for 10 Period Binomial")
plt.legend(['Mprice','V_Binom10'])
plt.show()


# # 15 Period

# In[138]:


V_binom15 = []
S0 = Sa # initial stock price
dt = ((date_objectA.date() - today).days)/30 # expiry
N = 10000 # number of samples
n = 15    # number of steps
u = []
d = []
q = []
r = 0.00001 # interest rate

sigma = C_IV_list # volatility
for i in range(0,len(C_IV_list)):
    u.append(np.exp(sigma[i]*np.sqrt(dt))) # up return
    down = np.exp(sigma[i]*np.sqrt(dt))
    d.append(1/down) # down return
for i in range(0,len(u)):
    q.append((np.exp(r*dt) - d[i])/(u[i]-d[i]))
S =[]
for i in range(0,len(Strikes)):
    U = np.random.binomial(n, q[i], N) # result of flipping a coin n times, with  N samples of size n
    S.append(S0*u[i]**U*d[i]**(n-U))
    V_binom15.append(np.exp(-r*dt*n)*np.sum(np.maximum(S[i]-C_K_list[i],0))/N)
plt.plot(C_K_list,Mprice, 'r')
plt.plot(C_K_list,V_binom15, 'b')
plt.xlabel("Strike Prices")
plt.ylabel("Option Prices")
plt.title("Option Prices Vs Strikes for 15 Period Binomial")
plt.legend(['Mprice','V_Binom15'])
plt.show()


# In[142]:


print("I am not sure why my the period increase is not fitting with the MarketPrice as steps increase, although I suspect it has something to do with my formulation of Market Price. If I was able to properly create the MarketPrice each increase in the number of steps would push the binomial line closer to the actual MarketPrice trend. This demonstrates how the binomial pricing model converges to Black-Scholes")


# In[ ]:




