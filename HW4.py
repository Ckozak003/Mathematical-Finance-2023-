#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from scipy.stats import norm
import yfinance as yf
from datetime import date, datetime
from dateutil import parser


# # Question 2

# In[11]:


def D_plus(r,sigma,T,K,So):
    return(((r+(sigma**2)/2)*T-log(K/So))/sigma*np.sqrt(T))      
def D_minus(r,sigma,T,K,So): 
    return(((r-(sigma**2)/2)*T-log(K/So))/sigma*np.sqrt(T))
def BSCall(S, K, T, r, sigma):
    Price = S * \
            norm.cdf(D_plus(r, sigma, T, K, S))-K*exp(-r*T) * \
            norm.cdf(D_minus(r, sigma, T, K, S))
    return Price 
def delta(r,sigma,tau,K,S):                  #fx
    return norm.cdf(D_plus(r,sigma,Tau,K,S))
def theta(r,sigma,tau,K,S):                  #ft
    return (-S*norm.pdf(D_plus(r,sigma,tau,K,S))*sigma/(2*np.sqrt(tau))) - \
            r*K*np.exp(-r*tau)*norm.cdf(D_minus(r,sigma,tau,K,S)) 
def gamma(r,sigma,tau,K,S):                  #fxx
    return (norm.pdf(D_plus(r,sigma,tau,K,S))/(S*sigma*np.sqrt(tau)))
def BMPath(T,dt):                       #creating BM path
    t = 0
    s = 0
    W =[0]
    while t < (T-dt):
        t+=dt
        r = np.random.normal(0,sqrt(dt))
        s += r
        W.append(s)
    return W


# In[31]:


S0 = 100
r = .01
u = .02
T = 1
dt = .001
sig = .2
F = np.zeros(1000)
Tau = np.zeros(1000)
Tau[0]=1
dSt = np.zeros(1000)
K = 80
W = [0]
dWs = BMPath(T,dt)
S = np.zeros(1000)
for i in range(0,999):
    S[0] = S0
    #print(W[i])
    #print(dWs[i], 5*'---')
    W.append(W[i] + dWs[i])
    #print(W[i+1], '\n')
    S[i+1] = S0*np.exp((u - .5*sig**2)*dt*(i+1) + sig*dWs[i+1])
    Tau[i+1] = T-((i+1)/1000)
for i in range(0,999):
    dSt[i] = S[i+1] - S[i]
time = np.linspace(0,1-dt,1000)
plt.plot(time,S)


# In[ ]:


for i in range(0,1000):
    F[i] = BSCall(S[i],K,Tau[i],r,sig)
F2 = np.zeros(1000)
F2[0] = F[0]
delta = np.zeros(1000)
t = np.zeros(1000)
g = np.zeros(1000)
for j in range(0,999):
    delta[j+1] = delta[j] + (norm.cdf(D_plus(r,sig,Tau[j+1],K,S[j+1]))*dSt[j+1])
    t[j+1] = t[j] + (theta(r,sig,Tau[j+1],K,S[j+1])*dt)
    g[j+1] = g[j] + (.5*gamma(r,sig,Tau[j+1],K,S[1+j])*sig**2*(S[j+1])**2*dt)
    F2[j+1] = F[0] + delta[j+1] + t[j+1] + g[j+1]

plt.plot(time,F, label = 'Direct')
#plt.plot(time,F2, 'r--', label = 'Greeks')
#plt.legend()
plt.xlabel('Time')
plt.ylabel('Price values')


# Theses paths should be identical, I am unsure of why there is significant difference between the two after a some time point(has to do with my code)

# # Question 3

# In[381]:


Apl = yf.Ticker("AAPL")
Apl_date = Apl.options
Expire = Apl_date[1]
date_objectA = parser.parse(Apl_date[1])           #creates object to find dates 
optA = Apl.option_chain(str(date_objectA.date()))  #gets the options for this stock based on expiration period
call_optA = optA.calls                             #gets you the call options based on the date 
today = date.today()
display(call_optA.iloc[30])
Strike = yf.Ticker('AAPL230310C00146000').history('max')    #strike = 146 in liquid range
Strike


# In[389]:


AAPL = yf.Ticker("AAPL")
historical = AAPL.history(period="max")
historical.tail(20)


# Empirically checking the delta with strike = 146

# In[387]:


Value_1 = Strike.Close[-1]
Value_2 = Strike.Close[-2]
Stock1 = historical.Close[-1]
Stock2 = historical.Close[-2]
dlta = abs((Value_2-Value_1)/(Stock2-Stock1))
print('The empircal value of delta is: ', dlta)


# Theoretical checking delta with strike = 146

# In[400]:


K = 146

stocks = historical.Close[-5:]
Sprice = Stock1
var = np.zeros(len(stocks))
mu = sum(stocks)/len(stocks)
for i in range(len(stocks)):
    var[i] = (stocks[i]-mu)**2
sigma = (sum(var)/len(stocks))**.5
r = .01
t = (date_objectA.date() - today).days/252

dlta2 = norm.cdf(D_plus(r,sigma,t,K,Sprice))
dlta2


# The theoretical and empirical are close, and doing this assignment over the weekend may have impacted the results somewhat. I would think that these two values should be nearly identical.

# Gamme calculations:   We need 3 points
# 

# In[401]:


Value_1 = Strike.Close[-1]
Value_2 = Strike.Close[-2]
Value_3 = Strike.Close[-3]
Stock1 = historical.Close[-1]
Stock2 = historical.Close[-2]
Stock3 = historical.Close[-3]
gam_emp = abs((Value_1-2*Value_2+Value_3)/(Stock1 + Stock2)*(Stock2-Stock3))
print("The empirical estimation of gamma is: ", gam_emp)


# In[404]:


gam_theory = norm.pdf(D_plus(r,sigma,t,K,Sprice))/(Sprice*sigma*np.sqrt(t))
print('The theoretical estimation of gamma is: ', gam_theory)


# The two are fairly similar.

# In[ ]:




