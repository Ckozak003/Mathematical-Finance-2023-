#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.stats import norm


# # Question 3

# a.)
# 
# MP risk equation:  u - r = sigma1 * theta1 + sigma2 + theta2

# In[93]:


#a continued
u = .08
sigma1 = .3
r = .03
sigma2 = .25
theta1 = .5
theta2 = ((u-r)-sigma1*theta1)/sigma2
print("Theta 2 is: ", round(theta2,ndigits = 2))

def D_plus(So,K,T,r,sigma):
    return(((r+(sigma**2)/2)*T-log(K/So))/(sigma*np.sqrt(T)))      #The d+ aspect when calculating BS value

def D_minus(So,K,T,r,sigma): 
    return(((r-(sigma**2)/2)*T-log(K/So))/(sigma*np.sqrt(T)))      #The d- aspect when calc BS value

def BSCall(S, K, T, r, sigma):                                     #Actually calc BS value
    Price = S * \
            norm.cdf(D_plus(S, K, T, r, sigma))-K*np.exp(-r*T) * \
            norm.cdf(D_minus(S, K, T, r, sigma))
    return Price 


# b.)

# In[94]:


MC_list = []
K = 100
N = 1000
T = 1
S = np.zeros(N)
S0 = 100
S[0] = S0
dt = T/N
MC_n = 5000
for i in range(0,MC_n):
    Wt1 = np.random.normal(0,np.sqrt(dt),N)
    Wt2 = np.random.normal(0,np.sqrt(dt),N)
    ZT1 = np.exp(-.5*theta1**2*T - theta1*Wt1[N-1])
    ZT2 = np.exp(-.5*theta2**2*T - theta2*Wt2[N-1])
    for j in range(0,N-1):
        dS = u*S[j]*dt + sigma1*S[j]*Wt1[j] + sigma2*S[j]*Wt2[j]
        S[j+1] = S[j] + dS
    E_Price = np.exp(-r*T)*np.maximum(0,S[N-1]-K)*(ZT1*ZT2)
    MC_list.append(E_Price)
Expected_MC_Price = sum(MC_list)/MC_n
print('The Monte Carlo simulated price of the physical model for this call option is: ', Expected_MC_Price)


# c.)

# In[23]:


theta1 = 1
theta2 = ((u-r)-sigma1*theta1)/sigma2
MC_list = []
K = 100
N = 1000
T = 1
S = np.zeros(N)
S0 = 100
S[0] = S0
dt = T/N
MC_n = 5000
for i in range(0,MC_n):
    Wt1 = np.random.normal(0,np.sqrt(dt),N)
    Wt2 = np.random.normal(0,np.sqrt(dt),N)
    ZT1 = np.exp(-.5*theta1**2*T - theta1*Wt1[N-1])
    ZT2 = np.exp(-.5*theta2**2*T - theta2*Wt2[N-1])
    for j in range(0,N-1):
        dS = u*S[j]*dt + sigma1*S[j]*Wt1[j] + sigma2*S[j]*Wt2[j]
        S[j+1] = S[j] + dS
    E_Price = np.exp(-r*T)*np.maximum(0,S[N-1]-K)*(ZT1*ZT2)
    MC_list.append(E_Price)
Expected_MC_Price = sum(MC_list)/MC_n
print('The Monte Carlo simulated price of the physical model for this call option is: ', Expected_MC_Price)


# d.)

# In[25]:


sigma = np.sqrt(sigma1**2 + sigma2**2)
BS_Price = BSCall(100,K,T,r,sigma)
print('The BS Call price of this option is: ', BS_Price)


# It is interesting to see that when sigma1 is .5 the price generated via Monte Carlo simulation is closer to the BS price than when sigma1 is 1. This could show how the better physical model attributes a weight of .5 to theta1 and -.4 to theta2.

# # Question 4

# a.)

# In[95]:


u3 = .07
sigma3 = .35
r = .03
sigma2 = .25
theta1 = .5
theta2 = ((u-r)-sigma1*theta1)/sigma2


# In[96]:


MC_list = []
K = 100
N = 1000
T = 1
X = np.zeros(N)
X0 = 95
X[0] = X0
dt = T/N
MC_n = 5000
for i in range(0,MC_n):
    Wt1 = np.random.normal(0,np.sqrt(dt),N)
    Wt2 = np.random.normal(0,np.sqrt(dt),N)
    ZT1 = np.exp(-.5*theta1**2*T - theta1*Wt1[N-1])
    ZT2 = np.exp(-.5*theta2**2*T - theta2*Wt2[N-1])
    for j in range(0,N-1):
        dX = u3*X[j]*dt + sigma3*X[j]*Wt1[j] 
        X[j+1] = X[j] + dX
    E_Price3 = np.exp(-r*T)*np.maximum(0,X[N-1]-K)*(ZT1*ZT2)
    MC_list.append(E_Price3)
Expected_MC_Price = sum(MC_list)/MC_n
V0 = Expected_MC_Price
print('The Monte Carlo simulated price of the physical model for this is: ', Expected_MC_Price)


# b.)

# In[90]:


theta1 = 1
theta2 = ((u-r)-sigma1*theta1)/sigma2
MC_list = []
K = 100
N = 1000
T = 1
X = np.zeros(N)
X0 = 95
X[0] = X0
dt = T/N
MC_n = 5000
for i in range(0,MC_n):
    Wt1 = np.random.normal(0,np.sqrt(dt),N)
    Wt2 = np.random.normal(0,np.sqrt(dt),N)
    ZT1 = np.exp(-.5*theta1**2*T - theta1*Wt1[N-1])
    ZT2 = np.exp(-.5*theta2**2*T - theta2*Wt2[N-1])
    for j in range(0,N-1):
        dX = u3*X[j]*dt + sigma3*X[j]*Wt1[j] 
        X[j+1] = X[j] + dX
    E_Price3 = np.exp(-r*T)*np.maximum(0,X[N-1]-K)*(ZT1*ZT2)
    MC_list.append(E_Price3)
Expected_MC_Price = sum(MC_list)/MC_n
print('The Monte Carlo simulated price of the physical model for this is: ', Expected_MC_Price)


# c.)
# Since this is a non-tradeable object, these numbers do not have real world implications. The price of the weather (for example) is not attainable so these simulations basically show the same results as problem 3 but with something you can't actually buy.

# d.)

# In[97]:


#S is from part b of problem 3

pi0 = V0
pi_t = np.zeros(N)
pi_t[0] = pi0


delta0 = norm.cdf(D_plus(S0,K,T,r,sigma3))
deltat = np.zeros(N)
deltat[0]=delta0


y0 = pi0-delta0*S0
yt = np.zeros(N)
yt[0] = y0

Tau = np.zeros(N+1)
for i in range(N+1):
    Tau[i] = 1 - (i/N)
for i in range(N-1):
    deltat[i+1] = norm.cdf(D_plus(S[i+1],K,Tau[i+1],r,sigma3))
    pi_t[i+1] = deltat[i]*S[i+1] + yt[i]*(1+r*dt) 
    yt[i+1] = pi_t[i+1] - deltat[i+1]*S[i+1]
print('The delta hedged portfolio value is', pi_t[N-1])
print('The value of (Xt-K)+ is: ', np.maximum(0,X[N-1]-K))


# As we can see, these values do not match each other.

# In[ ]:




