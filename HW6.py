#!/usr/bin/env python
# coding: utf-8

# In[16]:


from math import *
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# # Question 1

# In[223]:


def BMPath(T,dt):
    t = 0
    s = 0
    W = [0]
    while t < (T - dt):
        t+= dt
        x = np.random.normal(0,sqrt(dt))
        s += x
        W.append(s)
    return W
def indicator(x, y):              #for when values are >=
    if x >= y:
        return 1
    else:
        return 0
def indicator_strict(x,y):        #for when values are > (STRICTLY)
    if x > y:
        return 1
    else: 
        return 0
def indicator_middle_rightStrict(x,y,z):         #for when x<= y < z
    if x <= y < z:
        return 1
    else:
        return 0
def H_function(alpha, beta, k, b, W, M):       #H function formulation(too specific atm to use in all cases)
    indicator_1 = indicator(W,k)               #ADD MORE SCENARIOS TO THIS!!!
    indicator_2 = indicator(M,b)
    if k < 0:
        indicator_1 = 1
    ans = np.exp(alpha*W + beta * M)*indicator_1*indicator_2
    return ans


# In[23]:


T = 1
dt = .001
N = int(T/dt)
time = np.linspace(0,1,N)
W_path = BMPath(T,dt)
M_path = np.zeros(N)
M_path[0] = 0

for i in range(0,N-1):
    if M_path[i] < W_path[i]:
        M_path[i+1] = W_path[i]
    else:
        M_path[i+1] = M_path[i]
plt.plot(time, M_path, label = 'Path of Max Function')
plt.plot(time,W_path,label = 'Path of BM')
plt.legend()


# # Question 2

# PART A)

# In[89]:


alpha, s, b, k = 2,1,1,2
expectation_analy = np.exp(s*.5*alpha**2)*norm.cdf((s*alpha - k)/np.sqrt(s))
print(expectation_analy)


# In[102]:


MC_n = 100000
dt = 1/MC_n
Ws = np.random.normal(0,np.sqrt(s),MC_n)
MC_list = []
for i in range(0,MC_n):
    MC_list.append((indicator(Ws[i],k)*np.exp(Ws[i]*alpha)))
Value = sum(MC_list)/MC_n
print(Value)


# As shown above by the MC simulation, the expactation value matches the analytical value very closely.

# PART B)

# In[103]:


alpha, s, b, k = 2,1,2,1
expectation_analy_b =  np.exp(s*.5*alpha**2)*(norm.cdf((s*alpha - b)/np.sqrt(s)) + np.exp(2*alpha*b)*
                                             (norm.cdf((-s*alpha - b)/np.sqrt(s)) - norm.cdf((-s*alpha - 2*b + k)
                                                                                             /np.sqrt(s))))
print(expectation_analy_b)


# In[173]:


MC_n_b = 10000
dt_b = 1/MC_n_b
Ws_b = np.random.normal(0,np.sqrt(s),MC_n_b)
MC_list_b = []
M_pathB = np.zeros(MC_n_b)

for i in range(0,MC_n_b-1):
    if M_pathB[i] < Ws_b[i]:
        M_pathB[i+1] = Ws_b[i]
    else:
        M_pathB[i+1] = M_pathB[i]
        
for i in range(0,MC_n_b):
    #Comments are me playing around with different indicator functions
    #E1 = (indicator(Ws_b[i],b)*indicator_strict(M_pathB[i],b)*np.exp(Ws_b[i]*alpha))
    #E2 = (indicator_middle_rightStrict(k,Ws_b[i],b)*indicator_strict(M_pathB[i],b)*np.exp(Ws_b[i]*alpha))
    E = indicator(Ws_b[i],k)*indicator_strict(M_pathB[i],b)*np.exp(Ws_b[i]*alpha)
    MC_list_b.append(E)
    
Value_b = sum(MC_list_b)/MC_n_b
print(Value_b)


# As shown above, the Monte Carlo estimation closely matches the analytical value.

# PARCT C)

# In[182]:


alpha, beta, s, b = 2,2,1,2
expectation_analy_c =  (((beta + alpha) / (beta + alpha*2)) * 2 * np.exp(s * .5 * (alpha + beta)**2) *
                        norm.cdf((s * (alpha + beta) - b) / np.sqrt(s)) + ((2 * alpha)/(beta + 2*alpha)) *
                        np.exp(.5 * alpha**2 * s) * np.exp(b * (beta + 2*alpha)) * norm.cdf((-(alpha * s + b))/
                                                                                           np.sqrt(s)))
                        
print(expectation_analy_c)


# In[193]:


MC_n_c = 5000
dt_c = 1/MC_n_c
Ws_c = np.random.normal(0,np.sqrt(s),MC_n_c)
MC_list_c = []
M_pathC = np.zeros(MC_n_c)


for i in range(0,MC_n_c-1):
    if M_pathC[i] < Ws_c[i]:
        M_pathC[i+1] = Ws_c[i]
    else:
        M_pathC[i+1] = M_pathC[i]
        
for i in range(0,MC_n_c):
    E = indicator_strict(M_pathC[i],b)*np.exp(Ws_c[i]*alpha + beta*M_pathC[i]) 
    MC_list_c.append(E)

    
Value_c = sum(MC_list_c)/MC_n_c
print(Value_c)


# As shown above, the Monte Carlo estimation closely matches the analytical value.

# # Question 3

# Part a)
# 
# V0 = e^((-rT) - 1/2 * theta^2 * T) * S0 * H(theta, sigma, negative infinity, {1/sigma} * ln(K)) - K * H(theta, 0, neg infinity, {1/sigma} * ln(K))
#                                          

# Part b)

# In[266]:


S0 = 100
r = .03
sigma = .25
T = 1
dt = T/1000
N = int(T/dt)
K = 100
theta = (r-.5*sigma**2)/sigma

Wt = BMPath(T,dt)         #Brownian Motion
for i in Wt:
    i = i + theta*T

St = np.zeros(N)
time = np.linspace(0,1,N)

St[0] = S0
for i in range(N-1):
    St[i+1] = S0*np.exp((r-.5*sigma**2)*dt + sigma*Wt[i])      #stock prices just in case I need it later
    
    
M_s = np.zeros(N)
for i in range(N-1):
    if M_s[i] < St[i]:
        M_s[i+1] = St[i]
    else:
        M_s[i+1] = M_s[i]
    
M = np.zeros(N)

for i in range(0,N-1):          #finding max path
    if M[i] < Wt[i]:
        M[i+1] = Wt[i]
    else:
        M[i+1] = M[i]
coef = S0 * np.exp((-r*T) - .5*theta**2*T)
H1 = H_function(theta,sigma,-1,(1/sigma)*np.log(M_s[N-1]/St[N-1]), Wt[N-1], M[N-1])
H2 = H_function(theta,0,-1,(1/sigma)*np.log(M_s[N-1]/St[N-1]), Wt[N-1], M[N-1])
V0 = coef*H1 - K * H2
print(V0)


# Part c)

# In[269]:


MC_N = 5000
MC_L = []
for i in range(0,MC_N):
    Wt = BMPath(T,dt)         #Brownian Motion
    for i in Wt:
        i = i + theta*T

    St = np.zeros(N)
    time = np.linspace(0,1,N)

    St[0] = S0
    for i in range(N-1):
        St[i+1] = S0*np.exp((r-.5*sigma**2)*dt + sigma*Wt[i])      #stock prices just in case I need it later
    
    
    M_s = np.zeros(N)
    for i in range(N-1):
        if M_s[i] < St[i]:
            M_s[i+1] = St[i]
        else:
            M_s[i+1] = M_s[i]
    
    M = np.zeros(N)

    for i in range(0,N-1):          #finding max path
        if M[i] < Wt[i]:
            M[i+1] = Wt[i]
        else:
            M[i+1] = M[i]
    coef = S0 * np.exp((-r*T) - .5*theta**2*T)
    H1 = H_function(theta,sigma,-1,(1/sigma)*np.log(M_s[N-1]/St[N-1]), Wt[N-1], M[N-1])
    H2 = H_function(theta,0,-1,(1/sigma)*np.log(M_s[N-1]/St[N-1]), Wt[N-1], M[N-1])
    MC_L.append(coef*H1 - H2)
Value = sum(MC_L)/MC_N
print(Value)
print(V0 - Value)


# I could not get them very close. There may also be an issue in my derivation of V0.

# In[ ]:




