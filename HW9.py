#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode
from math import log


# In[2]:


#Functions

def D_plus(So,K,T,r,sigma):
    return(((r+(sigma**2)/2)*T-log(K/So))/(sigma*np.sqrt(T)))      #The d+ aspect when calculating BS value

def D_minus(So,K,T,r,sigma): 
    return(((r-(sigma**2)/2)*T-log(K/So))/(sigma*np.sqrt(T)))      #The d- aspect when calc BS value

def BSCall(S, K, T, r, sigma):                                     #Actually calc BS value
    Price = S * \
            norm.cdf(D_plus(S, K, T, r, sigma))-K*np.exp(-r*T) * \
            norm.cdf(D_minus(S, K, T, r, sigma))
    return Price 

def BSPut(S,K,T,r,sigma):
    Price = K*np.exp(-r*T) * \
            norm.cdf(-D_minus(S, K, T, r, sigma)) - S * \
            norm.cdf(-D_plus(S, K, T, r, sigma))
    return Price
def PerpOpt(x,r,sigma,K,Lstar,A):
    if (0 <= x) & (x < Lstar):
        return K-x
    else:
        return A*x**((-2*r)/sigma**2)


# # Question 1

# $1a) \space Let \space τ = inf(t ≥ 0 : St = L)\space ∧ \space T,\space where \space L = 105.\space Plot \space the \space histogram \space of \space τ$

# In[5]:


T = 3
S0 = 100
r = .02
sigma = .3
L = 105
N = 1000
dt = T/N

time = np.linspace(0,T,N)



MC_n = 10000
MC_List = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) # Easier way to get a BM path
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time[i+1] + sigma*W[i+1]))
        if St[i+1] >= L:
            MC_List.append(time[i+1]) 
            break
        elif len(St) == N:
            MC_List.append(time[i+1])
        else: continue
    
plt.hist(MC_List, density = True, bins = 100)
plt.ylabel('Frequency')
plt.xlabel('Stopping Time')
print(mode(MC_List))


# $b.)\space The \space optional\space sampling\space theorem\space states\space that\space \tilde{E}(e^{−rτ} S_τ ) = S_0.\space Verify\space this\space statement\space using\space Monte\space Carlo\space simulation\space and\space the\space τ\space in\space part\space a$

# In[19]:


T = 3
S0 = 100
r = .02
sigma = .3
L = 105
N = 1000
dt = T/N


MC_n = 10000
MC_List = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) 
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time[i+1] + sigma*W[i+1]))
    MC_List.append(np.exp(-r*3)*St[-1])
S_0_val = sum(MC_List)/MC_n
print(S_0_val)


# $C.) \space Consider\space a\space European\space call\space option\space on\space S\space with\space strike\space K = 100.\space Find\space the\space Black-Scholes\space price\space for\space this\space call\space option.\space Compare\space it\space with\space the\space expectation:\space \tilde{E}(e^{−rτ}(Sτ −K)^{+}),\space where\space τ\space is\space the\space stopping\space time\space from\space part\space a.\space What’s\space your\space expectation\space and\space does\space the\space result\space agree\space with\space your\space expectation?$

# In[27]:


K = 100
T = 3
S0 = 100
r = .02
sigma = .3
L = 105
N = 1000
dt = T/N


V_Call = BSCall(S0, K, T, r, sigma)


MC_n = 100000
MC_List = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) 
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time[i+1] + sigma*W[i+1]))
    MC_List.append(np.exp(-r*3)*max(St[-1]-K,0))
val = sum(MC_List)/MC_n
print(V_Call - val)

#DON'T RUN AGAIN IT TAKES FOREVER


# The above shows the difference between the the analytical value and the MC value.

# # Question 2

# In[122]:


K = 105
S0 = 100
St = np.linspace(20,60,N)
r = .02
sigma = .3
N = 1000
dt = T/N

dW = np.random.normal(0, np.sqrt(dt), N) 
W = np.cumsum(dW)

time = np.linspace(0,T,N)

Lstar = (2*r/(2*r + sigma**2))*K
A = (sigma**2/(2*r))*Lstar**(((2*r)/sigma**2) + 1)
V_perp = []
V_put = []
for i in range(0,N):
    V_perp.append(PerpOpt(St[i],r,sigma,K,Lstar,A))
for i in range(0,N):
    j = max((K-St[i]),0)
    V_put.append(j)
plt.plot(St,V_perp, label = 'v(x)')
plt.plot(St,V_put, label = '(K-x)+')
plt.vlines(Lstar, ymin = 40, ymax = 90,ls = '--', label = "L*")
plt.xlabel("Stock Price")
plt.ylabel("V")
plt.legend()
plt.show()

for i in range(0,N):         #graphs are too close to visually see the split, this gives a decent approx
    if (V_perp[i] - V_put[i]) > .0001:
        print(St[i])
        print(Lstar)
        break
    else: continue



# Part b.) As shown above, for any value of x, v(x) is greater than (K-x)+ when x > 32.41. This value is pretty close to our L*(32.31).

# Part c.)

# In[121]:


def pde_perp_put(r,v,x,sigma,L,A):
    if (0 <= x) & (x < L):
        return r*v + x*r
    else:
        c = ((-2*r)/sigma**2)
        return r*v - c*A*x**(c-1)*r*x - (.5* (2*r)*(sigma**2 + 2*r) *A*x**(c-2)/sigma**4) * x**2 * sigma**2 

    
St = np.linspace(29,35,N)
Lstar = (2*r/(2*r + sigma**2))*K
A = (sigma**2/(2*r))*Lstar**(((2*r)/sigma**2) + 1)
    
    
perp = []    
for i in range(0,N):
    perp.append(pde_perp_put(r,PerpOpt(St[i],r,sigma,K,Lstar,A),St[i],sigma,Lstar,A))

    
    
plt.plot(St,perp, label = 'f(x)')
plt.vlines(Lstar, ls = '--', ymin = 0, ymax = 2.5, label = "L*")
plt.xlabel("Stock Price")
plt.ylabel("V")
plt.legend()
plt.show()






# f(x) is shown to be zero on the same domain that v(x) > (K-x)+

# # Question 3

# Part a)

# In[135]:


T = 3
S0 = 100
r = .05
sigma = .1
K = 105
L= 95
N = 1000
dt = T/N


time = np.linspace(0,T,N)


MC_n = 10000
MC_List = []
MC_tau = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) 
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time[i+1] + sigma*W[i+1]))
        if St[i+1] >= L:
            MC_List.append(np.exp(-r*time[i+1])*np.max((K-St[i+1],0)))
            MC_tau.append(time[i+1])
            break
        elif i == N:
            MC_List.append(np.exp(-r*time[i+1])*np.max((K-St[i+1],0)))
            mC_tau.append(time[i+1])
        else: continue
            
            
plt.hist(MC_tau, bins = 100)
plt.ylabel('Frequency')
plt.xlabel('Stopping Time')
print(mode(MC_tau))

E_Val = sum(MC_List)/MC_n
print(E_Val)


# Part b.)

# In[127]:


T = 3
S0 = 100
r = .05
sigma = .1
K = 105
L= 95
N = 1000
dt = T/N



val_BS = BSPut(S0,K,T,r,sigma)

print(val_BS)

print('The difference between the MC simulated value and BS value is: ', E_Val - val_BS)


# As shown above, the MC simulated value is larger.

# # Question 4

# Part a.)

# In[24]:


#YOU MIGHT JUST NEED TO HAVE A SET RANGE OF STOCK PRICES FROM LIKE 100-140 OR SOMETHING!!!!

S0 = 110
r = .05
sigma = .1
K = 105
T = 10
N = 1000



dt = T/N
time = np.linspace(0,T,N)
#St = np.linspace(90,120,N)
Lstar = (2*r/(2*r + sigma**2))*K
A = (sigma**2/(2*r))*Lstar**(((2*r)/sigma**2) + 1)

MC_n = 5000
MC_List = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) 
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time[i+1] + sigma*W[i+1]))
        L = np.max((K - St[i+1]),0)
        if (abs(PerpOpt(St[i+1],r,sigma, K,Lstar,A) - L)) < .01:
            MC_List.append(time[i+1]) 
            break
        elif i == N:
            MC_List.append(time[i+1])
        else: continue
       

plt.hist(MC_List, bins = 50)
plt.ylabel('Frequency')
plt.xlabel('Stopping Time')
print(mode(MC_List))


# part b)

# In[7]:


S0 = 110
r = .05
sigma = .1
K = 105
T = 10
N = 1000



dt = T/N
time2 = np.linspace(0,T,N)
#St = np.linspace(90,120,N)
Lstar = (2*r/(2*r + sigma**2))*K
A = (sigma**2/(2*r))*Lstar**(((2*r)/sigma**2) + 1)

MC_n = 5000
MC_List = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) 
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time2[i+1] + sigma*W[i+1]))
        L = Lstar
        if (abs(St[i+1] - L)) < .01:
            MC_List.append(time2[i+1]) 
            break
        elif i == N-1:
            MC_List.append(time2[i+1])
        else: continue

plt.hist(MC_List, bins = 50)
plt.ylabel('Frequency')
plt.xlabel('Stopping Time')
print(mode(MC_List))


# While the graphs do not entirely match, the stopping time is around the same (~ 1)

# Part c)

# In[9]:


# Other variables used from last part
T = 1
N = 1000
K = 110     #previous K gave extremely large differences


time3 = np.linspace(0,T,N)

dt = T/N

Analytical = PerpOpt(S0,r,sigma,K,Lstar,A)

tau_star = 1

MC_n = 5000
MC_List = []
for i in range(0,MC_n-1):
    dW = np.random.normal(0, np.sqrt(dt), N) 
    W = np.cumsum(dW)
    St = [S0]
    for i in range(0,N-1):
        St.append(S0*np.exp((r-(sigma**2)/2)*time3[i+1] + sigma*W[i+1]))
    MC_List.append(np.exp(-r)*max((K-St[N-1]),0))
MC_val = sum(MC_List)/MC_n
print(Analytical, MC_val)
print("The difference between the MC and Analytical value is: ", MC_val - Analytical)

