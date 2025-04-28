"""
This program price a call option by backward induction on binomial tree. 
It compares the value obtained with value from Black-Scholes formula.
"""

S0 = 10.0
sigma = 0.2
r = 0.0
T = 0.5
K = 10.0

# Binomial tree - backward induction

import numpy as np
from scipy.stats import norm

N = 200
dt = T/N
u = np.exp(sigma*np.sqrt(dt))
d = 1/u
p = (np.exp(r*dt)-d)/(u-d)
S = np.zeros((N+1,N+1))
V = np.zeros(N+1) # option value

for i in range(N+1):
    for j in range(i+1):
        S[j,i] = S0*(u**j)*(d**(i-j))
        V[j] = max(S[j,i]-K,0)

for i in range(N-1,-1,-1):
    for j in range(i+1):
        V[j] = np.exp(-r*dt)*(p*V[j+1]+(1-p)*V[j])

# price option using Black-Scholes formula for comparison
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
black_scholes_value = S0*norm.cdf(d1) - K*(1-r*T)*norm.cdf(d2)

# put call parity
call = V[0]
put = call - S[0, N] + K*np.exp(-r*T)

print(f"Call Option Value\nUsing Backward Induction: {call}\nUsing Black-Scholes:{black_scholes_value}\n")
print(f"Put Option Value using Put-Call parity: {put}")