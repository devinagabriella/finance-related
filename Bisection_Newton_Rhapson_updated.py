import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
np.random.seed(1)

## Comparing Bisection and Newton-Rhapson Method to Compute Implied Volatility 
# #read data from excel file
data = pd.read_csv('implied vol surface.csv')

# Black Scholes European Call Formula
def BS(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# extract only 1M - 5Y maturities and 80% to 120% moneyness
S = 1 #spot price
r = 4/100
imp_vol = data.iloc[:11,5:14].values
imp_vol = imp_vol/100 # make implied volatility into percentage
K = np.array([80.0,90.0,95.0,97.5,100.0,102.5,105.0,110.0,120.0])/100
Tenor = np.array([1/12,2/12,3/12,6/12,9/12,12/12,18/12,2,3,4,5])
call= np.zeros(shape = imp_vol.shape)

# compute the European call option prices
for i in range(len(imp_vol)):
    for j in range(len(imp_vol[0])):
        sigma = imp_vol[i,j]
        call[i,j] = BS(S,K[j],r,imp_vol[i,j],Tenor[i])

# convert back to implied volatility using bisection method and Newton-Rhapson method
# Bisection method
def bisection(S0, K, r, C, T, a, b, tol, max_iter = 1000):
    for _ in range(max_iter):
        fa = BS(S0, K, r, a, T) - C
        m = (a+b)/2
        fm = BS(S0, K, r, m, T) - C
        if abs(fm) < tol:
            return m
        else:
            if fa*fm <0:
                b = m
            else:
                a = m

imp_vol_bisection = np.zeros(shape=imp_vol.shape)
mse_bisection = np.zeros(shape=imp_vol.shape)
for i in range(len(imp_vol)):
    for j in range(len(imp_vol[0])):
        imp_vol_bisection[i,j] = bisection(S,K[j],r,call[i,j],Tenor[i],a=0.1,b=1,tol=1e-8)
        mse_bisection[i,j] = (imp_vol[i,j] - imp_vol_bisection[i,j])**2
        
global_rmse_bisection = np.sqrt(np.mean(mse_bisection))
rmse_bisection_by_moneyness = np.sqrt(np.mean(mse_bisection,axis=0))
rmse_bisection_by_tenor = np.sqrt(np.mean(mse_bisection,axis=1))

# Newton-Rhapson Method
# First implement a function to find the derivative
def numerical_derivative(S0, K, r, sigma, T, C, h=1e-5):
    f_plus = BS(S0, K, r, sigma + h, T)
    f_minus = BS(S0, K, r, sigma - h, T)
    return (f_plus - f_minus) / (2 * h)

def Newton_Rhapson(S0, K, r, C, T, initial_sigma, tol, max_iter = 1000):
    sigma = initial_sigma
    for _ in range(max_iter):
        fx = BS(S0, K, r, sigma, T) - C
        vega_fx = numerical_derivative(S0, K, r, sigma, T, C)
        sigma_new = sigma - (fx/vega_fx)
        if abs(sigma-sigma_new) < tol:
            return sigma_new
        sigma = sigma_new

# Compute the implied volatility using Newton-Rhapson Method
imp_vol_NR = np.zeros(shape=imp_vol.shape)
mse_NR = np.zeros(shape=imp_vol.shape)
for i in range(len(imp_vol)):
    for j in range(len(imp_vol[0])):
        imp_vol_NR[i,j] = Newton_Rhapson(S,K[j],r,call[i,j],Tenor[i],initial_sigma=0.25,tol=1e-5)
        mse_NR[i,j] = (imp_vol[i,j] -imp_vol_NR[i,j])**2

global_rmse_NR = np.sqrt(np.mean(mse_NR))
rmse_NR_by_moneyness = np.sqrt(np.mean(mse_NR,axis=0))
rmse_NR_by_tenor = np.sqrt(np.mean(mse_NR,axis=1))

# Output------------------------------

print("Global RMSE (Bisection):", global_rmse_bisection)
print("Global RMSE (Newton-Raphson):", global_rmse_NR)

# Output RMSE by maturity into latex table
maturity_labels = data.iloc[:11,0].values
df_maturity = pd.DataFrame({
    'Maturity': maturity_labels,
    'Bisection RMSE': rmse_bisection_by_tenor,
    'Newton-Raphson RMSE': rmse_NR_by_tenor
})
latex_by_maturity = df_maturity.to_latex(index=False, float_format="%.5f")
print(latex_by_maturity)

# Output RMSE by moneyness into latex table
strike_labels = data.columns.values[5:14]
df_strike = pd.DataFrame({
    'Strike': strike_labels,
    'Bisection RMSE': rmse_bisection_by_moneyness,
    'Newton-Raphson RMSE': rmse_NR_by_moneyness
})
latex_by_strike = df_strike.to_latex(index=False, float_format="%.5f")
print(latex_by_strike)

