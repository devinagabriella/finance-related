import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
np.random.seed(1)

## Option Pricing and Stochastic Volatility Modelling 
"""
Compare between Euler discretizations schemes for Monte Carlo simulation under Heston Model
when the Feller condition is not satisfied. Out of the three variance function
in the variance process under the Heston model, we have the following schemes:
1. Absorption: take max(x,0) for all variance in the variance process.
2. Reflection: take abs(x) for all variance in the variance process.
3. Partial truncation: take max(x,0) for only the last variance function.
4. Full truncation: take max(x,0) for the second and the last variance function.
"""

# Heston model parameters
T = 5           # Time to maturity       
S0 = 100        # Initial stock price 
K = 100         # Strike price
r = 0.05        # Risk-free rate
V0 = 0.09       # Initial variance
theta = 0.09    # Long-term variance
kappa = 2       # Mean reversion speed
omega = 1       # Volatility of variance
rho = -0.3      # Correlation of Brownian motion
true_value = 34.9998 # True price of the European call option

# Simulation parameters
N = 50*T             # Number of time steps (e.g. 50 time steps each year)  
n_paths = 100000     # Number of simulations (paths)

# Black - Scholes model parameters
sigma = 0.3         # constant volatility of Black-Scholes model

def EulerDisc(S0, V0, T, steps_per_year, r, kappa, theta, omega, rho, scheme, n_paths):
    N = steps_per_year*T
    dt = T/N
    S = np.zeros((n_paths, N+1))
    V = np.zeros((n_paths, N+1)) 
    S[:, 0] = S0
    V[:, 0] = V0

    for i in range(N):
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = np.random.normal(0, 1, n_paths)
    
        dW_V = np.sqrt(dt) * Z1
        dW_S = rho * dW_V + np.sqrt(1 - rho**2) * np.sqrt(dt) * Z2

        # The Euler schemes options
        if scheme == 'a': #absorption
            V_f1 = np.maximum(V[:, i], 0) 
            V_f2 = np.maximum(V[:, i], 0)
            V_f3 = np.maximum(V[:, i], 0)

        elif scheme == 'r': #reflection
            V_f1 = abs(V[:, i]) 
            V_f2 = abs(V[:, i])
            V_f3 = abs(V[:, i])

        elif scheme == 'pt': #partial truncation
            V_f1 = V[:, i]
            V_f2 = V[:, i]
            V_f3 = np.maximum(V[:, i], 0)

        elif scheme == 'ft': #full truncation
            V_f1 = V[:, i]
            V_f2 = np.maximum(V[:, i], 0)
            V_f3 = np.maximum(V[:, i], 0)

        # use positive V_t for computation of S_t+1
        S[:,i+1] = S[:,i] * np.exp((r - 0.5 * V_f3) * dt + np.sqrt(V_f3) * dW_S)
        
        # negative V_t should stay negative for computation of V_t+1
        V[:,i+1] = V_f1 - kappa * dt * (V_f2 - theta) + omega * np.sqrt(V_f3) * dW_V
        
    return S[:, -1]

# Function to compute Bias, SE, RMSE and run time
def simulate_and_evaluate(omega, scheme, n_paths, steps_per_year, n_runs=5):

    results = []
    runtimes = []

    for _ in range(n_runs):
        start = time.time()
        S_T = EulerDisc(S0, V0, T, steps_per_year, r, kappa, theta, omega, rho, scheme, n_paths)
        payoff = np.exp(-r * T) * np.maximum(S_T - K, 0)
        price = np.mean(payoff)
        end = time.time()
        results.append(price)
        runtimes.append(end - start)
    
    mean_price = np.mean(results)
    std_error = np.sqrt(np.mean((results - np.mean(results))**2))
    bias = np.abs(mean_price - true_value)
    rmse = np.sqrt(bias**2 + std_error**2)
    avg_runtime = np.mean(runtimes)

    return {
        "Bias": bias,
        "Standard Error": std_error,
        "RMSE": rmse,
        "Run Time": avg_runtime,
        "Sample Mean Price": mean_price
    }

# Run for each scheme and for each pair of paths and steps/year
schemes = ['a','r','pt', 'ft'] 
scheme_names = {
    'a': 'Absorption',
    'r': 'Reflection',
    'pt': 'Partial truncation',
    'ft': 'Full truncation'
}
path_options = [10000, 40000, 160000]
steps_per_year_options = [20, 40, 80]

for scheme in schemes:
    for _ in range(3):
        print(f"{scheme_names[scheme]} | Paths: {path_options[_]}, Steps/year: {steps_per_year_options[_]}")
        result = simulate_and_evaluate(omega, scheme, path_options[_], steps_per_year_options[_])
        print(f"Result: Bias = {result['Bias']:.4f}, SE = {result['Standard Error']:.4f}, RMSE = {result['RMSE']:.4f}, Time = {result['Run Time']:.4f}")
        print('-' * 50)

## Find Delta and Gamma
"""
Compute Delta and Gamma of the European call option under Heston model
using the best performing schemes from the previous part, full truncation scheme.
Then compute Delta and Gamma under Black-Scholes model as comparison. 
"""
def heston_price(S0, delta_S, scheme='ft', n_runs = 5):
    # Computes C(T,S0), C(T, S0 + delta S), and C(T, S0 - delta S)
    C_minus = []
    C_0 = []
    C_plus = []
    for _ in range(n_runs):
        np.random.seed(25)
        for S in [S0 - delta_S, S0, S0 + delta_S]:
            ST = EulerDisc(S, V0, T, 50, r, kappa, theta, omega, rho, scheme, 100000)
            payoff = np.exp(-r * T) * np.maximum(ST - K, 0)

            if S == S0 - delta_S:
                C_minus.append(np.mean(payoff))
            elif S == S0:
                C_0.append(np.mean(payoff))
            elif S == S0 + delta_S:
                C_plus.append(np.mean(payoff))
    return np.array(C_minus), np.array(C_0), np.array(C_plus)

delta_S = 0.01 * S0
C_minus, C_0, C_plus = heston_price(S0, delta_S)

# Computes Delta and Gamma under the Heston model based on the finite difference method formula
delta = (C_plus - C_minus) / (2 * delta_S)
gamma = (C_plus - 2 * C_0 + C_minus) / (delta_S ** 2)
delta = np.mean(delta)
gamma = np.mean(gamma)

print(f"Heston Delta: {delta:.4f}")
print(f"Heston Gamma: {gamma:.6f}")

# Compute Delta and Gamma under the Black-Scholes model
def BS(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_delta = norm.cdf(d1)
    bs_gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    return bs_delta, bs_gamma

bs_delta, bs_gamma = BS(S0,K,r,sigma,T)
print(f"Black-Scholes Delta: {bs_delta:.4f}")
print(f"Black-Scholes Gamma: {bs_gamma:.6f}")

## Heston Model when Feller Condition Holds
"""
Price the European call option as the first part 
but with omega = 0.3 using the four schemes. 
The variance will not become negative since the Feller Condition now holds.
"""
for scheme in schemes:
    for _ in range(3):
        print(f"{scheme_names[scheme]} | Paths: {path_options[_]}, Steps/year: {steps_per_year_options[_]}")
        result = simulate_and_evaluate(0.3, scheme, path_options[_], steps_per_year_options[_])
        print(f"Result: Bias = {result['Bias']:.4f}, SE = {result['Standard Error']:.4f}, RMSE = {result['RMSE']:.4f}, Time = {result['Run Time']:.4f}")
        print('-' * 50)

## Price American Put Option
"""
Use the least square Monte Carlo method to estimate the price of an
American put option under the Heston model with the same parameters as 
the European call option. Use the full truncation scheme for the least square Monte Carlo.
Estimate continuation value using regression on asset price process (S) and variance process (V)
with the following basis function:

C(S,V) = a + bS + cS^2 + dV + eV^2 + fSV

"""
# Function to price the American put option using the least square Monte Carlo method
def american_put_lsm(S0, V0, T, K, r, kappa, theta, omega, rho, n_paths= 10000, steps_per_year =50):
    N = steps_per_year*T
    dt = T / N

    # Simulate S and V paths
    S = np.zeros((n_paths, N+1))
    V = np.zeros((n_paths, N+1))
    S[:, 0] = S0
    V[:, 0] = V0

    for i in range(N):
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)
        dW_V = np.sqrt(dt) * Z1
        dW_S = rho * dW_V + np.sqrt(1 - rho**2) * np.sqrt(dt) * Z2

        # Use the full truncation scheme of Euler discretizations
        V_f1 = V[:, i]
        V_f2 = np.maximum(V[:, i], 0)
        V_f3 = np.maximum(V[:, i], 0)

        S[:, i+1] = S[:, i] * np.exp((r - 0.5 * V_f3) * dt + np.sqrt(V_f3) * dW_S)
        V[:, i+1] = V_f1 - kappa * dt * (V_f2 - theta) + omega * np.sqrt(V_f3) * dW_V

    # Payoff matrix
    payoff = np.maximum(K - S, 0)
    cashflows = payoff[:, -1]

    # Backward induction
    for t in range(N-1, 0, -1):
        itm = payoff[:, t] > 0
        X = S[itm, t]
        Y = cashflows[itm] * np.exp(-r * dt)
        V_t = np.maximum(V[itm, t], 0)

        # Quadratic basis functions
        A = np.column_stack([np.ones(X.shape[0]), X, V_t, X**2, V_t**2, X * V_t])
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation = A @ coeffs

        exercise = payoff[itm, t] > continuation
        cashflows[itm] = np.where(exercise, payoff[itm, t], cashflows[itm] * np.exp(-r * dt))

    price = np.mean(cashflows * np.exp(-r * dt))
    return price

# Function to price the European put option
def european_put_price_heston():
    ST = EulerDisc(S0, V0, T, 50, r, kappa, theta, omega, rho, 'ft', 10000)
    payoff = np.exp(-r * T) * np.maximum(K - ST, 0)
    return np.mean(payoff)

# Output
american_price = american_put_lsm(S0, V0, T, K, r, kappa, theta, omega, rho)
european_price = european_put_price_heston()

print(f"American Put Price (Heston): {american_price:.4f}")
print(f"European Put Price (Heston): {european_price:.4f}")
print(f"Early exercise premium: {american_price - european_price:.4f}")