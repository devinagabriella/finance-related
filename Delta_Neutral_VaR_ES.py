"""
This program estimates Value-at-Risk (VaR), Mean-VaR  and Expected Shortfall (ES) for a hypothetical portfolio 
under both Normal and Student-t loss distributions. The portfolio has long position on a call option,
and position in the underlying stock, making the portfolio delta neutral. 
This program also compares three risk estimation approaches:
Monte Carlo full revaluation, linearized loss approximation, and the variance-covariance method.
Results are visualized to analyze distributional assumptions and method accuracy.
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display

# Global variables
St = 100  # Initial stock price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
T = 0.5  # Time to maturity
t = 0  # Current time
K = 100  # Strike price
dt = 1 / 252  # Time step (1 day)
confidence_interval = [0.95, 0.99]  # Confidence levels for VaR and ES
covariance_matrix = [
    [0.001 ** 2, -0.5 * 0.001 * 0.0001],
    [-0.5 * 0.001 * 0.0001, 0.0001 ** 2],
]  # Covariance matrix for the risk factors
risk_factor_mean = [0, 0]  # Mean of the risk factors
std = [0.001, 0.0001]  # Standard deviations of the risk factors
N = 10000  # Number of Monte Carlo simulations
N_stock = -norm.cdf((np.log(St / K) + (r + 1 / 2 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t)))  # Hedge ratio (Delta)


# %%
# Calculate Greek letters for the option
def calc_greeks():
    global theta, delta, rho, vega
    d1 = (np.log(St / K) + (r + 1 / 2 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    # Theta: rate of change of the option price with respect to time
    theta = -St * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    # Delta: rate of change of the option price with respect to the underlying asset's price
    delta = norm.cdf(d1)
    # Rho: rate of change of the option price with respect to the risk-free interest rate
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    # Vega: rate of change of the option price with respect to volatility
    vega = St * norm.pdf(d1) * np.sqrt(T)

calc_greeks()

# %%
# Function to calculate the price of a call option using the Black-Scholes formula
def pricing_call(t, T, K, S, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    value = S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    return value


# %%
# Monte Carlo simulation for full revaluation method
def monte_carlo_full_revaluation():
    np.random.seed(89)  # Set random seed for reproducibility
    losses = []
    for _ in range(N):
        # Initial portfolio value
        Vt_stock = St * N_stock
        Vt_option = pricing_call(t, T, K, St, r, sigma)
        Vt = Vt_stock + Vt_option

        # Simulate new risk factors
        simulated_risk_factors = np.random.multivariate_normal(risk_factor_mean, covariance_matrix)
        St_delta = St * np.exp(simulated_risk_factors[0])
        sigma_delta = simulated_risk_factors[1] + sigma

        # Revalued portfolio under new risk factors
        Vt_delta_stock = N_stock * St_delta
        Vt_delta_option = pricing_call(dt, T, K, St_delta, r, sigma_delta)
        Vt_delta = Vt_delta_stock + Vt_delta_option

        # Calculate loss
        loss = -(Vt_delta - Vt)
        losses.append(loss)

    return losses


# %%
# Monte Carlo simulation for linearized loss method
def monte_carlo_linearized_loss():
    np.random.seed(89)  # Set random seed for reproducibility
    losses = []
    for _ in range(N):
        # Simulate new risk factors
        simulated_risk_factors = np.random.multivariate_normal(risk_factor_mean, covariance_matrix)
        # Calculate loss using linear approximation (theta and vega)
        loss = -(theta * dt + vega * simulated_risk_factors[1])
        losses.append(loss)

    return losses

# %%
# Variance-covariance method for risk measurement
def variance_covariance_method():
    c = theta * dt
    b = [0, vega]

    # Calculate mean and variance of the loss distribution
    mean = -c - np.dot(b, risk_factor_mean)
    variance = np.dot(b, np.dot(covariance_matrix, b))

    # Calculate VaR and ES for given confidence intervals
    VaR = [mean + np.sqrt(variance) * norm.ppf(alpha_i) for alpha_i in confidence_interval]
    VaR_mean = [np.sqrt(variance) * norm.ppf(alpha_i) for alpha_i in confidence_interval]
    ES = [mean + np.sqrt(variance) * norm.pdf(norm.ppf(alpha_i)) / (1 - alpha_i) for alpha_i in confidence_interval]

    return {'VaR_alpha': VaR, 'Var_alpha_mean': VaR_mean, 'Expected_shortfalls': ES}


# %%
# Calculate risk measures (VaR and ES) from losses
def calculate_risk_measures(losses):
    sorted_losses = np.sort(losses)[::-1]  # Sort losses in descending order
    indices = [int(np.floor((1 - alpha_i) * N)) for alpha_i in confidence_interval]  # Calculate indices for VaR
    VaR_alpha = sorted_losses[indices]  # Get VaR values
    Var_alpha_mean = [VaR_alpha_i - np.mean(losses) for VaR_alpha_i in VaR_alpha]  # Calculate VaR mean
    ES = [np.mean([x for x in losses if x > VaR_alpha_i]) for VaR_alpha_i in VaR_alpha]  # Calculate ES

    return {'VaR_alpha': VaR_alpha, 'Var_alpha_mean': Var_alpha_mean, 'Expected_shortfalls': ES}


# %%
# Compute results for all methods
def compute_results():
    results = {}
    methods = ['monte_carlo_full_revaluation', 'monte_carlo_linearized_loss', 'variance_covariance_method']
    functions = [monte_carlo_full_revaluation, monte_carlo_linearized_loss, variance_covariance_method]

    for method, function in zip(methods, functions):
        if method != 'variance_covariance_method':
            losses = function()
            risk_measures = calculate_risk_measures(losses)
            results[method] = {'Realized_LF': losses, 'risk_measures': risk_measures}
        else:
            risk_measures = function()
            results[method] = {'Realized_LF': np.nan, 'risk_measures': risk_measures}

    return results

# %%
# Plot loss functions
def plot_loss_functions(results):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), tight_layout=True)
    titles = ['Monte Carlo Full Revaluation', 'Monte Carlo on Linearized Loss ']
    data = [value['Realized_LF'] for key, value in results.items() if 'Realized_LF' in value]

    for ax, dataset, title in zip(axs.ravel(), data, titles):
        ax.hist(dataset, bins=50, alpha=0.6, color='g', edgecolor='black', density=True)
        ax.set_title(title)
        ax.set_xlabel('Loss')
        ax.set_ylabel('Frequency')

    plt.show()

# %%
# Run the computations and plot the results
results = compute_results()
plot_loss_functions(results)

# %%
# Display risk measures in a tabular format
df1 = pd.DataFrame(results['monte_carlo_full_revaluation']['risk_measures'], index=[0.95, 0.99])
df2 = pd.DataFrame(results['monte_carlo_linearized_loss']['risk_measures'], index=[0.95, 0.99])
df3 = pd.DataFrame(results['variance_covariance_method']['risk_measures'], index=[0.95, 0.99])

display(df1, df2, df3)

# Extracting the risk measures for each method
VaR_full_revaluation = results['monte_carlo_full_revaluation']['risk_measures']['VaR_alpha']
VaR_linearized_loss = results['monte_carlo_linearized_loss']['risk_measures']['VaR_alpha']
VaR_variance_covariance = results['variance_covariance_method']['risk_measures']['VaR_alpha']

mean_VaR_full_revaluation = results['monte_carlo_full_revaluation']['risk_measures']['Var_alpha_mean']
mean_VaR_linearized_loss = results['monte_carlo_linearized_loss']['risk_measures']['Var_alpha_mean']
mean_VaR_variance_covariance = results['variance_covariance_method']['risk_measures']['Var_alpha_mean']

ES_full_revaluation = results['monte_carlo_full_revaluation']['risk_measures']['Expected_shortfalls']
ES_linearized_loss = results['monte_carlo_linearized_loss']['risk_measures']['Expected_shortfalls']
ES_variance_covariance = results['variance_covariance_method']['risk_measures']['Expected_shortfalls']

# Labels for the x-axis
confidence_levels = [0.95, 0.99]

# Plotting VaR comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(confidence_levels, VaR_full_revaluation, 'o-', label='Full Revaluation')
plt.plot(confidence_levels, VaR_linearized_loss, 's-', label='Linearized Loss')
plt.plot(confidence_levels, VaR_variance_covariance, 'd-', label='Variance-Covariance')
plt.xlabel('Confidence Level')
plt.ylabel('VaR')
plt.title('Comparison of VaR')
plt.legend()

# Plotting mean-VaR comparison
plt.subplot(1, 3, 2)
plt.plot(confidence_levels, mean_VaR_full_revaluation, 'o-', label='Full Revaluation')
plt.plot(confidence_levels, mean_VaR_linearized_loss, 's-', label='Linearized Loss')
plt.plot(confidence_levels, mean_VaR_variance_covariance, 'd-', label='Variance-Covariance')
plt.xlabel('Confidence Level')
plt.ylabel('mean-VaR')
plt.title('Comparison of mean-VaR')
plt.legend()

# Plotting ES comparison
plt.subplot(1, 3, 3)
plt.plot(confidence_levels, ES_full_revaluation, 'o-', label='Full Revaluation')
plt.plot(confidence_levels, ES_linearized_loss, 's-', label='Linearized Loss')
plt.plot(confidence_levels, ES_variance_covariance, 'd-', label='Variance-Covariance')
plt.xlabel('Confidence Level')
plt.ylabel('Expected Shortfall (ES)')
plt.title('Comparison of Expected Shortfall (ES)')
plt.legend()

plt.tight_layout()
plt.show()