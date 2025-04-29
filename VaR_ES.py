"""
This program estimates Value-at-Risk (VaR) and Expected Shortfall (ES) for a portfolio of assets 
using Monte Carlo simulation. It models asset returns under both normal and Student-t distributions 
with varying degrees of freedom to capture different tail behaviors. The program performs the following steps:

1. Initialize portfolio parameters: initial prices, number of shares, volatilities, and distribution settings.
2. Simulate N = 10,000 portfolio losses by generating random log-returns and applying them to asset prices.
3. Sort the simulated loss distributions for each assumed return distribution.
4. Calculate Value-at-Risk (VaR) and Expected Shortfall (ES) across confidence levels from 90% to 99%.
5. Visualize:
    - Loss distributions with fitted normal probability density functions (PDFs).
    - VaR curves as functions of the confidence level.
    - Mean-corrected VaR curves (VaR - mean(loss)).
    - Expected Shortfall (ES) curves as functions of the confidence level.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm

# Initializing data
N = 10000
S = np.array([100, 50, 25, 75, 150])
shares = np.array([1, 3, 5, 2, 4]) 
sigma = np.array([0.001, 0.002, 0.003, 0.0015, 0.0025]) 
nu = np.array([3, 10, 50])
alpha = np.arange(0.9, 1, 0.01) 

# Generate N independent realizations of log returns and apply them to the loss functions

np.random.seed(89)  # For reproducibility

# Normal distribution loss function
def loss_functions_normal():
    loss_list = []
    for _ in range(N):
        losses = 0
        for i in range(len(S)):
            loss = np.sum(-shares[i] * S[i] * (np.exp(np.random.normal(0, sigma[i])) - 1))
            losses += loss
        loss_list.append(losses)
    return loss_list

# Student-t distribution loss function
def loss_functions_t(df):
    loss_list = []
    for _ in range(N):
        losses = 0
        for i in range(len(S)):
            loss = np.sum(-shares[i] * S[i] * (np.exp(stats.t.rvs(df, 0, sigma[i])) - 1))
            losses += loss
        loss_list.append(losses)
    return loss_list

# Calculate and store results
result = {'Normal': np.sort(loss_functions_normal())[::-1]}
for df in nu:
    result[f't-distribution df={df}'] = np.sort(loss_functions_t(df))[::-1]

# Compute VaR and ES
def compute_risk_measures(sorted_losses):
    VaR = []
    ES = []
    for conf_level in alpha:
        VaR_index = int(np.floor((1 - conf_level) * N))
        VaR_value = sorted_losses[VaR_index]
        VaR.append(VaR_value)
        ES_value = np.mean(sorted_losses[VaR_index:])
        ES.append(ES_value)
    return {'VaR': VaR, 'ES': ES}

# Compute and store risk measures for each distribution
risk_measures = {dist: compute_risk_measures(result[dist]) for dist in result}
result.update({'Risk Measures': risk_measures})

# Plot 1: Histograms of Losses for all distributions
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

dist_names = ['t-distribution df=3', 't-distribution df=10', 't-distribution df=50','Normal']
for i, dist in enumerate(dist_names):
    ax = axs[i//2, i%2]
    losses = result[dist]
    ax.hist(losses, bins=50, density=True, alpha=0.6, color='g', edgecolor='black')
    
    # Plot normal probability density function for reference
    mean, std = np.mean(losses), np.std(losses)
    x = np.linspace(min(losses), max(losses), 100)
    ax.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label=f'Normal PDF: Mean={mean:.2f}, Std={std:.2f}')
    
    ax.set_title(f'Loss Distribution - {dist}')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Probability Density')
    ax.legend()

plt.tight_layout()
plt.show()

# Plot 2: VaR vs. Alpha
plt.figure(figsize=(10, 6))
for dist in dist_names:
    plt.plot(alpha, risk_measures[dist]['VaR'], label=f'VaR - {dist}', linestyle="-")
plt.xlabel("Confidence Level (alpha)")
plt.ylabel("Value at Risk (VaR)")
plt.title("VaR vs. Confidence Level (alpha)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Mean-VaR vs. Alpha
plt.figure(figsize=(10, 6))
for dist in dist_names:
    VaR_mean = np.array(risk_measures[dist]['VaR']) - np.mean(result[dist])
    plt.plot(alpha, VaR_mean, label=f'Mean-VaR - {dist}', linestyle="-")
plt.xlabel("Confidence Level (alpha)")
plt.ylabel("Mean-VaR")
plt.title("Mean-VaR vs. Confidence Level (alpha)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 4: ES vs. Alpha
plt.figure(figsize=(10, 6))
for dist in dist_names:
    plt.plot(alpha, risk_measures[dist]['ES'], label=f'ES - {dist}', linestyle="-")
plt.xlabel("Confidence Level (alpha)")
plt.ylabel("Expected Shortfall (ES)")
plt.title("Expected Shortfall (ES) vs. Confidence Level (alpha)")
plt.legend()
plt.grid(True)
plt.show()
