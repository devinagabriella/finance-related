"""
This program simulates how adverse-selection risk in order flow influences bid-ask spreads 
and the pace of price discovery in dealer markets. Adopting the Glosten-Milgrom sequential-trade framework, 
it models 100 trades where a competitive dealer updates beliefs about the asset's 
true value, adjusts quotes, and progressively reduces pricing errors. 
The simulation also explores how the presence of informed traders 
affects the speed of learning and the width of bid-ask spreads.
"""

import numpy as np
import matplotlib.pyplot as plt

# Model parameters
V_h = 102
V_l = 98
pi = 0.3
theta_0 = 0.5
v_true = V_h   # True value of the asset
n_orders = 100

# Pre-computed values
delta_V = V_h - V_l

# Probability that any order is a buy (π + (1 - π)/2 = 0.65)
pr_buy = pi + (1 - pi) / 2

# Storage
theta_t = [theta_0]
mu_t = [theta_0 * V_h + (1 - theta_0) * V_l]
p_t = []
order_history = []

# Generate 100 orders
orders = np.random.choice(["buy", "sell"], size=n_orders, p=[pr_buy, 1 - pr_buy])

# Simulation loop
theta = theta_0

for i, order in enumerate(orders):
    mu = theta * V_h + (1 - theta) * V_l

    # Ask spread s^a_t
    numerator_a = pi * theta * (1 - theta)
    denominator_a = pi * theta + (1 - pi) / 2
    s_ask = numerator_a / denominator_a * delta_V

    # Bid spread s^b_t
    numerator_b = pi * theta * (1 - theta)
    denominator_b = pi * (1 - theta) + (1 - pi) / 2
    s_bid = numerator_b / denominator_b * delta_V

    # Transaction price
    if order == "buy":
        pt = mu + s_ask
        # Bayes update for a buy
        num = theta * (pi + (1 - pi) / 2)
        denom = theta * (pi + (1 - pi) / 2) + (1 - theta) * ((1 - pi) / 2)
        theta = num / denom
    else:
        pt = mu - s_bid
        # Bayes update for a sell
        num = theta * ((1 - pi) / 2)
        denom = theta * ((1 - pi) / 2) + (1 - theta) * (pi + (1 - pi) / 2)
        theta = num / denom

    # Save values
    theta_t.append(theta)
    mu_t.append(theta * V_h + (1 - theta) * V_l)
    p_t.append(pt)
    order_history.append(order)

# Plotting beliefs and prices
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(theta_t[:-1], label="Belief θₜ")
plt.title("Evolution of Dealer's Beliefs")
plt.xlabel("Order #")
plt.ylabel("θₜ")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(p_t, label="Transaction Price pₜ")
plt.title("Transaction Prices Over Time")
plt.xlabel("Order #")
plt.ylabel("pₜ")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

#-----------------------------------------------------------

mu_0 = theta_0*V_h + (1-theta_0) *V_l #initial dealer's estimate of the value
pr_buy = pi/2 + (1-pi)/2
pr_sell = 1-pr_buy

p_t = []
mu_t = [mu_0]
order_history = []
theta_t = [theta_0]

# generate random 0 1 based on the prob of buy and sell
# 0 for buy 1 for sell
# 100 random numbers
# calculate the beliefs and the price at each t
# repeat for 10 times

# Generate 100 orders
orders = np.random.choice(["buy", "sell"], size=n_orders, p=[pr_buy, 1 - pr_buy])

#print(orders[:5])
for i,order in enumerate(orders):
    if order == 'buy':
        #calculate p_t
        a_t_numerator = pi*theta_t[i]*(1-theta_t[i])*(V_h-V_l)
        a_t_denominator = pi*theta_t[i]+((1-pi)/2)
        p_t.append(mu_t[i] + (a_t_numerator/a_t_denominator))

        #update belief for next value (theta)
        theta_numer = (1+pi)*theta_t[i]/2
        theta_denom = (pi*theta_t[i]) + ((1-pi)/2)
        theta_t.append(theta_numer/theta_denom)
    
    elif order == 'sell':
        #calculate p_t
        b_t_numerator = pi*theta_t[i]*(1-theta_t[i])*(V_h-V_l)
        b_t_denominator = pi*(1-theta_t[i])+((1-pi)/2)
        p_t.append(mu_t[i] - (b_t_numerator/b_t_denominator))

        #update belief for next value (theta)
        theta_numer = (1-pi)*theta_t[i]/2
        theta_denom = (pi*(1-theta_t[i])) + ((1-pi)/2)
        theta_t.append(theta_numer/theta_denom)
    
    #Update value estimate (mu) and order history
    mu_t.append(theta_t[i+1]*V_h + (1-theta_t[i+1])*V_l)
    order_history.append(order)

print(theta_t[:5], p_t[:5])

PD = [(x - V_h)**2 for x in p_t]

print(PD[:5])

# Plotting beliefs and prices
plt.figure(figsize=(12, 5))
#plt.subplot(1, 2, 1)
plt.plot(theta_t[:-1], label="Belief θₜ")
plt.title("Evolution of Dealer's Beliefs")
plt.xlabel("Order #")
plt.ylabel("θₜ")
plt.grid()
plt.legend()

#plt.subplot(1, 2, 2)
plt.plot(p_t, label="Transaction Price pₜ")
plt.title("Transaction Prices Over Time")
plt.xlabel("Order #")
plt.ylabel("pₜ")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()