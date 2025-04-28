# finance-related
Algorithms that are all finance related, some use randomly generated numbers from distributions, some use real data (e.g. from Yahoo finance, Bloomberg Terminal). 

1. [Comparison of Euler Schemes under the Heston Model](https://github.com/devinagabriella/finance-related/blob/main/Euler_schemes_Heston_model.py)
2. [Adverse Selection and Price Discovery under Sequential Trading](https://github.com/devinagabriella/finance-related/blob/main/Glosten_Milgrom_simulation.py)
3. [Bisection vs Newton-Rhapson Method](https://github.com/devinagabriella/finance-related/blob/main/Bisection_Newton_Rhapson_updated.py)
4. [Random Number Generators](https://github.com/devinagabriella/finance-related/blob/main/random_num_gen.py)


## Descriptions

### 1. [Comparison of Euler Schemes under the Heston Model](https://github.com/devinagabriella/finance-related/blob/main/Euler_schemes_Heston_model.py)
Compare between Euler discretizations schemes under the Heston stochastic volatility model when the Feller condition is not satisfied. The Euler schemes are the different treatment of the variance function in the variance process of the Heston model when it hits negative.

### 2. [Adverse Selection and Price Discovery under Sequential Trading](https://github.com/devinagabriella/finance-related/blob/main/Glosten_Milgrom_simulation.py)
Simulate random orders based on the Glosten-Milgrom model to analyze market dynamics and price discovery. Using Bayesian belief updating to track dealer's belief of the asset value in a sequential trade environment. Analyze the effect of different amount of informed traders on the speed of price discovery and on the bid-ask spread.

### 3. [Bisection vs Newton-Rhapson Method](https://github.com/devinagabriella/finance-related/blob/main/Bisection_Newton_Rhapson_updated.py)
Compare between bisection and Newton-Rhapson method to estimate implied volatility. The idea is to use [implied volatility surface data](https://github.com/devinagabriella/finance-related/blob/main/implied%20vol%20surface.csv) to find European call option prices using Black-Scholes formula, then estimate back the implied volatility surface using bisection method and Newton-Rhapson method.

### 4. [Random Number Generators](https://github.com/devinagabriella/finance-related/blob/main/random_num_gen.py)
   a. Compute mean and variance of a Brownian motion using Monte Carlo simulation then compare it with the theoretical values.
   
   b. Generate correlated random numbers using Cholesky decomposition and eigenvector (spectral) decomposition.


