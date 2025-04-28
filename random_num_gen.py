import numpy as np
import matplotlib.pyplot as plt

"""Random Number Generation
1. Using the simulation method to calculate following expectation and variance and compare them with the theoretical values.
2. Test if a correlation matrices are semi definite positive and apply the Cholesky decomposition and Eigenvector decomposition to generate random numbers."""

#1. Using the simulation method to calculate following expectation and variance and compare them with the theoretical values.
#set parameters
n = 100
N=10000
T = 2
delta = T/n
X = np.zeros(N)
t = np.linspace(delta,T,n)
for i in range(N):
    W_t = np.random.normal(0,np.sqrt(t)) #Brownian motion
    dW_t = np.random.normal(0, np.sqrt(delta),size=n) #increment of Brownian motion
    #store different realization of the process for each Monte Carlo simulation
    X[i]= np.sum(W_t*dW_t)

#calculate the expectation and variance
E = np.mean(X)
V = np.var(X)
print('Expectation:', E)
print('Variance:', V)

#plot the histogram of simulated stochastic integral with T=2, n=100
plt.hist(X, bins=30, density=True)
plt.title('Histogram of simulated stochastic integral with T=2, n=100')
plt.xlabel('Integral Value')
plt.ylabel('Density')
plt.show()

#2. Test if a correlation matrices are semi definite positive and apply the Cholesky decomposition and Eigenvector decomposition to generate random numbers.
C = np.array([[1,0.5,0.9],[0.5,1,1],[0.9,1,1]])
D = np.array([[1,0.6,0.3],[0.6,1,0.5],[0.3,0.5,1]])

#check if the correlation matrices are semi definite positive
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

print('C is semi definite positive:', is_pos_def(C))
print('D is semi definite positive:', is_pos_def(D))

#apply Cholesky decomposition to generate random number manually
#Cholesky decomposition to get lower triangular matrix
L = np.linalg.cholesky(D)
#Spectral decomposition to get the square root of the eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(D)
#We know that D = eigenvectors * diag(eigenvalues) * eigenvectors^T 
#such that V = eigenvectors * diag(sqrt(eigenvalues)) and V*V^T = D
V = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
#generate standard normal random number
Z = np.random.normal(0,1,(3,10))
#Use cholesky decomposition to get correlated random number
R1 = np.dot(L,Z)
R2 = np.dot(V,Z)

print('Correlated random number using Cholesky decomposition:', R1)
print('Correlated random number using Spectral decomposition:', R2)

