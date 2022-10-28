# Using Newton's method
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

def CostOLS(X, y, beta):
    return (1.0/n)*np.sum((y-X @ beta)**2)

n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)
a = np.random.rand(3)
y = a[0] + a[1]*x + a[2]*x**2
X = np.c_[np.ones((n,1)), x]
XT_X = X.T @ X
beta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
# Hessian matrix
H = (2.0/n)* XT_X
# Note that here the Hessian does not depend on the parameters beta
invH = np.linalg.pinv(H)
EigValues, EigVectors = np.linalg.eig(H)
print("Own inversion")
print(beta_linreg)
beta = np.random.randn(2,1)
Niterations = 5

# define the gradient
training_gradient = grad(CostOLS, 2)

print(np.shape(beta))
print(np.shape(X))
print(np.shape(y))
print(np.shape(invH))

for iter in range(Niterations):
    gradients = training_gradient(X, y, beta)
    beta -= invH @ gradients
print("beta from own Newton code")
print(beta)