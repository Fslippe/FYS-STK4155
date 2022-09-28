from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

# Make data.
n = int(1e3)
x = np.linspace(0, 1, n+1)
y = np.linspace(0, 1, n+1)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))

    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def design_matrix(degree):
    #Setting up design matrix with dependency on x and y for a chosen degree
    #[x,y,x²,y²,xy,...,x^(degree-1)y,xy^(degree-1)]
    #First finding number of variables (columns) in X
    n_variables = 0
    for i in range(1, degree+1):
        n_variables += i+1
        print(i+1)
    print(n_variables)
    X = np.zeros((n+1, n_variables))

    #Adding columns and their x, y dependency up to a given degree
    idx = 0
    for i in range(degree):
        for j in range(i+1,  degree+2):
            if i+j-1 <= degree:
                if i==j-1 and j-1 !=0:
                    X[:,idx] = x[0,:]**i*y[0,:]**(j-1)
                    print(idx,":", "x:", i, "y:", j-1)
                    idx += 1
                    print()
                elif j-1 !=0:
                    X[:,idx] = x[0,:]**i*y[0,:]**(j-1)
                    print(idx,":", "x:", i, "y:", j-1)

                    idx += 1
                    X[:,idx] = x[0,:]**(j-1)*y[0,:]**i
                    print(idx,":", "x:", j-1, "y:", i)

                    idx += 1
    return X


X = design_matrix(20)
