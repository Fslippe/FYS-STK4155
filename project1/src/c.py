from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from functions import *
plt.rcParams.update({"font.size": 16})

def bias_variance_tradeoff(franke=True, x=None, y=None, z=None, n=22, std=0.2, maxdegree=15, n_B=100, plot=True, method="OLS", lamda=1, show=True, seed=200, save=False):
    """
    Calculates the bias variance tradeoff using bootstrap and OLS
    Takes in
    - n:         number of datapoints generated
    - std:       standard deviation of normal distributed noise in z
    - maxdegree: Highest polynomial degree
    - n_B        Number of bootstrap iterations

    Generates a plot showing the tradeoff
    """
    polydegree = np.arange(0, maxdegree+1)
    bias = np.zeros(maxdegree+1)
    variance = np.zeros(maxdegree+1)
    MSE = np.zeros(maxdegree+1)

    if franke:
        x, y, z = make_data(n, std, seed)

    else:
        std = 0

    for i in range(maxdegree+1):  # For increasing complexity
        X = design_matrix(x, y, i)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        z_pred = bootstrap(X_train, X_test, z_train, z_test, n_B, method=method, lamda=lamda)

        bias[i] = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True).T)**2) - std**2
        variance[i] = np.mean((z_pred -  np.mean(z_pred, axis=1, keepdims=True))**2)
        MSE[i] = np.mean(np.mean((z_test - z_pred.T)**2, axis=1, keepdims=True))
    if plot:
        plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        if method == "OLS":
            plt.title(r"$\sigma=$%.1f, $n=$%i" %(std, n))
        else:
            plt.title(r"$\sigma=$%.1f,  $n=$%i,   $\lambda=$%.2e" %(std, n, lamda))
        plt.plot(polydegree, MSE, "-o", label="Error")
        plt.plot(polydegree, bias, "-o", label="bias")
        plt.plot(polydegree, variance, "-o", label="Variance")
        plt.xlabel("Polynomial degree")
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        plt.legend()
        if save != False:
            plt.savefig("../figures/%s.png" %(save), dpi=300, bbox_inches="tight")
        if show:
            plt.show()
    else:
        return MSE, bias, variance

def main():
    n = 30
    std = 0.2
    maxdegree = 15
    bias_variance_tradeoff(n=n, std=std, maxdegree=maxdegree, method="OLS", save="bias_variance_tradeoff")
    n = 100
    std = 0.2
    maxdegree = 15
    bias_variance_tradeoff(n=n, std=std, maxdegree=maxdegree, method="OLS", save="bias_variance_100")
if __name__ == '__main__':
    main()
