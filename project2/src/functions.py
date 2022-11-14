import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
import autograd.numpy as np
import pandas as pd
from matplotlib import cm
from autograd import elementwise_grad
from autograd import grad
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample, shuffle
from sklearn.linear_model import Lasso, Ridge
import time
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.rcParams.update({"font.size": 11})


def OLS(X, z):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z

    return beta

def ridge_regression(X, z, lamda):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    N = X.shape[1]
    beta = np.linalg.pinv(X.T @ X + lamda*np.eye(N)) @ X.T @ z

    return beta

def MSE(data, model):
    """
    takes in actual data and modelled data to find
    Mean Squared Error
    """
    MSE = mean_squared_error(data.ravel(), model.ravel())
    return MSE

def accuracy(y_test, pred):
    """Accuracy score for binary prediction"""
    return np.sum(np.where(pred==y_test.ravel(), 1, 0)) / len(y_test)

def R2(data, model):
    """
    takes in actual data and modelled data to find
    R2 score
    """
    R2 = r2_score(data.ravel(), model.ravel())
    return R2


def design_matrix_1D(x, degree):
    """
    one dimensional design matrix for x and input degree
    Retruns
    design matrix X
    """
    N = len(x)
    X = np.ones((N, degree+1))

    for i in range(1, degree+1):
        X[:,i] = x**i

    return X

def make_data(n, noise_std, seed=1, terrain=False):
    """
    Make data z=f(x)+noise for n steps and normal distributed
    noise with standard deviation equal to noise_std
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    x, y = np.meshgrid(x, y)

    noise = np.random.normal(0, noise_std, size=(n+1,n+1))
    z = FrankeFunction(x, y) + noise
    return x, y, z.ravel()

def FrankeFunction(x, y):
    """
    Generate the franke function for input x and y
    returns function value
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def design_matrix(x, y, degree):
    """
    Setting up design matrix with dependency on x and y for a chosen degree
    [x,y,xy,x²,y²,...]
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree+1)*(degree+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X


def test_func_1D(x, degree, noise=0):
    """
    A one dimensional test function 
    takes in:
    - x some input values
    - degree of 1d function 
    - noise to add to function
    returns:
    - function value
    """
    np.random.seed(100)
    a = np.random.rand(degree + 1)
    f_x = 0
    for i in range(degree + 1):
        f_x += a[i]*x**i

    return f_x + noise


def plot_3d_trisurf(x, y, z, scale_std=1, scale_mean=0, savename=None, azim=110, title=""):
    """
    3D plot of Franke function prediction
    - x                 input 1D x-values
    - y                 input 1D y-values
    - z                 input 1D z-values
    - scale_std=1       old std of z if standard scaled
    - scale_mean=0      old mean of z if standard scaled 
    - savename=None     savename of plot 
    - azim=110          direction of plot
    - title=""          title of plot
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(title)
    surf = ax.plot_trisurf(x, y, z*scale_std + scale_mean, cmap=cm.coolwarm, linewidth=0.2, antialiased=False)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylabel(r"$y$")
    ax.view_init(azim=azim)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    if savename != None:
        plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')


