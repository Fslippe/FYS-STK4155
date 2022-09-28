from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.utils import resample
from random import random, seed


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def create_X(x, y, n):

    if len(x.shape)>1:
        x = np.ravel(x)    #flatten x
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X


def OLS(X, z):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    A = np.linalg.pinv(X.T @ X)
    beta = A @ X.T @ z
    return beta


def mean_scaler(Xtrain,Xtest,ztrain,ztest):
    """
    Takes in design matrix and dataset and returns test and train data
    scaled by subtracting the mean
    """
    X_mean = np.mean(Xtrain, axis=0)
    X_train = Xtrain - X_mean
    X_test = Xtest - X_mean

    z_mean = np.mean(ztrain, axis=0)
    z_train = ztrain - z_mean
    z_test = ztest - z_mean

    #z_test = z_test - np.mean(z_test, axis=0)#.reshape(int((n+1)*0.8), int((n+1)*0.8))
    return X_train,X_test,z_train,z_test


def features(matrix1,matrix2,degree,):
    l = int((degree + 1) * (degree + 2) / 2)

    matrix1 = matrix1[:,:l]
    matrix2 = matrix2[:,:l]


    return matrix1,matrix2


def MSE(y,ypred):
    return np.mean((y-ypred)**2)

def R2(y,ypred):
    return 1 - np.sum((y - ypred)**2) / np.sum((y - np.mean(y)) ** 2)


def bootstrap_OLS(Xtrain,Xtest,ztrain,ztest,nbootstraps):
    z_pred = np.empty((len(ztest), nbootstraps))
    for i in range(nbootstraps):
        X_, z_ = resample(Xtrain, ztrain)
        beta = OLS(X_, z_)
        z_pred[:, i] = (Xtest@beta).ravel()

    return z_pred
