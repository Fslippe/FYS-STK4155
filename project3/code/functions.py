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
from sklearn.utils import resample, shuffle
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
    return np.sum(np.where(pred == y_test.ravel(), 1, 0)) / len(y_test)


def bootstrap(X_train, X_test, y_train, y_test, model, n_B):
    #score = np.zeros(n_B)

    for i in range(n_B):
        X_, y_ = resample(X_train, y_train)
        model.fit(X_, y_, epochs=100, batch_size=32, verbose=0)
        #score[i] = model.evaluate(X_test, y_test)[1]

    return model  # score


def R2(data, model):
    """
    takes in actual data and modelled data to find
    R2 score
    """
    R2 = r2_score(data.ravel(), model.ravel())
    return R2
