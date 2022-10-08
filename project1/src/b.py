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

def plot_train_test(train, test, ylabel, save=False):
    """
    Comparing parameter found using test and train data
    Can be used for MSE and R2
    """
    degree_array = np.arange(1, np.size(train)+1)
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(degree_array, train, label="train sample")
    plt.plot(degree_array, test, label="test sample")
    plt.legend()
    plt.xlabel("Degree of polynomial")
    plt.ylabel(ylabel)
    if save != False:
        print("Saving")
        plt.savefig("../figures/%s.png" %(save), dpi=300, bbox_inches='tight')
    plt.show()

def plot_MSE(MSE, save=False):
    """
    Takes in and plotting MSE for chosen label and line type.
    Saving if save=True
    """
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    degree_array = np.arange(1, np.size(MSE)+1, 1)
    plt.plot(degree_array, MSE)
    plt.xlabel("Degree of polynomial")
    plt.ylabel("MSE")

    if save != False:
        print("Saving")
        plt.savefig("../figures/%s.png" %(save), dpi=300, bbox_inches="tight")
    plt.show()


def plot_R2(R2, save=False):
    """
    Takes in and plotting R2 for chosen label and line type.
    Saving if save=True
    """
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    degree_array = np.arange(1, np.size(R2)+1, 1)
    plt.plot(degree_array, R2)
    plt.xlabel("Degree of polynomial")
    plt.ylabel("$R^2$")

    if save != False:
        print("Saving")
        plt.savefig("../figures/%s.png" %(save), dpi=300, bbox_inches="tight")
    plt.show()


def compare_scale(X, z):
    """
    Comparing MSE and R2 between scaled and non scaled data
    Data is scaled by subtracting mean
    """
    x, y, z = make_data(30, 0.2, seed=200)

    MSE_OLS = np.zeros(11)
    MSE_OLS_scaled = np.zeros(11)
    R2_OLS_scaled = np.zeros(11)
    R2_OLS = np.zeros(11)

    for degree in range(11):
        X = design_matrix(x, y, degree)
        #Splitting data in train and test
        X_train, X_test, z_train, z_test = train_test_split(X, z)

        #Non-scaled prediction
        beta_OLS = OLS(X_train, z_train)
        z_pred_OLS = (X_test @ beta_OLS)
        MSE_OLS[degree] = MSE(z_test, z_pred_OLS)
        R2_OLS[degree] = R2(z_test, z_pred_OLS)

        #scaled prediction
        X_train_scaled = X_train - np.mean(X_train, axis=0)
        z_train_scaled = z_train - np.mean(z_train)
        X_test_scaled = X_test - np.mean(X_train, axis=0)

        beta_OLS = OLS(X_train_scaled, z_train_scaled)
        z_pred_OLS = (X_test_scaled @ beta_OLS) + np.mean(z_train)
        MSE_OLS_scaled[degree] = MSE(z_test, z_pred_OLS)
        R2_OLS_scaled[degree] = R2(z_test, z_pred_OLS)

    plt.plot(np.arange(0,11), abs(MSE_OLS - MSE_OLS_scaled))
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"$abs(MSE - MSE_{scaled})$")
    plt.yscale("log")
    plt.savefig("../figures/compare_scale_mse.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.plot(np.arange(0,11), abs(R2_OLS - R2_OLS_scaled))
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"$abs(R^2 - R^2_{scaled})$")
    plt.yscale("log")

    plt.savefig("../figures/compare_scale_r2.png", dpi=300, bbox_inches="tight")
    plt.show()


def scaled_OLSprediction(X, z, var_beta=False):
    """
    Takes in design matrix X and data z
    Performs a split in train and test data
    Scales the data by subtracting the mean
    performs OLS and finding z using both train and test data
    """
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    z_train_scaled = z_train - np.mean(z_train)
    X_test_scaled = X_test - np.mean(X_train, axis=0)

    beta_OLS = OLS(X_train_scaled, z_train_scaled)
    z_pred_OLS = (X_test_scaled @ beta_OLS) + np.mean(z_train)
    z_pred_train_OLS = (X_train_scaled @ beta_OLS) + np.mean(z_train)

    if var_beta:
        var_test = np.diag(np.linalg.pinv(X_test.T @ X_test))*0.2**2
        var_train = np.diag(np.linalg.pinv(X_train.T @ X_train))*0.2**2
        return z_test, z_train, z_pred_OLS, z_pred_train_OLS, beta_OLS, var_test, var_train
    else:
        return z_test, z_train, z_pred_OLS, z_pred_train_OLS, beta_OLS


def MSE_R2(x, y, z, maxdegree):
    """
    calculates MSE, R2 and beta for mean scaled data for all degrees 1,..maxdegree
    Takes in
    - dataset x, y, z
    - maxdegree: highest degree to calculate MSE and R2
    returns MSE and R2 arrays of length maxdegree
    """
    mse = np.zeros(maxdegree)
    r2 = np.zeros(maxdegree)
    mse_train = np.zeros(maxdegree)
    r2_train = np.zeros(maxdegree)
    beta_n = int((maxdegree+1)*(maxdegree+2)/2)
    beta_OLS = np.zeros((maxdegree, beta_n))

    for degree in range(1, maxdegree+1):
        n = int((degree+1)*(degree+2)/2)
        X = design_matrix(x, y, degree)
        if degree == 5:
            z_test, z_train, z_pred_OLS, z_pred_train_OLS, beta_OLS[degree-1, :n], var_test, var_train = scaled_OLSprediction(X, z, var_beta=True)
        else:
            z_test, z_train, z_pred_OLS, z_pred_train_OLS, beta_OLS[degree-1, :n] = scaled_OLSprediction(X, z)

        mse[degree-1] = MSE(z_test, z_pred_OLS)
        mse_train[degree-1] = MSE(z_train, z_pred_train_OLS)
        r2[degree-1] = R2(z_test, z_pred_OLS)
        r2_train[degree-1] = R2(z_train, z_pred_train_OLS)

    return mse, mse_train, r2, r2_train, beta_OLS, var_test, var_train

def resample_MSE_R2(n, std, maxdegree, resamples):
    """
    Using function to get a smooth curve of MSE and R2 for different degrees
    """

    mse_re = np.zeros((resamples, maxdegree))
    r2_re = np.zeros((resamples, maxdegree))
    mse_train_re = np.zeros((resamples, maxdegree))
    r2_train_re = np.zeros((resamples, maxdegree))
    for i in range(resamples):
        x, y, z = make_data(n, std, seed = np.random.randint(10000))
        mse_re[i,:], mse_train_re[i,:], r2_re[i,:], r2_train_re[i,:], beta = MSE_R2(x, y, z, maxdegree)

    mse_re = np.mean(mse_re, axis=0)
    r2_re = np.mean(r2_re, axis=0)
    mse_train_re = np.mean(mse_train_re, axis=0)
    r2_train_re = np.mean(r2_train_re, axis=0)

    return mse_re, mse_train_re, r2_re, r2_train_re

def plot_beta_degree(beta, save=False):
    """
    plotting betas for different choises of polynomial degrees
    """
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(np.size(beta, axis=0)):
        plt.scatter(np.arange(0, len(beta[i])), beta[i], label="Degree %i" %(i+1))
        plt.plot(np.arange(0, len(beta[i])), beta[i], "--")
    plt.ylabel(r"$\beta_j$")
    plt.xlabel("$j $")
    plt.legend(loc="upper right")
    if save != False:
        print("Saving")
        plt.savefig("../figures/%s.png" %(save), dpi=300, bbox_inches="tight")
    plt.show()


def main():
    degree = 5
    n = 30
    std = 0.2
    x, y, z = make_data(n, std, seed=200)
    X = design_matrix(x, y, degree)
    #comparing MSE and R2 of scaled and non-scaled data
    compare_scale(X, z)

    #MSE and R2 as functions of poly degree
    maxdegree = 5
    mse, mse_train, r2, r2_train, beta_OLS, var_test, var_train = MSE_R2(x, y, z, maxdegree)
    plot_beta_degree(beta_OLS, save="beta_degree")
    for i in range(len(var_test)):
        print(r"$\beta_{%i}$ & %.2f & %.2f \\" %(i,( var_test[i]), (var_train[i])))


    plot_R2(r2, save="r2_ols_5")
    plot_MSE(mse, save="mse_ols_5")

    #Plot of MSE for train and test prediction
    maxdegree = 17
    mse, mse_train, r2, r2_train, beta_OLS = MSE_R2(x, y, z, maxdegree)
    plot_train_test(r2_train, r2, r"$R^2$", save="R2_train_test")
    plot_train_test(mse_train, mse, r"MSE", save="MSE_train_test")

    #Resample of MSE and R2 to get a smooth plot
    print("\n---RESAMPLE---")
    mse, mse_train, r2, r2_train = resample_MSE_R2(n, std, maxdegree, resamples=200)
    plot_train_test(r2_train, r2, r"$R^2$", save="R2_train_test_resample")
    plot_train_test(mse_train, mse, r"MSE", save="MSE_train_test_resample")

    print("Degree with lowest MSE:", np.argmin(mse)+1)
    print("MSE:", np.min(mse))

if __name__ == '__main__':
    main()
