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
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

def plot_MSE(MSE, save=False, savename="MSE_degree"):
    """
    Takes in and plotting MSE for chosen label and line type.
    Saving if save=True
    """
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    degree_array = np.arange(1, np.size(MSE)+1, 1)
    plt.plot(degree_array, MSE)
    plt.xlabel("Degree of polynomial")
    plt.ylabel("MSE")

    if save==True:
        plt.savefig("../figures/%s.png" %(savename))
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

    if save==True:
        plt.savefig("../figures/R2_degree.png")
    plt.show()


def plot_beta_degree(beta_degree):
    """
    Plotting how the different betas change by increasing the polynomial degree
    """
    degree_array = np.arange(1, np.size(beta_degree, axis=0)+1, 1)
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(degree_array, beta_degree)
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Values of parameters beta")
    plt.show()

def compare_scale(X, z):
    """
    Comparing MSE and R2 between scaled and non scaled data
    Data is scaled by subtracting mean
    """
    #Splitting data in train and test
    X_train, X_test, z_train, z_test = train_test_split(X, z)

    #Non-scaled prediction
    beta_OLS = OLS(X_train, z_train)
    z_pred_OLS = (X_test @ beta_OLS)
    MSE_OLS = MSE(z_test, z_pred_OLS)
    R2_OLS = R2(z_test, z_pred_OLS)

    #scaled prediction
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    z_train_scaled = z_train - np.mean(z_train)
    X_test_scaled = X_test - np.mean(X_train, axis=0)

    beta_OLS = OLS(X_train_scaled, z_train_scaled)
    z_pred_OLS = (X_test_scaled @ beta_OLS) + np.mean(z_train)
    MSE_OLS_scaled = MSE(z_test, z_pred_OLS)
    R2_OLS_scaled = R2(z_test, z_pred_OLS)

    print("---MSE---")
    print("Non-scaled MSE:", MSE_OLS)
    print("Scaled MSE:", MSE_OLS_scaled)
    print("MSE diff:", abs(MSE_OLS - MSE_OLS_scaled))
    print("\n---R2---")
    print("Non-scaled R2:", R2_OLS)
    print("Scaled R2:", R2_OLS_scaled)
    print("R2 diff:", abs(R2_OLS - R2_OLS_scaled))

def scaled_OLSprediction(X, z):
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
        z_test, z_train, z_pred_OLS, z_pred_train_OLS, beta_OLS[degree-1, :n] = scaled_OLSprediction(X, z)
        mse[degree-1] = MSE(z_test, z_pred_OLS)
        mse_train[degree-1] = MSE(z_train, z_pred_train_OLS)
        r2[degree-1] = R2(z_test, z_pred_OLS)
        r2_train[degree-1] = R2(z_train, z_pred_train_OLS)

    return mse, mse_train, r2, r2_train, beta_OLS

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

def plot_beta_degree(beta):
    """
    plotting betas for different choises of polynomial degrees
    """
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(np.size(beta, axis=0)):
        plt.scatter(np.arange(0, len(beta[i])), beta[i], label="Degree %i" %(i+1))
        plt.plot(np.arange(0, len(beta[i])), beta[i], "--")
    plt.ylabel(r"$\beta_j$")
    plt.xlabel("$j $")
    plt.legend()
    plt.savefig("../figures/beta_degree.png")
    plt.show()


def main():
    degree = 5
    n = 30
    std = 0.2
    x, y, z = make_data(n, std)
    X = design_matrix(x, y, degree)

    #comparing MSE and R2 of scaled and non-scaled data
    compare_scale(X, z)

    #MSE and R2 as functions of poly degree
    maxdegree = 5
    mse, mse_train, r2, r2_train, beta_OLS = MSE_R2(x, y, z, maxdegree)
    plot_beta_degree(beta_OLS)

    plot_R2(r2)
    plot_MSE(mse)

    #Plot of MSE for train and test prediction
    maxdegree = 17
    mse, mse_train, r2, r2_train, beta_OLS = MSE_R2(x, y, z, maxdegree)
    plot_train_test(r2_train, r2, r"$R^2$", save="../figures/R2_train_test.png")
    plot_train_test(mse_train, mse, r"MSE", save="../figures/MSE_train_test.png")

    #Resample of MSE and R2 to get a smooth plot
    print("\n---RESAMPLE---")
    mse, mse_train, r2, r2_train = resample_MSE_R2(n, std, maxdegree, resamples=200)
    plot_train_test(r2_train, r2, r"$R^2$", save="../figures/R2_train_test_resample.png")
    plot_train_test(mse_train, mse, r"MSE", save="../figures/MSE_train_test_resample.png")

    print("Degree with lowest MSE:", np.argmin(mse)+1)
    print("MSE:", np.min(mse))

if __name__ == '__main__':
    main()
