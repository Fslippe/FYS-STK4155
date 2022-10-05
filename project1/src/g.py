
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
from functions import *
from b import *
from f import *
from e import bias_variance_lamda

def standard_scale(X, z):
    """
    Performs a split in train and test data
    and scales by subtracting the mean
    and dividing by the standard deviation
    takes in:
    - X: Design matrix
    - z: Dataset
    returns:
    - X_train
    - X_test
    - z_train
    - z_test
    """
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    X_train_mean = np.mean(X_train, axis=0)
    z_train_mean = np.mean(z_train)
    X_train_std = np.std(X_train)
    z_train_std = np.std(z_train)
    X_train = (X_train - X_train_mean)/X_train_std
    X_test = (X_test - X_train_mean)/X_train_std
    z_train = (z_train - z_train_mean)/z_train_std
    z_test = (z_test - z_train_mean)/z_train_std

    return X_train, X_test, z_train, z_test

def MSE_R2_2(x, y, z, maxdegree, scaler, method):
    """
    calculates MSE, R2 and beta for mean scaled data for all degrees 1,..maxdegree
    Takes in
    - dataset x, y, z
    - maxdegree: highest degree to calculate MSE and R2
    returns MSE and R2 arrays of length maxdegree
    """
    mse = np.zeros(maxdegree)
    r2 = np.zeros(maxdegree)

    for degree in range(1, maxdegree+1):
        X = design_matrix(x, y, degree)
        if scaler == "STANDARD":
            X_train, X_test, z_train, z_test = standard_scale(X, z)

        if method == "OLS":
            beta = OLS(X_train, z_train)
            z_pred = X_test @ beta

        mse[degree-1] = MSE(z_test, z_pred)
        r2[degree-1] = R2(z_test, z_pred)

    return mse, r2

# Load the terrain
def main():
    terrain = imread('../data/SRTM_data_Norway_1.tif')
    print(np.shape(terrain))
    n = 20
    n_skip = 2
    std = 0
    maxdegree = 25
    terrain = terrain[100:n*n_skip+1+100,100:n*n_skip+1+100][1::n_skip]
    np.random.seed(200)
    noise = 0 #np.random.normal(0, std, size=(n+1,n+1))
    x, y, z = make_data(n*n_skip, std)
    x = x[1::n_skip]
    y = y[1::n_skip]
    z = (terrain + noise*np.mean(terrain)).ravel()
    mean_scale = np.mean(z)
    std_scale = np.std(z)
    z_scaled = (z - mean_scale)/std_scale

    #mse, r2 = MSE_R2_2(x, y, z, maxdegree, "STANDARD", "OLS")
    #plot_MSE(mse)
    #plot_R2(r2)

    run_best_lambda = True

    if run_best_lambda:
        lmb_ols, deg_ols, mse_ols = lamda_degree_MSE(x, y, z_scaled, "OLS", std, n_lmb = 1, maxdegree = maxdegree, k_folds = 5, max_iter = 100, save=False)
        lmb_ridge, deg_ridge, mse_ridge = lamda_degree_MSE(x, y, z_scaled, "RIDGE", std, n_lmb = 20, maxdegree = maxdegree, k_folds = 5, max_iter = 100, save=False, lmb_min=-13, lmb_max=-3)
        lmb_lasso, deg_lasso, mse_lasso = lamda_degree_MSE(x, y, z_scaled, "LASSO", std, n_lmb = 20, maxdegree = maxdegree, k_folds = 5, max_iter = 100, save=False, lmb_min=-13, lmb_max=-3)
        print("OLS:", lmb_ols, deg_ols, mse_ols)
        print("RIDGE:", lmb_ridge, deg_ridge, mse_ridge)
        print("LASSO:", lmb_lasso, deg_lasso, mse_lasso)

    else:

        deg_ols = 19
        mse_ols = 0.010703657377970114

        lmb_ridge = 3.792690190732254e-12
        deg_ridge = 20
        mse_ridge = 0.009997042109663854

        lmb_lasso = 1.6237767391887177e-09
        deg_lasso = 22
        mse_lasso = 0.11060598506946814

    compare_3d(x, y, z_scaled, 0, deg_ols, lmb_ridge, deg_ridge, lmb_lasso, deg_lasso, name_add="test_2", std=std_scale, mean=mean_scale)

    lamda = np.logspace(2, 2, 1)
    n_B = 20
    bias_variance_lamda(n, std, maxdegree, n_B, "RIDGE", lamda, "real", False, x, y, z_scaled)
    plt.show()
    bias_variance_lamda(n, std, maxdegree, n_B, "LASSO", lamda, "real", False, x, y, z_scaled)
    plt.show()



if __name__ == '__main__':
    main()
