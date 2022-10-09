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

def MSE_R2_standard_scale(x, y, z, maxdegree):
    """
    calculates MSE, R2 and beta using OLS for mean scaled data for all degrees 1,..maxdegree
    Takes in
    - dataset x, y, z
    - maxdegree: highest degree to calculate MSE and R2
    returns MSE and R2 arrays of length maxdegree
    """
    mse = np.zeros(maxdegree)
    r2 = np.zeros(maxdegree)

    for degree in range(1, maxdegree+1):
        X = design_matrix(x, y, degree)
        X_train, X_test, z_train, z_test = standard_scale(X, z)

        beta = OLS(X_train, z_train)
        z_pred = X_test @ beta

        mse[degree-1] = MSE(z_test, z_pred)
        r2[degree-1] = R2(z_test, z_pred)

    return mse, r2

# Load the terrain
def main():
    terrain = imread('../data/SRTM_data_Norway_1.tif')
    n = 20
    n_skip = 2 #skipping every n_skip index
    #n = 40
    #n_skip = 1
    std = 0
    maxdegree = 25
    n_B = 100 #Bootstrap iterations
    terrain = terrain[100:n*n_skip+1+100,100:n*n_skip+1+100][1::n_skip] #slice of terrain
    np.random.seed(200)
    noise = 0 #np.random.normal(0, std, size=(n+1,n+1)) #To test for added noise to data
    x, y, z = make_data(n*n_skip, std)

    x = x[1::n_skip]
    y = y[1::n_skip]
    z = (terrain + noise*np.mean(terrain)).ravel()
    mean_scale = np.mean(z)
    std_scale = np.std(z)
    z_scaled = (z - mean_scale)/std_scale #Standard scale

    run_best_lambda = False
    run_tradeoff = True
    run_mse_r2 = True

    if run_mse_r2:
        mse, r2 = MSE_R2_standard_scale(x, y, z, maxdegree)
        plot_MSE(mse, save="mse_deg_real")
        plot_R2(r2, save="r2_deg_real")

    if run_tradeoff:
        """Tradeoff for different choices of degrees and lambdas for the three methods"""
        bias_variance_tradeoff(franke=False, x=x, y=y, z=z_scaled, maxdegree=maxdegree, method="OLS", lamda=1, show=True, save="ols_real_tradeoff")
        plt.show()
        lamda = np.logspace(-13, -1, 4)
        bias_variance_lamda(n, std, maxdegree, n_B, "RIDGE", lamda, "real", False, x, y, z_scaled)
        plt.show()
        bias_variance_lamda(n, std, maxdegree, n_B, "LASSO", lamda, "real", False, x, y, z_scaled)
        plt.show()

    if run_best_lambda:
        """change run_best_lambda to True to run. Will produce same results as found under else:"""
        lmb_ols, deg_ols, mse_ols = lamda_degree_MSE(x, y, z_scaled, "OLS", std, n_lmb = 1, maxdegree = maxdegree, k_folds = 5, max_iter = 100, save=False)
        print("OLS:", lmb_ols, deg_ols, mse_ols)
        lmb_ridge, deg_ridge, mse_ridge = lamda_degree_MSE(x, y, z_scaled, "RIDGE", std, n_lmb = 20, maxdegree = maxdegree, k_folds = 5, max_iter = 100, save=False, lmb_min=-13, lmb_max=-3)
        print("RIDGE:", lmb_ridge, deg_ridge, mse_ridge)
        lmb_lasso, deg_lasso, mse_lasso = lamda_degree_MSE(x, y, z_scaled, "LASSO", std, n_lmb = 20, maxdegree = 32, k_folds = 5, max_iter = 1000, save=True, lmb_min=-14, lmb_max=-3)
        print("LASSO:", lmb_lasso, deg_lasso, mse_lasso)

    else:
        deg_ols = 19
        mse_ols = 0.010703657377970114

        lmb_ridge = 3.792690190732254e-12
        deg_ridge = 20
        mse_ridge = 0.009997042109663854

        lmb_lasso = 1.1288378916846884e-10
        deg_lasso = 29
        mse_lasso = 0.068238282615363

    compare_3d(x, y, z_scaled, 0, deg_ols, lmb_ridge, deg_ridge, lmb_lasso, deg_lasso, name_add="n40", std=std_scale, mean=mean_scale, azim=60)

if __name__ == '__main__':
    main()
