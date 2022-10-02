
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
from functions import *
from b import *
from f import lamda_degree_MSE

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

def plot_3d_trisurf(x, y, z, scale_std=1, scale_mean=0, savename=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(x, y, z*scale_std + scale_mean, cmap=cm.coolwarm, linewidth=0.2, antialiased=False)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%i"))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylabel(r"$y$")
    ax.view_init(azim=110)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    if savename != None:
        plt.savefig("../figures/%s.png" %(savename))

# Load the terrain
def main():
    terrain = imread('../data/SRTM_data_Norway_1.tif')

    n = 100
    std = 0.2
    degree = 50 # polynomial order
    maxdegree = 25
    terrain = terrain[100:n+1+100,100:n+1+100]
    print(np.mean(terrain))
    noise = np.random.normal(0, std, size=(n+1,n+1))
    x, y, z = make_data(n, std)
    z = (terrain + noise*100).ravel()
    #mse, r2 = MSE_R2_2(x, y, z, maxdegree, scaler, method)
    #plot_MSE(mse)
    #plot_R2(r2)
    method = "OLS"

    scaler = StandardScaler()
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x.ravel(), y.ravel(), z)
    X_train = design_matrix(x_train, y_train, degree)
    X_test = design_matrix(x_test, y_test, degree)

    scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    z_train = (z_train - np.mean(z_train)) / np.std(z_train)
    z_test = (z_test - np.mean(z_train)) / np.std(z_train)
    #X_test = scaler.transform(X_test)


    x, y, z = make_data(n, std)
    z = (z - np.mean(z))/np.std(z)

    #lamda_degree_MSE(x, y, z, "OLS", std, n_B = 100, n_lmb = 20, maxdegree = maxdegree, k_folds = 5, max_iter = 1000, save=False)
    if method == "OLS":
        beta = OLS(X_train, z_train)
        z_pred = beta @ X_test.T
    plot_3d_trisurf(x_test, y_test, z_pred, np.std(z_test), np.mean(z_test))
    x, y, z = make_data(n, std)
    print(np.shape(x))
    print(np.shape(y))

    plot_3D(x, y, terrain)
    plt.show()

    #X_test = design_matrix(x_test, y_test, degree)


    #X_train, X_test_tmp, z_train, z_test_tmp = standard_scale(X_train, z_train)
    #X_train, X_test_tmp, z_train, z_test_tmp = standard_scale(X_test, z_test)


    #beta = OLS(X_train, z_train)
    #z_pred = X_test @ beta

    #compare_scale(X, z)
    #MSE(z_test, z_)
    #mse, mse_train, r2, r2_train, beta_OLS = MSE_R2(x, y, z, maxdegree)
    #plot_train_test(mse_train, mse, "mse")

    # Show the terrain
    #x, y, z = make_data(44, std)

    #ax.view_init(azim=70)


if __name__ == '__main__':
    main()
