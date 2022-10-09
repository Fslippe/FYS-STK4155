from functions import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def sklearn_cross_validation(X, z, k_folds, lamda, method="RIDGE"):
    """
    Uses sklearns funtion for crossvalidation and ridge regression
    to find MSE
    Takes in:
    - X:        Design matrix of some degree
    - z:        Matching dataset
    - k_folds:  number of k_folds in the cross validation algorithm
    - lamda:    chosen lamda for the Ridge regression
    Returns:
    - MSE as a mean over the MSE returned by the cross validation function
    """
    if method == "RIDGE":
        model = Ridge(alpha = lamda)
    elif method == "LASSO":
        model = Lasso(alpha = lamda)
    elif method == "OLS":
        model = LinearRegression()

    k_fold = KFold(n_splits = k_folds, shuffle=True)
    MSE = cross_val_score(model, X, z[:, np.newaxis], scoring='neg_mean_squared_error', cv=k_fold)
    return np.mean(-MSE)

def compare_crossval_bootstrap(x, y, z, maxdegree, k_folds, n_B, method, lamda=1):
    """
    Comparing bootstrap and cross validation for different degrees
    Takes in:
    - x:            x data
    - y:            y data
    - z:            z data
    - maxdegree     Maximum degree to plot for (plots from a degree of 1)
    - k_folds:      number of k_folds in the cross validation algorithm
    - n_B:          Number of Bootstrap iterations
    - lamda (opt):  chosen lamda if method is RIDGE or LASSO

    Plots compartison plots between bootrap and cross validation and saves them
    """
    mse_B = np.zeros(maxdegree)
    mse_cv = np.zeros(maxdegree)
    poly = np.arange(1, maxdegree+1)

    for degree in range(1, maxdegree+1):
        X = design_matrix(x, y, degree)
        mse_cv[degree-1] = cross_validation(X, z, k_folds, lamda, method)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        z_pred_B = np.mean(bootstrap(X_train, X_test, z_train, z_test, n_B, method, lamda), axis=1)
        mse_B[degree-1] = MSE(z_pred_B, z_test)

    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("%s for %i kfolds and %i Bootstrap iterations" %(method, k_folds, n_B))
    plt.xlabel("Polynomial Degree")
    plt.ylabel(r"$MSE$")
    plt.plot(poly, mse_B, label="Bootstrap")
    plt.plot(poly, mse_cv, label="Cross validation")
    plt.legend()
    plt.savefig("../figures/boot_cv_comp_%s_%i_%i.png" %(method, n_B, k_folds), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    degree = 5
    n = 30
    std = 0.2
    k_folds = 5
    maxdegree = 15
    n_B = 100
    x, y, z = make_data(n, std, seed=200)#np.random.randint(101))
    X = design_matrix(x, y, degree)
    lamda = 1

    method = "OLS"
    #Comparing Sklearn and implemented cross validation method
    M = cross_validation(X, z, k_folds, lamda, method, scale=False)
    M_sk = sklearn_cross_validation(X, z, k_folds, lamda)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    z_pred_B = np.mean(bootstrap(X_train, X_test, z_train, z_test, n_B=n_B, method=method, lamda=lamda), axis=1)
    M_B = MSE(z_pred_B, z_test)
    print("sklearn:", M_sk)
    print("Manual:", M)
    print("Bootstrap:", M_B)
    print("Diff Crossval Bootstrap:", abs(M-M_B))
    print("Diff maunal sklearn:", abs(M_sk-M))
    compare_crossval_bootstrap(x, y, z, maxdegree, k_folds, n_B, method, lamda)


if __name__ == '__main__':
    main()
