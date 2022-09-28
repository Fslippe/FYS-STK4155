from functions import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler



def sklearn_cross_validation(X, z, k_folds, lamda):
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
    ridge = Ridge(alpha = lamda, fit_intercept=False)
    k_fold = KFold(n_splits = k_folds)
    MSE = cross_val_score(ridge, X, z[:, np.newaxis], scoring='neg_mean_squared_error', cv=k_fold)

    return np.mean(-MSE)

def main():
    degree = 5
    n = 300
    std = 0.5
    k_folds = 10
    x, y, z = make_data(n, std, seed=100)#np.random.randint(101))
    X = design_matrix(x, y, degree)
    lamda = 1
    M = cross_validation(X, z, k_folds, lamda, "ridge", scale=False)
    M_sk = sklearn_cross_validation(X, z, k_folds, lamda)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    z_pred_B = np.mean(bootstrap(X_train, X_test, z_train, z_test, n_B=200, method="RIDGE", lamda=lamda), axis=1)
    M_B = MSE(z_pred_B, z_test)
    print("sklearn:", M_sk)
    print("Manual:", M)
    print("Bootstrap:", M_B)
    print("Diff Crossval Bootstrap:", abs(M-M_B))
    print("Diff maunal sklearn:", abs(M_sk-M))

if __name__ == '__main__':
    main()
