from functions import *
from c import bias_variance_tradeoff
from d import sklearn_cross_validation

def bias_variance_lamda(n, std, maxdegree, n_B, method, lamda):
    """
    runs biass variance tradeoff for chosen method for different lamdas
    """
    i = 0

    mse = np.zeros((len(lamda)+1, maxdegree+1))
    bias = np.zeros((len(lamda)+1, maxdegree+1))
    variance = np.zeros((len(lamda)+1, maxdegree+1))

    poly = np.arange(0, maxdegree+1)
    for lmb in lamda:
        #bias_variance_tradeoff(n=n, std=std, maxdegree=maxdegree, n_B=n_B, plot=True, method=method, lamda=lmb)

        mse[i], bias[i], variance[i] =  bias_variance_tradeoff(n=n, std=std, maxdegree=maxdegree, n_B=n_B, plot=False, method=method, lamda=lmb)
        print(mse[i])
        plt.plot(poly, bias[i], "--", label="Bias")
        plt.plot(poly, mse[i], "-", label="mse")
        plt.plot(poly, variance[i], "-", label="variance")
        i +=1

    plt.show()

def cross_validation_lamda(n, std, k_folds, method, lamda, degree=5, savefig=False):
    x, y, z = make_data(n, std, seed=100)
    X = design_matrix(x, y, degree)
    i = 0

    M = np.zeros(len(lamda))

    for lmb in lamda:
        if method == "LASSO":
            M[i] = sklearn_cross_validation(X, z, k_folds, lmb, method)
        else:
            M[i] = cross_validation(X, z, k_folds, lmb, method, scale=False)
        i +=1

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
    plt.rcParams.update({"font.size": 8})
    plt.title("Cross validation MSE for %i kfolds" %(k_folds))
    plt.xlabel(r"$\lambda$")
    plt.ylabel("$MSE$")
    plt.plot(lamda, M, label="%s Degree: %i" %(method, degree))
    argmin = np.argmin(M)
    print("\n%s" %(method))
    print("min MSE:", M[argmin])
    print("lambda:", lamda[argmin])
    plt.scatter(lamda[argmin], M[argmin], label=r"Min: $\lambda=$%.2e, $MSE=$%.5f" %(lamda[argmin], M[argmin]))
    plt.xscale("log")
    plt.legend()
    if savefig == True:
        plt.savefig("../figures/cross_val_lambda_ridge.png", dpi=300, bbox_inches='tight')
    return lamda[argmin]

def main():
    n = 15
    std = 0.2
    maxdegree = 15
    n_B = 100
    method = "RIDGE"
    lamda = np.logspace(-4, 1, 3)
    k_folds = 10
    bias_variance_lamda(n, std, maxdegree, n_B, method, lamda)
    for degree in range(6, 11):
        lamda = np.logspace(-6.5, -3.5, 100)
        lmb_min = cross_validation_lamda(n, std, k_folds, method, lamda, degree=degree)
        lamda = np.logspace(-4.5, 0.5, 100)
    plt.legend(loc="upper right")
    plt.savefig("../figures/cross_val_lambda_ridge_deg_6-10.png", dpi=300, bbox_inches='tight')
    plt.show()

    #cross_validation_lamda(n, std, k_folds, method, lamda, degree=10)

if __name__ == '__main__':
    main()
