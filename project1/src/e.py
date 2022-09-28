from functions import *
from c import bias_variance_tradeoff

def bias_variance_lamda(n, std, maxdegree, n_B, method, lamda):
    """
    runs biass variance tradeoff for chosen method for different lamdas
    """
    i = 0
    for lmb in lamda:
        bias_variance_tradeoff(n=n, std=std, maxdegree=maxdegree, n_B=n_B, plot=True, method=method, lamda=lmb)

        #MSE[i], bias[i], variance[i] = bias_variance_tradeoff(n=n, std=std, maxdegree=maxdegree, n_B=n_B, plot=False, method=method, lamda=lmb)
        #i +=1

def cross_validation_lamda(n, std, k_folds, method, lamda, degree=5):
    x, y, z = make_data(n, std, seed=100)
    X = design_matrix(x, y, degree)
    i = 0

    M = np.zeros(len(lamda))

    for lmb in lamda:
        M[i] = cross_validation(X, z, k_folds, lmb, method, scale=False)
        i +=1

    plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)

    plt.rcParams.update({"font.size": 14})
    plt.title("Cross validation MSE for %s regression" %(method))
    plt.xlabel(r"$\lambda$")
    plt.ylabel("$MSE$")
    plt.plot(lamda, M)
    argmin = np.argmin(M)
    plt.plot(lamda[argmin], M[argmin], "ro", label=r"Min: $\lambda=$%.5f, $MSE=$%.5f" %(lamda[argmin], M[argmin]))
    plt.legend()
    plt.xscale("log")
    plt.savefig("../figures/cross_val_lambda_ridge.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    n = 15
    std = 0.2
    maxdegree = 15
    n_B = 100
    method = "RIDGE"
    lamda = np.logspace(-5, 1, 5)
    k_folds = 10
    #bias_variance_lamda(n, std, maxdegree, n_B, method, lamda)
    lamda = np.logspace(-10, 6, 1000)
    cross_validation_lamda(n, std, k_folds, method, lamda)
    lamda = np.logspace(-4.5, 0.5, 1000)
    cross_validation_lamda(n, std, k_folds, method, lamda)
if __name__ == '__main__':
    main()
