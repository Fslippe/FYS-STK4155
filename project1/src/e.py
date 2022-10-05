from functions import *
from c import bias_variance_tradeoff
from d import sklearn_cross_validation

def bias_variance_lamda(n, std, maxdegree, n_B, method, lamda, name_add="", franke=True, x=None, y=None, z=None):
    """
    runs biass variance tradeoff for chosen method for different lamdas
    """
    i = 0

    mse = np.zeros((len(lamda)+1, maxdegree+1))
    bias = np.zeros((len(lamda)+1, maxdegree+1))
    variance = np.zeros((len(lamda)+1, maxdegree+1))

    poly = np.arange(0, maxdegree+1)
    for lmb in lamda:
        mse[i], bias[i], variance[i] =  bias_variance_tradeoff(franke, x, y, z, n=n, std=std, maxdegree=maxdegree, n_B=n_B, plot=False, method=method, lamda=lmb, seed=200)
        plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("%s Tradeoff for $\lambda=$ %.0e" %(method, lmb))
        plt.plot(poly, mse[i], label="mse")
        plt.plot(poly, bias[i], label=r"Bias$^2$")
        plt.plot(poly, variance[i], label="variance")
        plt.xlabel("Polynomial degree")
        plt.legend()
        plt.savefig("../figures/tradeoff_%s_%.0e%s.png" %(method, lmb, name_add), dpi=300, bbox_inches="tight")

        i +=1
    print(np.min(bias))
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
        plt.savefig("../figures/cross_val_lambda_ridge.png", dpi=300, bbox_inches="tight")
    return lamda[argmin]

def main():
    n = 30
    std = 0.2
    maxdegree = 15
    n_B = 100
    method = "RIDGE"
    lamda = np.logspace(-10, -5, 6)
    k_folds = 5
    bias_variance_lamda(n, std, maxdegree, n_B, method, lamda)
    plt.show()
    method = "LASSO"
    lamda = np.logspace(-14, -8, 6)
    k_folds = 5
    bias_variance_lamda(n, std, maxdegree, n_B, method, lamda)
    plt.show()

    #cross_validation_lamda(n, std, k_folds, method, lamda, degree=10)

if __name__ == '__main__':
    main()
