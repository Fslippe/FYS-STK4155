from functions import *
from c import bias_variance_tradeoff
from d import sklearn_cross_validation

def bias_variance_lamda(n, std, maxdegree, n_B, method, lamda, name_add="", franke=True, x=None, y=None, z=None, max_iter=100):
    """
    runs biass variance tradeoff for chosen method for different lamdas
    and plots the tradeoffs
    takes in:
    - n:                number of steps of produced data
    - std:              std of noise added to data
    - n_B:              number of Bootstrap iterations
    - method:           Regression method
    - lamda:            1D array of different lamdas to run the tradeoff for
    - name_add (opt):   To add at end of saved filename
    - Franke (opt):     If using Franke function True otherwise False
    - x (opt):          if Franke False - x data
    - y (opt):          if Franke False - y data
    - z (opt):          if Franke False - z data
    - max_iter (opt):   maximum number of iterations used for lasso prediction
    """
    i = 0

    mse = np.zeros((len(lamda)+1, maxdegree+1))
    bias = np.zeros((len(lamda)+1, maxdegree+1))
    variance = np.zeros((len(lamda)+1, maxdegree+1))

    poly = np.arange(0, maxdegree+1)
    for lmb in lamda:
        mse[i], bias[i], variance[i] =  bias_variance_tradeoff(franke, x, y, z, n=n, std=std, maxdegree=maxdegree, n_B=n_B, plot=False, method=method, lamda=lmb, seed=200, max_iter=max_iter)
        plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("%s Tradeoff for $\lambda=$ %.0e" %(method, lmb))
        plt.plot(poly, mse[i], label="mse")
        plt.plot(poly, bias[i], label=r"Bias$^2$")
        plt.plot(poly, variance[i], label="variance")
        plt.xlabel("Polynomial degree")
        plt.legend()
        plt.savefig("../figures/tradeoff_%s_%.0e%s.png" %(method, lmb, name_add), dpi=300, bbox_inches="tight")

        i +=1

def main():
    n = 30
    std = 0.2
    maxdegree = 20
    n_B = 100

    method = "RIDGE"
    lamda = np.logspace(-12, -7, 6)
    k_folds = 5
    bias_variance_lamda(n, std, maxdegree, n_B, method, lamda, name_add="_20", max_iter=1000)
    plt.show()
    method = "LASSO"
    lamda = np.logspace(-6, -1, 6)
    k_folds = 5
    bias_variance_lamda(n, std, maxdegree, n_B, method, lamda, name_add="_20", max_iter=1000)
    plt.show()

if __name__ == '__main__':
    main()
