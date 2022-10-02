from functions import *
from e import *
from c import bias_variance_tradeoff
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import seaborn as sns

plt.rcParams.update({"font.size": 15})
#sns.set_style("whitegrid")

def best_lamda_plot(k_folds = 10, n_lmb = 200):
    """PLOTTING RESULTS FROM CROSS VALIDATION"""

    degree = 6
    n = 100
    std = 0.2
    lamda = np.logspace(-7, -1, n_lmb)
    M_ridge = np.zeros(n_lmb)
    M_OLS = np.zeros(n_lmb)
    M_lasso = np.zeros(n_lmb)
    lmb_min_lasso = cross_validation_lamda(n, std, k_folds, "LASSO", lamda, degree, savefig=False)
    lmb_min_ridge = cross_validation_lamda(n, std, k_folds, "RIDGE", lamda, degree, savefig=False)
    plt.savefig("../figures/lasso_ridge_mse_lambda.png", dpi=300, bbox_inches='tight')
    plt.show()
    return lamda_ridge, lamda_lasso

def lamda_degree_MSE(x, y, z, method, std, n_B = 100, n_lmb = 50, maxdegree = 15, k_folds = 5, max_iter = 1000, save=True):
    """
    Function to find best degree and lambda parameter
    for the chosen regression method
    """

    degree = np.arange(1, maxdegree+1)
    lamda = np.logspace(-12, -1, n_lmb)

    if method == "RIDGE" or method == "LASSO":
        degree, lamda = np.meshgrid(degree,lamda)
        mse = np.zeros(np.shape(degree))

        for i in range(maxdegree):
            X = design_matrix(x, y, degree[0, i])
            for j in range(n_lmb):
                mse[j, i] = cross_validation(X, z, k_folds, lamda[j, i], method, max_iter)
            print("\n\n\n ---DEGREE---- %i\n\n\n" %(i))

    elif method == "OLS":
        mse = np.zeros(np.shape(degree))
        for i in range(maxdegree):
            X = design_matrix(x, y, degree[i])
            mse[i] = cross_validation(X, z, k_folds, method=method)

    argmin = np.unravel_index(np.argmin(mse), mse.shape)
    print("---%s---" %(method))
    print("Degree of lowest MSE for %i kfolds" %(k_folds), degree[argmin])
    plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if method != "OLS":
        print("Lambda of lowest MSE for %i kfolds" %(k_folds), lamda[argmin])
        plt.contourf(degree, lamda, mse, 50, cmap="RdBu")
        plt.colorbar(label=r"$MSE$")
        plt.ylabel(r"$\lambda$")
        plt.yscale("log")
        plt.scatter(degree[argmin], lamda[argmin], marker="x", s=80, label=r"min MSE: %.3f, Lambda: %.2e" %(mse[argmin], lamda[argmin]))
        plt.legend(fontsize=12)

    else:
        plt.plot(degree, mse, "--o", fillstyle="none")
        plt.ylabel(r"$MSE$")
        plt.scatter(degree[argmin], mse[argmin], color="k", marker="x", s=80, label="min MSE: %.3f" %(mse[argmin]))
        plt.legend()

    plt.xlabel("Degree")

    plt.grid(True)

    if save:
        plt.savefig("../figures/best_lambda_%s_0%i.png" %(method, std*10), dpi=300, bbox_inches='tight')
    plt.show()

    return lamda[argmin], degree[argmin], mse[argmin]

def main():
    degree = 6
    n = 22
    std = 0.2
    maxdegree = 15

    n = 50
    x, y, z = make_data(n, std, seed=200)

    lmb, deg, mse = lamda_degree_MSE(x, y, z, "OLS", std, save=True)
    #lmb_ridge, deg_ridge, mse_ridge = lamda_degree_MSE("RIDGE", std)
    #lmb_lasso, deg_lasso, mse_lasso = lamda_degree_MSE("LASSO", std)
    lmb_ridge = 1.21e-4
    lmb_lasso = 1.53e-5
    x, y, z = make_data(n, std, seed=100)#np.random.randint(101))
    X = design_matrix(x, y, degree)
    #bias_variance_tradeoff(n, std, maxdegree, method="RIDGE", lamda=lmb_ridge, show=False)
    #plt.savefig("../figures/bias_variance_best_ridge_05.png", dpi=300, bbox_inches='tight')
    bias_variance_tradeoff(n, std, maxdegree, method="OLS",show=False)
    plt.savefig("../figures/bias_variance_ols_05.png", dpi=300, bbox_inches='tight')

    #bias_variance_tradeoff(n, std, maxdegree, method="LASSO", lamda=lmb_lasso,show=False)
    #plt.savefig("../figures/bias_variance_best_lasso_05.png", dpi=300, bbox_inches='tight')


    #lamda_ridge, lamda_lasso = best_lamda_plot()
    lamda_ridge = 9.771241535346501e-06 #Values taken from print when running best_lamda_plot
    lamda_lasso = 0.0006747544053110693
    #bias_variance_tradeoff(n, std, maxdegree, method="RIDGE", lamda=lamda_ridge)
    #bias_variance_tradeoff(n, std, maxdegree, method="LASSO", lamda=lamda_lasso)





if __name__ == '__main__':
    main()
