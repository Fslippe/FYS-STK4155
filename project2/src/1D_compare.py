from functions import * 
from grid_search import * 
from gradient_decent import * 

def grid_search_GD_OLS(eta, moment, iterations, X, y, descent, method, batch_size=0, lamda=0):
    cost = "Ridge"
    if lamda == 0:
        cost = "OLS"
    
    mse = np.zeros(y.shape[0])
    plt.figure()
    for m in method:
        if m == "none":
            G = GradientDescent("RIDGE", m, eta, moment=0, lamda=lamda,iterations=iterations)
        else:
            G = GradientDescent("RIDGE", m, eta, moment, lamda=lamda,iterations=iterations)
        if descent == "GD":
            pred = G.GD(X, y, eval=True)
        else:
            pred = G.SGD(X, y, batch_size, eval=True)

        mse = np.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            mse[i] = MSE(y, pred[i])
        plt.plot(mse, label="%s, mse=%.5f" %(m, np.min(mse))) 
    
    plt.legend()
    if descent == "SGD":
        plt.title(r"%s %s $\eta=$ %s, batchsize=%i" %(descent, cost, eta, batch_size))
    else:
        plt.title(r"%s %s $\eta=$ %s" %(descent, cost, eta))

    plt.ylabel(r"$MSE$")
    plt.xlabel("iteration") 
    plt.savefig("../figures/%s_methods_%s_eta_%s.pdf" %(descent, cost, eta), dpi=300, bbox_inches='tight' )

def compare_ridge(eta, moment, iterations, X, y, descent, lamda, method, batch_size=0):
    mse = np.zeros((len(eta), len(lamda)))
    mse_ridge = np.zeros(len(lamda))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            if method == "none":
                G = GradientDescent("RIDGE", method, eta[i], moment=0, lamda=lamda[j],iterations=iterations)
            else:
                G = GradientDescent("RIDGE", method, eta[i], moment, lamda=lamda[j],iterations=iterations)
            if descent == "GD":
                beta = G.GD(X, y)
                pred = X @ beta
            else:
                beta = G.SGD(X, y, batch_size)
                pred = X @ beta
            mse[i, j] = MSE(y, pred)
            mse_ridge[j] = MSE(y, X @ ridge_regression(X, y, lamda[j]))
    plt.figure()
    plt.title(method)
    df = pd.DataFrame(mse, columns=np.log10(lamda), index=eta)
    sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.001, vmax=0.1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s_%s_eta_lmb.png" %(method, descent), dpi=300, bbox_inches='tight')
    plt.figure()
    plt.plot(lamda, mse_ridge, label="Ridge mse min: %.5f" %(np.min(mse_ridge)))
    plt.ylabel("MSE")
    plt.xlabel(r"$\lambda$")
    plt.legend()
    plt.xscale("log")
    plt.savefig("../figures/ridge_mse_lamda.pdf", dpi=300, bbox_inches='tight')
    
def compare_batch_size(eta, moment, iterations, X, y, lamda, method, batch_sizes):
    if method =="moment":
        G = GradientDescent("RIDGE", method, eta, moment=moment, lamda=lamda,iterations=iterations)
    else:
        G = GradientDescent("RIDGE", method, eta, moment=0, lamda=lamda,iterations=iterations)
    mse = np.zeros(iterations)
    plt.figure()
    plt.title(method)
    for bs in batch_sizes:
        pred = G.SGD(X, y, bs, eval=True)
        for i in range(iterations):
            mse[i] = MSE(y, pred[i])
        plt.plot(mse, label="batch_size=%s" %(bs))
    plt.legend()
    plt.ylabel(r"$MSE$")
    plt.xlabel("iteration") 
    plt.savefig("../figures/SGD_batch_size_%s.pdf" %(method), dpi=300, bbox_inches='tight' )

def main():
    n = 100
    np.random.seed(100)
    x = np.linspace(-1, 1, n)
    print(np.shape(x))
    noise = np.random.normal(0, 0.1, n)
    y = test_func_1D(x, 4, noise).reshape(n, 1)
    y = (y - np.mean(y))
    X = design_matrix_1D(x, 4)
    eta = 0.1 
    moment = 0.3
    iterations=100
    method = ["ADAM", "RMSprop", "AdaGrad", "none", "momentum"]

    grid_search_GD_OLS(eta, moment, iterations, X, y, method=method, descent="GD" )
    grid_search_GD_OLS(eta, moment, iterations, X, y, method=method, batch_size=10, descent="SGD" )
    plt.show()
    iterations=200

    beta = OLS(X, y)
    pred_OLS = X @ beta
    print("MSE OLS", MSE(y, pred_OLS))
    lamda = np.logspace(-6,-1, 6)
    eta = np.array([0.005, 0.01, 0.1, 0.2, 0.5])
    batch_sizes = [5, 10, 20, 25, 50]
    for m in method:
        #compare_ridge(eta, moment, iterations, X, y, "GD", lamda, m, batch_size=0)
        #compare_ridge(eta, moment, iterations, X, y, "SGD", lamda, m, batch_size=10)
        compare_batch_size(eta=0.1, moment=0, iterations=iterations, X=X, y=y, lamda=0, method=m, batch_sizes=batch_sizes)
    plt.show()

if __name__ == "__main__":
    main()