from functions import * 
from grid_search import * 
from gradient_decent import * 
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 11})

def grid_search_GD_OLS(eta, moment, iterations, X_train, X_test, y_train, y_test, descent, method, batch_size=0, lamda=0):
    cost = "Ridge"
    if lamda == 0:
        cost = "OLS"
    
    plt.figure()
    for m in method:
        if m == "none":
            G = GradientDescent("RIDGE", m, eta, moment=0, lamda=lamda,iterations=iterations, seed=100)
        else:
            G = GradientDescent("RIDGE", m, eta, moment, lamda=lamda,iterations=iterations, seed=100)
        if descent == "GD":
            pred = G.GD(X_train, y_train, eval=True, X_test=X_test)
        else:
            pred = G.SGD(X_train, y_train, batch_size, eval=True, X_test=X_test)

        mse = np.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            mse[i] = MSE(y_test, pred[i])
        plt.plot(mse, label="%s, mse=%.5f" %(m, np.min(mse))) 
    
    plt.legend()
    if descent == "SGD":
        plt.title(r"%s %s $\eta=$ %s, batchsize=%i" %(descent, cost, eta, batch_size))
    else:
        plt.title(r"%s %s $\eta=$ %s" %(descent, cost, eta))

    plt.ylabel(r"$MSE$")
    plt.xlabel("iteration") 
    plt.savefig("../figures/%s_methods_%s_eta_%s.png" %(descent, cost, eta), dpi=300, bbox_inches='tight' )

def compare_ridge(eta, moment, iterations, X_train, X_test, y_train, y_test, descent, lamda, method, batch_size=0):
    mse = np.zeros((len(eta), len(lamda)))
    mse_ridge = np.zeros(len(lamda))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            if method == "none":
                G = GradientDescent("RIDGE", method, eta[i], moment=0, lamda=lamda[j],iterations=iterations, seed=100)
            else:
                G = GradientDescent("RIDGE", method, eta[i], moment, lamda=lamda[j],iterations=iterations, seed=100)
            if descent == "GD":
                beta = G.GD(X_train, y_train)
                pred = X_test @ beta
            else:
                beta = G.SGD(X_train, y_train, batch_size)
                pred = X_test @ beta
            mse[i, j] = MSE(y_test, pred)
            mse_ridge[j] = MSE(y_test, X_test @ ridge_regression(X_train, y_train, lamda[j]))
    plt.rcParams.update({'font.size': 11})
    plt.figure()
    df = pd.DataFrame(mse, columns=np.log10(lamda), index=eta)
    plt.title(method)

    sns.heatmap(df, annot=True, linewidths=0, cbar_kws={"label": r"$MSE$"}, vmin=0.001, vmax=0.05, fmt=".5f")

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s_%s_eta_lmb.png" %(method, descent), dpi=300, bbox_inches='tight')
    plt.figure()
    plt.plot(lamda, mse_ridge, label="Ridge mse min: %.5f" %(np.min(mse_ridge)))
    plt.ylabel("MSE")
    plt.xlabel(r"$\lambda$")
    plt.legend()
    plt.xscale("log")
    plt.savefig("../figures/ridge_mse_lamda.png", dpi=300, bbox_inches='tight')
    
def compare_batch_size(eta, moment, iterations, X_train, X_test, y_train, y_test, lamda, method, batch_sizes):
    if method =="moment":
        G = GradientDescent("RIDGE", method, eta, moment=moment, lamda=lamda,iterations=iterations, seed=100)
    else:
        G = GradientDescent("RIDGE", method, eta, moment=0, lamda=lamda,iterations=iterations, seed=100)
    mse = np.zeros(iterations)
    plt.figure()
    plt.title(method)
    for bs in batch_sizes:
        pred = G.SGD(X_train, y_train, bs, eval=True, X_test=X_test)
        for i in range(iterations):
            mse[i] = MSE(y_test, pred[i])

        plt.plot(mse, label="batch_size=%s" %(bs))
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(r"$MSE$")
    plt.xlabel("iteration") 
    plt.savefig("../figures/SGD_batch_size_%s.png" %(method), dpi=300, bbox_inches='tight' )

def main():
    n = 100
    np.random.seed(100)
    x = np.linspace(-1, 1, n)
    print(np.shape(x))
    noise = np.random.normal(0, 0.1, n)
    y = test_func_1D(x, 4, noise).reshape(n, 1)
    y = (y - np.mean(y))
    X = design_matrix_1D(x, 6) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    print(np.max(np.abs(X)))
    eta = 0.1 
    moment = 0.3
    iterations=100
    method = ["ADAM", "RMSprop", "AdaGrad", "none", "momentum"]
    #grid_search_GD_OLS(eta, moment, iterations, X_train, X_test, y_train, y_test, method=method, descent="GD" )
    #grid_search_GD_OLS(eta, moment, iterations, X_train, X_test, y_train, y_test, method=method, batch_size=16, descent="SGD" )
    #plt.show()
    eta = 0.01 
    #grid_search_GD_OLS(eta, moment, iterations, X_train, X_test, y_train, y_test, method=method, descent="GD" )
    #grid_search_GD_OLS(eta, moment, iterations, X_train, X_test, y_train, y_test, method=method, batch_size=16, descent="SGD" )
    plt.show()
    beta = OLS(X_train, y_train)
    pred_OLS = X_test @ beta
    print("MSE OLS", MSE(y_test, pred_OLS))
    lamda = np.array([ 1e-6, 1e-4, 1e-2, 1e-1, 10**(-0.2)])
    eta = np.array([0.01, 0.1, 0.2, 0.5, 0.8])
    batch_sizes = [4, 8, 16, 20, 40]
    for m in method:
        #compare_ridge(eta, moment, iterations, X_train, X_test, y_train, y_test, "GD", lamda, m, batch_size=0)
        #compare_ridge(eta, moment, iterations, X_train, X_test, y_train, y_test, "SGD", lamda, m, batch_size=16)
        compare_batch_size(eta=0.1, moment=0, iterations=20, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, lamda=0, method=m, batch_sizes=batch_sizes)
    plt.show()

if __name__ == "__main__":
    main()