from functions import * 
from grid_search import * 
from gradient_decent import * 

def grid_search_GD_OLS(eta, moment, iterations, X, y, descent, batch_size=0, lamda=0):
    cost = "Ridge"
    if lamda == 0:
        cost = "OLS"
    
    mse = np.zeros(y.shape[0])
    method = ["ADAM", "RMSprop", "AdaGrad", "none", "momentum"]
    plt.figure()
    for m in method:
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

def main():
    n = 100
    np.random.seed(100)
    x = np.linspace(-1, 1, n)
    noise = np.random.normal(0, 0.1, n)
    y = test_func_1D(x, 4, noise).reshape(n, 1)
    y = (y - np.mean(y))
    X = design_matrix_1D(x, 4)
    eta = 0.1 
    moment = 0.2
    iterations=100
    grid_search_GD_OLS(eta, moment, iterations, X, y, descent="GD" )
    iterations=100
    grid_search_GD_OLS(eta, moment, iterations, X, y, batch_size=10, descent="SGD" )
    plt.show()
    beta = OLS(X, y)
    pred_OLS = X @ beta
    print("MSE OLS", MSE(y, pred_OLS))

    

if __name__ == "__main__":
    main()