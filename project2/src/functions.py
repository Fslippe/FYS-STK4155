from re import A
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
import autograd.numpy as np
import pandas as pd
from matplotlib import cm
from autograd import elementwise_grad
from autograd import grad
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample, shuffle
from sklearn.linear_model import Lasso, Ridge
import time
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.rcParams.update({"font.size": 14})

def no_tune(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size):
    return eta*grads + delta_mom*change

def no_tune_SGD(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size):
    change = eta*grads + delta_mom*change 
    return change

def AdaGrad(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size):
    """
    AdaGrad tuning method
    """
    grad_square += grads**2
    alpha = eta/(1e-8 + np.sqrt(grad_square))
    change = alpha*grads + delta_mom*change
    
    return change

def RMSprop(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size):
    """
    RMSprop tuning method
    """
    s_t = grad_square
    grad_square += grads**2
    s_t = s_t*rho1 + grad_square*(1 - rho1)
    alpha = eta/(1e-8 + np.sqrt(s_t))
    change = alpha*grads + delta_mom*change
    return change

def ADAM(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size):
    """
    ADAM tuning method
    """
    s_t = grad_square
    m_t = grads_pre
    m_t = rho1*m_t + (1-rho1)*grads
    grad_square += grads**2
    s_t = rho2*s_t + (1-rho2)*grad_square 
    m_t = m_t / (1-rho1)
    s_t = s_t / (1-rho2)
    change = eta * m_t / (1e-8 + np.sqrt(s_t)) + delta_mom*change
    return change

def OLS(X, z, inversion_method="", iterations=1000, n_epochs=50, tuning_func=no_tune,
        batch_size=5, delta_mom=0, eta=0.1, t0=5, t1=50, rho1=0.9, rho2=0.99, lamda=0):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    if inversion_method == "GD":
        beta = GD(X, z, rho1, rho2, eta, iterations, delta_mom, tuning_func)
    elif inversion_method == "SGD":
        if tuning_func==no_tune:
            tuning_func = no_tune_SGD
        beta = SGD(X, z, n_epochs, batch_size, rho1, rho2, eta, iterations, delta_mom, tuning_func)
    else:
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z

    return beta

def ridge_regression(X, z, lamda, inversion_method="", iterations=1000, n_epochs=50, tuning_func=no_tune,
        batch_size=5, delta_mom=0, eta=0.1, t0=5, t1=50, rho1=0.9, rho2=0.99):

    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    if inversion_method == "GD":
        beta = GD(X, z, rho1, rho2, eta, iterations, delta_mom, tuning_func, lamda=lamda)
    elif inversion_method == "SGD":
        if tuning_func==no_tune:
            tuning_func = no_tune_SGD
        beta = SGD(X, z, n_epochs, batch_size, rho1, rho2, eta, iterations, delta_mom, tuning_func, lamda=lamda)
    else:
        N = X.shape[1]
        beta = np.linalg.pinv(X.T @ X + lamda*np.eye(N)) @ X.T @ z

    return beta

def lasso_regression(X, z, lamda, max_iter=int(1e2), tol=1e-2):
    """
    Sklearns function for lasso regression to find beta
    Takes in:
    - X:        Design matrix of some degree
    - z:        Matching dataset
    - lamda:    chosen lamda for the lasso regression
    returns:
    - beta
    """
    lasso = Lasso(lamda, tol=tol, max_iter=max_iter)
    lasso.fit(X, z)
    return lasso


def GD(X, z, rho1, rho2, eta, iterations, delta_mom, tuning_func, epoch=1, m=1, start=1, batch_size=1, t0=1, t1=1, lamda=0):
    """
    Gradient descent method to find theta for OLS and Ridge
    """
    np.random.seed(100)
    n = X.shape[0] #
    N = X.shape[1]
    theta = np.random.randn(N, 1)
    grad_square = np.zeros((N, 1))
    change = 0
    grads = 0

    for i in range(iterations):
        grads_pre = grads
        grads = 2/n * X.T @ (X @ theta - z) + 2*lamda*theta
        change = tuning_func(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size)
        theta -= change 

    return theta


def SGD(X, z, n_epochs, batch_size, rho1, rho2, eta, iterations, delta_mom, tuning_func, lamda=0):
    """
    Stochastic Gradient decent method to find theta for OLS and Ridge
    """
    np.random.seed(100)

    n = X.shape[0] #
    N = X.shape[1]
    
    m = int(n/batch_size) # minibatches
    theta = np.random.randn(N,1)
    change = 0
    for epoch in range(n_epochs):
        X_shuffle, y_shuffle = shuffle(X, z)
        grad_square = np.zeros((N, 1))
        grads = 0
        for start in range(0, n, batch_size):
            X_batch = X_shuffle[start:start+batch_size]
            y_batch = y_shuffle[start:start+batch_size]
            grads_pre = grads
            grads = 2/batch_size * X_batch.T @ ((X_batch @ theta) - y_batch) + 2*lamda*theta       
            change = tuning_func(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size)
            theta -= change
    
    return theta

def C_OLS(X, y, beta, n_div=True):
    """
    Cost function of OLS
    """
    if n_div == False:
        return np.sum((y - X @ beta)**2)
    elif n_div == True:
        n = X.shape[0]
        return 1/n * np.sum((y - X @ beta)**2)

def C_Ridge(X, y, beta, lamda, n_div=True):
    """
    Cost function of Ridge
    """
    if n_div == False:
        return np.sum((y - X @ beta)**2)
    elif n_div == True:
        n = X.shape[0]
        return 1/n * np.sum((y - X @ beta)**2)

def C_OLS(X, y, beta, n_div=True):
    """
    Cost function of OLS
    """
    if n_div == False:
        return np.sum((y - X @ beta)**2)
    elif n_div == True:
        n = X.shape[0]
        return 1/n * np.sum((y - X @ beta)**2)

def GD_newton(X, y, iterations=1000):
    """
    Using Newton's method and autograd
    """
    n = X.shape[0] 
    N = X.shape[1]
    H = 2/n *X.T @ X 
    H_inv = np.linalg.pinv(H)
    beta = np.random.randn(N,1)
    train_grad = grad(C_OLS, 2)

    for i in range(iterations):
        grads = train_grad(X, y, beta)
        beta -= H_inv @ grads 

    return beta


def MSE(data, model):
    """
    takes in actual data and modelled data to find
    Mean Squared Error
    """
    MSE = mean_squared_error(data.ravel(), model.ravel())
    return MSE

def R2(data, model):
    """
    takes in actual data and modelled data to find
    R2 score
    """
    R2 = r2_score(data.ravel(), model.ravel())
    return R2


def design_matrix_1D(x, degree):
    N = len(x)
    X = np.ones((N, degree+1))

    for i in range(1, degree+1):
        X[:,i] = x**i

    return X

def make_data(n, noise_std, seed=1, terrain=False):
    """
    Make data z=f(x)+noise for n steps and normal distributed
    noise with standard deviation equal to noise_std
    """
    np.random.seed(seed)
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    x, y = np.meshgrid(x, y)

    noise = np.random.normal(0, noise_std, size=(n+1,n+1))
    z = FrankeFunction(x, y) + noise
    return x, y, z.ravel()

def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def design_matrix(x, y, degree):
    """
    Setting up design matrix with dependency on x and y for a chosen degree
    [x,y,xy,x²,y²,...]
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((degree+1)*(degree+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def plot_inversion_compare(X, y, x, method, tuning_func, delta_mom, lamda, n_epochs, batch_size, label):
    beta = method(X, y, inversion_method="GD", tuning_func=tuning_func, lamda=lamda)
    pred = X @ beta 

    beta_momentum = method(X, y, inversion_method="GD", tuning_func=tuning_func, delta_mom=delta_mom, lamda=lamda)
    pred_momentum = X @ beta_momentum

    beta_SGD = method(X, y, inversion_method="SGD", tuning_func=tuning_func, n_epochs=n_epochs, batch_size=batch_size, lamda=lamda)
    pred_SGD = X @ beta_SGD 

    beta_SGD_momentum = method(X, y, inversion_method="SGD", tuning_func=tuning_func, n_epochs=n_epochs, batch_size=batch_size, delta_mom=delta_mom, lamda=lamda)
    pred_SGD_momentum = X @ beta_SGD_momentum

    plt.plot(x, pred, label="GD %s %.4f" %(label, MSE(y,pred)))
    plt.plot(x, pred_momentum, "--", label="GD mom %s %.4f" %(label, MSE(y, pred_momentum)))
    plt.plot(x, pred_SGD, label="SGD %s %.4f" %(label, MSE(y,pred_SGD)))
    plt.plot(x, pred_SGD_momentum, linestyle="dotted", label="SGD mom %s %.4f" %(label, MSE(y, pred_SGD_momentum)))
    plt.legend()

def test_learning_rate_GD_OLS(X, y, eta, tune_func=no_tune, iterations=1000,):

    n_eta = len(eta)
    mse = np.zeros(n_eta)
    for i in range(n_eta):
        beta = OLS(X, y, inversion_method="GD", tuning_func=tune_func, eta=eta[i], iterations=iterations)
        pred = X @ beta 
        mse[i] = MSE(y, pred)
    
    plt.plot(eta, mse)
    argmin = np.argmin(mse)
    plt.scatter(eta[argmin], mse[argmin], label="min eta=%.2e mse=%.2f" %(eta[argmin], mse[argmin]))
    plt.ylim(0, 0.2)
    plt.legend()
    plt.show()

def test_learning_rate_GD_ridge(X, y, eta, lamda, tune_func=no_tune, iterations=1000):
    n_eta = len(eta)
    n_lamda = len(lamda)
    mse = np.zeros((n_eta, n_lamda))
    for i in range(n_eta):
        for j in range(n_lamda):
            beta = ridge_regression(X, y, inversion_method="GD", tuning_func=tune_func,lamda=lamda[j],  eta=eta[i], iterations=iterations)
            pred = X @ beta 
            mse[i, j] = MSE(y, pred)
    

    df = pd.DataFrame(mse, columns=np.log10(lamda), index=np.around(eta,decimals=2))
    sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"})
    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.show()

def test_func_1D(x, degree, noise):
    np.random.seed(100)
    a = np.random.rand(degree + 1)
    f_x = 0
    for i in range(degree + 1):
        f_x += a[i]*x**i

    return f_x + noise


def plot_3d_trisurf(x, y, z, scale_std=1, scale_mean=0, savename=None, azim=110, title=""):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title(title)
    surf = ax.plot_trisurf(x, y, z*scale_std + scale_mean, cmap=cm.coolwarm, linewidth=0.2, antialiased=False)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylabel(r"$y$")
    ax.view_init(azim=azim)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout(pad=1.5, w_pad=0.7, h_pad=0.2)
    if savename != None:
        plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')


def main():
    n = 100
    np.random.seed(100)

    x = np.linspace(-1, 1, n)
    noise = np.random.normal(0, 0.1, n)
    y = test_func_1D(x, 4, noise).reshape(n, 1)
    X = design_matrix_1D(x, 4)
    lamda = 0.01
    delta_mom = 0.3
    n_epochs = 50 
    batch_size = 5 
    eta = np.linspace(0.05, 0.7, 5)
    lamda = np.logspace(-8, -1, 8)
    #test_learning_rate_GD_OLS(X, y, eta, iterations=500, tune_func=no_tune)
    #test_learning_rate_GD_ridge(X, y, eta, lamda, iterations=1000, tune_func=ADAM)

    f_x = y.ravel() - noise
    plt.title("OLS")
    plt.plot(x, f_x,"k", linewidth=10)
    plt.plot(x, y,"k", linewidth=1)

    plot_inversion_compare(X, y, x, OLS, no_tune, delta_mom, lamda, n_epochs, batch_size, label="")
    plot_inversion_compare(X, y, x, OLS, AdaGrad, delta_mom, lamda, n_epochs, batch_size, label="AdaGrad")
    plot_inversion_compare(X, y, x, OLS, ADAM, delta_mom, lamda, n_epochs, batch_size, label="ADAM")
    plot_inversion_compare(X, y, x, OLS, RMSprop, delta_mom, lamda, n_epochs, batch_size, label="RMSprop")
    plt.show()
    plt.title("RIDGE")
    lamda = 0.00001
    plt.plot(x, f_x,"k", linewidth=10)
    plt.plot(x, y,"k", linewidth=1)


    plot_inversion_compare(X, y, x, ridge_regression, no_tune, delta_mom, lamda, n_epochs, batch_size, label="")
    plot_inversion_compare(X, y, x, ridge_regression, AdaGrad, delta_mom, lamda, n_epochs, batch_size, label="AdaGrad")
    plot_inversion_compare(X, y, x, ridge_regression, ADAM, delta_mom, lamda, n_epochs, batch_size, label="ADAM")
    plot_inversion_compare(X, y, x, ridge_regression, RMSprop, delta_mom, lamda, n_epochs, batch_size, label="RMSprop")
    plt.show()
if __name__ == "__main__":
    main()