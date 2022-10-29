


import matplotlib.pyplot as plt
import autograd.numpy as np
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
plt.rcParams.update({"font.size": 14})
import time

def no_tune(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1):
    return eta*grads + delta_mom*change

def no_tune_SGD(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1):
    eta = learning_schedule(epoch*m + start/batch_size, t0, t1)
    change = eta*grads + delta_mom*change 
    return change

def AdaGrad(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1):
    """
    ADAM tuning method
    """
    grad_square += grads**2
    alpha = eta/(1e-8 + np.sqrt(grad_square))
    change = alpha*grads + delta_mom*change
    
    return change

def RMSprop(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1):
    """
    RMSprop tuning method
    """
    s_t = grad_square
    grad_square += grads**2
    s_t = s_t*rho1 + grad_square*(1 - rho1)
    alpha = eta/(1e-8 + np.sqrt(s_t))
    change = alpha*grads + delta_mom*change
    return change

def ADAM(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1):
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
        batch_size=5, delta_mom=0, eta=0.1, t0=5, t1=50, rho1=0.9, rho2=0.99):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    if inversion_method == "GD":
        beta = GD(X, z, rho1, rho2, eta, iterations, delta_mom, tuning_func)
    elif inversion_method == "SGD":
        if tuning_func==no_tune:
            tuning_func = no_tune_SGD
        beta = SGD(X, z, n_epochs, batch_size, t0, t1, rho1, rho2, eta, iterations, delta_mom, tuning_func)
    else:
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z

    return beta

def ridge_regression(X, z, lamda):
    """
    Manual function for ridge regression to find beta
    Takes in:
    - X:        Design matrix of some degree
    - z:        Matching dataset
    - lamda:    chosen lamda for the Ridge regression
    returns:
    - beta
    """
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


def GD(X, z, rho1, rho2, eta, iterations, delta_mom, tuning_func, epoch=1, m=1, start=1, batch_size=1, t0=1, t1=1):
    """
    Stochastic Gradient method to find theta
    """
    n = X.shape[0] #
    N = X.shape[1]
    theta = np.random.randn(N, 1)
    grad_square = np.zeros((N, 1))
    change = 0
    grads = 0

    for i in range(iterations):
        grads_pre = grads
        grads = 2/n * X.T @ ((X @ theta) - z)
        change = tuning_func(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1)
        theta -= change 

    return theta



def SGD(X, z, n_epochs, batch_size, t0, t1, rho1, rho2, eta, iterations, delta_mom, tuning_func):
    """
    Stochastic Gradient decent method to find theta
    """
    n = X.shape[0] #
    N = X.shape[1]
    
    m = int(n/batch_size) # minibatches
    theta = np.random.randn(N,1)
    change = 0
    grads = 0

    for epoch in range(n_epochs):
        X_shuffle, y_shuffle = shuffle(X, z)
        grad_square = np.zeros((N, 1))
        for start in range(0, n, batch_size):
            X_batch = X_shuffle[start:start+batch_size]
            y_batch = y_shuffle[start:start+batch_size]
            grads_pre = grads
            grads = 2/batch_size * X_batch.T @ ((X_batch @ theta) - y_batch)
            change = tuning_func(grads_pre, grads, grad_square, rho1, rho2, eta, change, delta_mom, epoch, m, start, batch_size, t0, t1)
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

def learning_schedule(t, t0, t1):
    return t0 / (t + t1)


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


def design_matrix(x, degree):

    if len(x.shape) > 1:
        x = np.ravel(x)

    N = len(x)
    X = np.ones((N, degree+1))

    for i in range(1, degree+1):
        X[:,i] = x**i

    return X

def main():
    np.random.seed(100)
    n = 100
    x = np.linspace(-1, 1, n)

    a = np.random.rand(5)
    f_x = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4
    y = (f_x + np.random.normal(0, 0.1, n)).reshape(n, 1)
    X = design_matrix(x, 4)


    beta = OLS(X, y)
    pred = X @ beta

    beta_GD = OLS(X, y, inversion_method="GD")
    pred_GD = X @ beta_GD 

    beta_GD_momentum = OLS(X, y, inversion_method="GD", delta_mom=0.3)
    pred_GD_momentum = X @ beta_GD_momentum

    beta_SGD = OLS(X, y, inversion_method="SGD", n_epochs=50, batch_size=5)
    pred_SGD = X @ beta_SGD 

    beta_SGD_momentum = OLS(X, y, inversion_method="SGD", n_epochs=50, batch_size=5, delta_mom=0.3)
    pred_SGD_momentum = X @ beta_SGD_momentum

    beta_GD_ada = OLS(X, y, inversion_method="GD", tuning_func=AdaGrad)
    pred_GD_ada = X @ beta_GD_ada

    beta_GD_ada_mom = OLS(X, y, inversion_method="GD", delta_mom=0.3, tuning_func=AdaGrad)
    pred_GD_ada_mom = X @ beta_GD_ada_mom

    beta_SGD_ada = OLS(X, y, inversion_method="SGD", n_epochs=50, batch_size=5, tuning_func=AdaGrad)
    pred_SGD_ada = X @ beta_SGD_ada

    beta_SGD_ada_mom = OLS(X, y, inversion_method="SGD", n_epochs=50, batch_size=5, delta_mom=0.3, tuning_func=AdaGrad)
    pred_SGD_ada_mom = X @ beta_SGD_ada_mom
    
    beta_SGD_rms = OLS(X, y, inversion_method="SGD", n_epochs=50, batch_size=5, tuning_func=RMSprop)
    pred_SGD_rms = X @ beta_SGD_rms

    beta_SGD_rms_mom = OLS(X, y, inversion_method="SGD", n_epochs=50, batch_size=5, delta_mom=0.3, tuning_func=RMSprop)
    pred_SGD_rms_mom = X @ beta_SGD_rms_mom

    beta_GD_rms = OLS(X, y, inversion_method="GD", n_epochs=50, batch_size=5, tuning_func=RMSprop)
    pred_GD_rms = X @ beta_GD_rms

    beta_GD_rms_mom = OLS(X, y, inversion_method="GD", n_epochs=50, batch_size=5, delta_mom=0.3, tuning_func=RMSprop)
    pred_GD_rms_mom = X @ beta_GD_rms_mom

    beta_GD_ADAM = OLS(X, y, inversion_method="GD", n_epochs=50, batch_size=5, tuning_func=ADAM)
    pred_GD_ADAM = X @ beta_GD_ADAM

    beta_GD_ADAM_mom = OLS(X, y, inversion_method="GD", n_epochs=50, batch_size=5, delta_mom=0.3, tuning_func=ADAM)
    pred_GD_ADAM_mom = X @ beta_GD_ADAM_mom
    

    plt.plot(x, pred, label="pred %.4f" %(MSE(y,pred)))
    plt.plot(x, f_x, label="actual %.4f" %(MSE(y,f_x)))
    plt.plot(x, pred_SGD, label="SGD %.4f" %(MSE(y,pred_SGD)))
    plt.plot(x, pred_GD_ada, linestyle="--",label="GD ada %.4f" %(MSE(y,pred_GD_ada)))
    plt.plot(x, pred_SGD_ada, label="SGD ada %.4f" %(MSE(y,pred_SGD_ada)))
    plt.plot(x, pred_GD_ada_mom, linestyle="--",label="GD ada mom %.4f" %(MSE(y,pred_GD_ada_mom)))
    plt.plot(x, pred_SGD_ada_mom, label="SGD ada mom %.4f" %(MSE(y,pred_SGD_ada_mom)))
    plt.plot(x, pred_GD, "--", label="GD %.4f" %(MSE(y,pred_GD)))
    plt.plot(x, pred_GD_momentum, linestyle="dotted", label="GD mom %.4f" %(MSE(y, pred_GD_momentum)))
    plt.plot(x, pred_SGD_momentum, linestyle="dotted", label="SGD mom %.4f" %(MSE(y, pred_SGD_momentum)))
    plt.plot(x, pred_SGD_rms_mom, linestyle="dotted", label="SGD RMS mom %.4f" %(MSE(y, pred_SGD_rms_mom)))
    plt.plot(x, pred_SGD_rms, linestyle="dotted", label="SGD RMS  %.4f" %(MSE(y, pred_SGD_rms)))
    plt.plot(x, pred_GD_rms_mom, linestyle="dotted", label="GD RMS mom %.4f" %(MSE(y, pred_GD_rms_mom)))
    plt.plot(x, pred_GD_rms, linestyle="dotted", label="GD RMS  %.4f" %(MSE(y, pred_GD_rms)))
    plt.plot(x, pred_GD_ADAM_mom, linestyle="dotted", label="GD ADAM mom %.4f" %(MSE(y, pred_GD_ADAM_mom)))
    plt.plot(x, pred_GD_ADAM, linestyle="dotted", label="GD ADAM  %.4f" %(MSE(y, pred_GD_ADAM)))
    
    plt.legend()


    plt.show()
if __name__ == "__main__":
    main()