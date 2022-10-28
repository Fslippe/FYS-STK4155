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

def OLS(X, z, inversion_method="", iterations=1000, n_epochs=50, batch_size=5):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    if inversion_method == "SG":
        beta = SG(X, z, iterations)
    if inversion_method == "SGD":
        beta = SGD(X, z, n_epochs, batch_size, t0=5, t1=50)
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

def SG(X, y, iterations):
    """
    Stochastic Gradient method to find theta
    """
    n = X.shape[0] #
    N = X.shape[1]
    H = 2/n * X.T @ X 
    eigval, eigvec = np.linalg.eig(H)
    eta = 1 / np.max(eigval)
    theta = np.random.randn(N, 1)

    for i in range(iterations):
        grads = 2/n * X.T @ ((X @ theta) - y)
        theta -= eta*grads 

    return theta

def SGD(X, y, n_epochs, batch_size, t0=5, t1=50):
    """
    Stochastic Gradient decent method to find theta
    """
    n = X.shape[0] #
    N = X.shape[1]

    m = int(n/batch_size) #minibatches
    H = 2/n *X.T @ X
    eigval, eigvec = np.linalg.eig(H)
    eta = 1 / np.max(eigval)
    theta = np.random.randn(N,1)
    
    for epoch in range(n_epochs):
        X_shuffle, y_shuffle = shuffle(X, y)

        for start in range(0, n, batch_size):
            X_batch = X_shuffle[start:start+batch_size]
            y_batch = y_shuffle[start:start+batch_size]
            grads = 2/batch_size * X_batch.T @ ((X_batch @ theta) - y_batch)
            eta = learning_schedule(epoch*m + start/batch_size, t0, t1)
            theta -= eta*grads 
    
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

def GD_momentum(X, y, iterations=1000):
    n = X.shape[0] 
    N = X.shape[1]
    H = 2/n *X.T @ X 
    y = y.reshape(n, 1)
    H_inv = np.linalg.pinv(H)
    beta = np.random.randn(N,1)
    train_grad = grad(C_OLS, 2)
    print(np.shape(beta))
    print(np.shape(X))
    print(np.shape(y))
    print(np.shape(H_inv))

    for i in range(iterations):
        grads = train_grad(X, y, beta)
        beta -= H_inv @ grads 

    return beta

def SGD_momentum(X, y, n_epochs, batch_size, t0=5, t1=50, delta_mom=0.3):
    """
    Stochastic Gradient decent method to find theta
    """
    n = X.shape[0] 
    N = X.shape[1]
    y = y.reshape(n, 1)

    m = int(n/batch_size) #minibatches
    H = 2/n *X.T @ X
    eigval, eigvec = np.linalg.eig(H)
    eta = 1 / np.max(eigval)
    theta = np.random.randn(N,1)
    change = 0 

    train_grad = grad(C_OLS, 2)

    for epoch in range(n_epochs):
        X_shuffle, y_shuffle = shuffle(X, y)
        for start in range(0, n, batch_size):
            X_batch = X_shuffle[start:start+batch_size]
            y_batch = y_shuffle[start:start+batch_size]
            grads = 1/batch_size * train_grad(X_batch, y_batch, theta, n_div=False)
            eta = learning_schedule(epoch*m + start/batch_size, t0, t1)
            new_change = eta*grads+delta_mom*change
            # take a step
            theta -= new_change
            # save the change
            change = new_change
    return theta

def SGD_mom_ex(X, y):
    n = X.shape[0]
    H = (2.0/n)* X.T @ X
    EigValues, EigVectors = np.linalg.eig(H)
    print(f"Eigenvalues of Hessian Matrix:{EigValues}")

    theta = np.random.randn(2,1)
    eta = 1.0/np.max(EigValues)
    Niterations = 100

    # Note that we request the derivative wrt third argument (theta, 2 here)
    training_gradient = grad(C_OLS,2)
    n_epochs = 50
    M = 5   #size of each minibatch
    m = int(n/M) #number of minibatches
    t0, t1 = 5, 50
    def learning_schedule(t):
        return t0/(t+t1)

    theta = np.random.randn(3,1)

    change = 0.0
    delta_momentum = 0.3

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]

            gradients = (1.0/M)*training_gradient(xi, yi, theta, False)
            eta = learning_schedule(epoch*m+i)
            # calculate update
            new_change = eta*gradients+delta_momentum*change
            # take a step
            theta -= new_change
            # save the change
            change = new_change
    print("theta from own sdg with momentum")
    print(theta)
    return theta

def learning_schedule(t, t0, t1):
    return t0 / (t + t1)

def ex(X, y, t0=5, t1=50):
    n = X.shape[0] #
    N = X.shape[1]
    n_epochs = 50
    M = 5 #size of each minibatch
    m = int(n/M) #number of minibatches
    t0, t1 = 5, 50
    theta = np.random.randn(N,1)

    for epoch in range(n_epochs):
    # Can you figure out a better way of setting up the contributions to each batch?
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)* xi.T @ ((xi @ theta)-yi)
            eta = learning_schedule(epoch*m+i, t0, t1)
            theta = theta - eta*gradients
    return theta

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
    a = np.random.rand(3)
    f_x = a[0] + a[1]*x + a[2]*x**2
    y = f_x.reshape(n, 1)
    X = design_matrix(x, 2)

    beta = OLS(X, y)
    beta_SG = OLS(X, y, inversion_method="SG")
    beta_SGD = OLS(X, y, inversion_method="SGD")
    beta_GD_momentum = GD_momentum(X, y)
    beta_SGD_momentum = SGD_momentum(X, y, n_epochs=50, batch_size=5)
    beta_SGD_mom_ex = SGD_mom_ex(X, y)


    pred = X @ beta
    pred_SG = X @ beta_SG 
    pred_SGD = X @ beta_SGD 
    pred_ex = X @ ex(X, y)
    pred_GD_momentum = X @ beta_GD_momentum
    pred_SGD_momentum = X @ beta_SGD_momentum
    pred_SGD_mom_ex = X @ beta_SGD_mom_ex


    plt.plot(x, pred, label="pred")
    #plt.plot(x, pred_SGD, label="SGD")
    #plt.plot(x, pred_ex, label="ex")
    plt.plot(x, pred_GD_momentum, "--", label="GD mom")
    plt.plot(x, pred_SGD_momentum, linestyle="dotted", label="SGD mom")
    plt.plot(x, pred_SGD_mom_ex, label="ex")
    
    plt.legend()


    plt.show()
if __name__ == "__main__":
    main()