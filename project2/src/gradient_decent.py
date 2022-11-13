from functions import *
from autograd import grad, elementwise_grad

class GradientDescent:
    def __init__(self, cost, method, eta=0.1, moment=0, lamda=0, iterations=1000, rho_1=0.9, rho_2=0.99, eps=1e-8, seed=None):
        self.lamda = lamda 
        self.moment = moment
        self.iter = iterations
        self.rho_1 = rho_1 
        self.rho_2 = rho_2
        self.eps = eps
        self.grad_square = 0
        self.eta = eta
        self.s = 0
        self.m = 0
        self.t = 0
        self.delta = 0
        self.method_s = method
        if seed != None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = np.random.randint(1000) 

        if cost == "OLS":
            self.gradient = grad(self.cost_OLS, 2)
        elif cost == "RIDGE":
            self.gradient = grad(self.cost_Ridge, 2)
        elif cost =="LOGREG":
            self.gradient = self.logreg_grad

        if method == "RMSprop":
            self.method = self.RMSprop 
        elif method == "ADAM":
            self.method = self.ADAM 
        elif method == "AdaGrad":
            self.method = self.AdaGrad
        else:
            self.method = self.constant

    def SGD(self, X, y, batch_size, eval=False, X_test=np.zeros(1)):
        n, N = np.shape(X)
        self.theta = np.random.randn(N, 1)
        random = self.seed
        if eval == True:
            pred = np.zeros((self.iter, X_test.shape[0]))
            i = 0
        for epoch in range(self.iter):
            self.t += 1
            self.grad_square = 0
            X_shuffle, y_shuffle = shuffle(X, y, random_state=random)
            random += 1
            for start in range(0, n, batch_size):
                X_batch = X_shuffle[start:start+batch_size]
                y_batch = y_shuffle[start:start+batch_size]    
                self.grads = self.gradient(X_batch, y_batch, self.theta)
                self.method()

            if eval == True:
                pred[i] = X_test @ self.theta[:,0]
                i += 1

        if eval == True:
            return pred
        else:
            return self.theta

    def GD(self, X, y, eval=False, X_test=np.zeros(1)):
        n, N = np.shape(X)
        self.theta = np.random.randn(N, 1)
        self.t = 0
        if eval == True:
            pred = np.zeros((self.iter, X_test.shape[0]))

        for i in range(self.iter):
            self.t += 1
            if self.method_s != "AdaGrad":
                self.grad_square = 0
            self.grads = self.gradient(X, y, self.theta) 
            self.method()
            
            if eval == True:
                pred[i] = X_test @ self.theta[:,0]

        if eval == True:
            return pred
        else:
            return self.theta

    def ADAM(self):
        self.grad_square += self.grads**2
        self.m = self.rho_1*self.m + (1-self.rho_1)*self.grads
        self.s = self.rho_2*self.s + (1-self.rho_2)*self.grad_square
        m = self.m / (1-self.rho_1**self.t)
        s = self.s / (1-self.rho_2**self.t)
        self.theta -= self.eta * m / (np.sqrt(s) + self.eps)
    
    def RMSprop(self):
        self.grad_square += self.grads**2
        self.s = self.rho_1*self.s + (1-self.rho_1)*self.grad_square 
        self.theta -= self.eta * self.grads / (np.sqrt(self.s) + self.eps)
    
    def AdaGrad(self):
        self.grad_square += self.grads**2
        self.theta -= self.eta / (np.sqrt(self.grad_square) + self.eps) *self.grads

    def constant(self):
        self.delta = self.eta*self.grads + self.moment*self.delta
        self.theta -= self.delta

    def predict_accuracy(self, X, t):
        pred = np.exp(X @ self.theta) / (1+ np.exp(X @ self.theta))
        accuracy = np.sum(np.where(pred < 0.5, 0, 1) == t, axis=0) / t.shape[0]
        return accuracy


    def cost_OLS(self, X, y, beta):
        n = X.shape[0]
        return np.sum((X @ beta - y)**2) /n 

    def cost_Ridge(self, X, y, beta):
        n = X.shape[0]
        return (np.sum((X @ beta - y)**2) + self.lamda*np.sum(beta**2)) / n

    def logreg_grad(self, X, y, beta):
        gradient = - X.T @ (y - np.exp(X @ beta) / (1+ np.exp(X @ beta))) + 2*self.lamda*beta 
        return gradient 

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

    f_x = y.ravel() - noise
    plt.title("OLS")
    plt.plot(x, f_x,"k", linewidth=10)
    plt.plot(x, y,"k", linewidth=1)
    G = GradientDescent(cost="RIDGE", method="", eta = 0.1, moment=0, lamda=0, iterations=1000)
    pred_GD = G.GD(X, y)
    pred_SGD = G.SGD(X, y, 50)
    pred_GD = X @ pred_GD 
    pred_SGD = X @ pred_SGD 


    plt.plot(x, pred_GD, label="GD")
    plt.plot(x, pred_SGD, label="SGD")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()