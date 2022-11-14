from functions import *
from autograd import grad, elementwise_grad

class GradientDescent:
    """
    Gradient descent method for GD and SGD using different tuning methods and momentum

    - cost                  Cost function to use
    - method                Tuning method
    - eta=0.1               learning rate
    - moment=0              momentum
    - lamda=0               L2 norm lambda 
    - iterations=1000       iterations
    - rho_1=0.9             rho_1 used in RMSprop and ADAM
    - rho_2=0.99            rho_" used in ADAM
    - eps=1e-8              value to avoid divide by 0
    - seed=None             Seed for initialization of random values
    """   

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
        """
        Stochastic gradient descent method
        - X             Train input matrix
        - y             Train target data 
        - batch_size    batch size 
        - eval          if True needs also X_test as input. perform prediction for test data at every iteration
        - X_test        test data to perform prediction at every iteration
        """
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
        """
        Gradient descent method
        - X             Train input matrix
        - y             Train target data 
        - eval          if True needs also X_test as input. perform prediction for test data at every iteration
        - X_test        test data to perform prediction at every iteration
        """

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
        """ADAM tuning method"""
        self.grad_square += self.grads**2
        self.m = self.rho_1*self.m + (1-self.rho_1)*self.grads
        self.s = self.rho_2*self.s + (1-self.rho_2)*self.grad_square
        m = self.m / (1-self.rho_1**self.t)
        s = self.s / (1-self.rho_2**self.t)
        self.theta -= self.eta * m / (np.sqrt(s) + self.eps)
    
    def RMSprop(self):
        """RMSprop tuning method"""
        self.grad_square += self.grads**2
        self.s = self.rho_1*self.s + (1-self.rho_1)*self.grad_square 
        self.theta -= self.eta * self.grads / (np.sqrt(self.s) + self.eps)
    
    def AdaGrad(self):
        """AdaGrad tuning method"""
        self.grad_square += self.grads**2
        self.theta -= self.eta / (np.sqrt(self.grad_square) + self.eps) *self.grads

    def constant(self):
        """No tuning method - can use momentum"""
        self.delta = self.eta*self.grads + self.moment*self.delta
        self.theta -= self.delta

    def predict_accuracy(self, X, t):
        """
        Predict accuracy for test and target data 
        - X     input matrix
        - t     target matrix
        returns
        - accuracy
        """
        pred = np.exp(X @ self.theta) / (1+ np.exp(X @ self.theta))
        accuracy = np.sum(np.where(pred < 0.5, 0, 1) == t, axis=0) / t.shape[0]
        return accuracy


    def cost_OLS(self, X, y, beta):
        """OLS cost function for any given input X, target y, and parameter beta"""
        n = X.shape[0]
        return np.sum((X @ beta - y)**2) /n 

    def cost_Ridge(self, X, y, beta):
        """Ridge cost function for any given input X, target y, and parameter beta"""
        n = X.shape[0]
        return (np.sum((X @ beta - y)**2) + self.lamda*np.sum(beta**2)) / n

    def logreg_grad(self, X, y, beta):
        """
        Gradient of the logistic cost function
        for any given input X, target y, and parameter beta
        """
        gradient = - X.T @ (y - np.exp(X @ beta) / (1+ np.exp(X @ beta))) + 2*self.lamda*beta 
        return gradient 