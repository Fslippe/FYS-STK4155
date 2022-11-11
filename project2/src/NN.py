import numpy as np 
import random 
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error


class NeuralNetwork :

    def __init__(self, X, Y, neurons, epochs, batch_size, eta, lamda=0., moment=0, seed=None, initialize_weight="random", initialize_bias="zeros", activation="sigmoid", cost="error", last_activation=None):
        if seed != None:
            np.random.seed(seed)
            self.seed = seed 
        else:
            self.seed = 0
        self.X = X
        self.Y = Y 
        self.n_inputs = X.shape[0]
        self.n_outputs = Y.shape[0]
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.neurons = neurons #Neurons in each hidden layer 
        self.n_layers = len(neurons)
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.eta = eta 
        self.lamda = lamda
        self.moment = moment
        self.initialize_weight = initialize_weight
        self.initialize_bias = initialize_bias
        self.initialize_arrays()
        self.last_activation = last_activation
        self.activation = activation
        # Activation functions
        self.act = self.activation_function(activation)
        self.act_grad = self.activation_gradient(activation)

        if last_activation == None:
            self.act_out = self.act
            self.act_grad_out = self.act_grad
        else:
            self.act_out = self.activation_function(last_activation)
            self.act_grad_out = self.activation_gradient(last_activation)


        # Cost functions
        if cost == "error":
            self.cost = self.error 
            self.cost_grad = self.error_grad
        elif cost == "cross entropy":
            self.cost = self.cross_entropy 
            self.cost_grad = self.cross_entropy_grad

    def activation_function(self, s):
        if s == "relu":
            return self.relu
        elif s == "lrelu":
            return self.l_relu
        elif s == "softmax":
            return self.softmax
        elif s == "none":
            return self.none
        else:
            return self.sigmoid 
        
    def activation_gradient(self, s):
        if s == "relu":
            return self.relu_grad
        elif s == "lrelu":
            return self.l_relu_grad
        elif s == "softmax":
            return self.softmax_grad
        elif s == "none":
            return self.none_grad
        else:
            return self.sigmoid_grad

    def initialize_arrays(self):
        # Initialize Weight
        self.weight = np.zeros(self.n_layers +1, dtype=object)
        self.bias = np.zeros(self.n_layers +1, dtype=object)

        if self.initialize_weight == "zeros":
            self.weight[0] = np.zeros((int(self.neurons[0]), self.input_dim))*np.sqrt(2) / np.sqrt(self.input_dim)
            for i in range(1, self.n_layers):
                self.weight[i] = np.zeros((int(self.neurons[i]), int(self.neurons[i-1])))*np.sqrt(2) / np.sqrt(self.neurons[i-1])
            self.weight[-1] = np.zeros((self.output_dim, int(self.neurons[-1])))*np.sqrt(2) / np.sqrt(self.neurons[-1])

        elif self.initialize_weight == "random":
            self.weight[0] = np.random.randn(int(self.neurons[0]), self.input_dim)
            for i in range(1, self.n_layers):
                self.weight[i] = np.random.randn(int(self.neurons[i]), int(self.neurons[i-1]))
            self.weight[-1] = np.random.randn(self.output_dim, int(self.neurons[-1]))

            
        elif self.initialize_weight == "random scaled":
            self.weight[0] = np.random.randn(int(self.neurons[0]), self.input_dim) * np.sqrt(2) / np.sqrt(self.input_dim)
            for i in range(1, self.n_layers):
                self.weight[i] = np.random.randn(int(self.neurons[i]), int(self.neurons[i-1])) * np.sqrt(2) / np.sqrt(self.neurons[i-1])
            self.weight[-1] = np.random.randn(self.output_dim, int(self.neurons[-1])) * np.sqrt(2) / np.sqrt(self.neurons[-1])

        # Initialize Bias
        if self.initialize_bias == "zeros":   
            for i in range(self.n_layers):
                self.bias[i] = np.zeros(int(self.neurons[i])) + 0.01
            self.bias[-1] = np.zeros((self.output_dim)) + 0.01      

        elif self.initialize_bias == "random":
            for i in range(self.n_layers):
                self.bias[i] = np.random.randn(int(self.neurons[i]))
            self.bias[-1] = np.random.randn(self.output_dim) 

        # Initialize
        self.a_l = np.zeros(self.n_layers + 1, dtype=object)
        self.z_l = np.zeros(self.n_layers + 1, dtype=object)
        self.error = np.zeros(self.n_layers + 1, dtype=object)
        for i in range(self.n_layers):
            self.a_l[i] = np.zeros((self.batch_size, int(self.neurons[i])))
            self.z_l[i] = np.zeros((self.batch_size, int(self.neurons[i])))
        
        #output layer
        self.a_l[-1] = np.zeros((self.batch_size, self.output_dim))
        self.z_l[-1] = np.zeros((self.batch_size, self.output_dim))    
        
        # Initialize gradients
        self.weight_grad = np.zeros(self.n_layers + 1, dtype=object)
        self.bias_grad = np.zeros(self.n_layers + 1, dtype=object)
        for i in range(self.n_layers):
            self.weight_grad[i] = np.zeros(self.weight[i].shape)
            self.bias_grad[i] = np.zeros(self.bias[i].shape)       


    def feed_forward(self):
        self.z_l[0] = self.X_batch @ self.weight[0].T + self.bias[0]
        self.a_l[0] = self.act(self.z_l[0])

        for i in range(1, self.n_layers):
            self.z_l[i] = self.a_l[i-1] @ self.weight[i].T + self.bias[i]
            self.a_l[i] = self.act(self.z_l[i])

        self.z_l[-1] = self.a_l[-2] @ self.weight[-1].T + self.bias[-1]

        self.a_l[-1] = self.act_out(self.z_l[-1])


    def SGD(self):
        i = self.seed
        for epoch in range(self.epochs):
            X_shuffle, Y_shuffle = shuffle(self.X, self.Y, random_state=i)
            i += 1
            for start in range(0, self.n_inputs, self.batch_size):
                self.X_batch = X_shuffle[start:start+self.batch_size]
                self.Y_batch = Y_shuffle[start:start+self.batch_size]
                self.feed_forward()
                self.backprop()
                self.update()

    def backprop(self):
        for i in range(self.n_layers):
            self.error[i] = np.zeros((self.batch_size, int(self.neurons[i])))
        self.error[-1] = np.zeros((self.batch_size, self.output_dim))

        if self.last_activation == "softmax" or self.activation == "relu" or self.activation == "lrelu":
            self.error[-1] = self.a_l[-1] - self.Y_batch
        else:
            self.error[-1] = self.cost_grad(self.a_l[-1], self.Y_batch) * self.act_grad_out(self.z_l[-1])
        #backpropagation: 
        for i in range(self.n_layers - 1, -1, -1):
            self.error[i] = (self.error[i + 1] @ self.weight[i + 1]) * self.act_grad(self.z_l[i])
        
        self.weight_grad[0] = self.error[0].T @ self.X_batch 
        self.bias_grad[0] = np.sum(self.error[0], axis=0) 

        for i in range(1, self.n_layers+1):
            self.weight_grad[i] = self.error[i].T @ self.a_l[i-1]  + self.lamda*self.weight[i]
            self.bias_grad[i] = np.sum(self.error[i], axis=0) + self.lamda*self.bias[i]
 
    def update(self):
        for i in range(self.n_layers + 1):
            delta_w = self.weight[i] * self.moment - self.eta / self.batch_size *self.weight_grad[i]
            delta_b = self.bias[i] *self.moment - self.eta / self.batch_size * self.bias_grad[i]
            self.weight[i] += delta_w
            self.bias[i] += delta_b
    

    def feed_forward_out(self, x):
        z_o = x @ self.weight[0].T + self.bias[0]
        a_o = self.act(z_o)#self.act_out(z_o)

        for i in range(1, self.n_layers):
            z_o = a_o @ self.weight[i].T + self.bias[i]
            a_o = self.act(z_o)

        z_o = a_o @ self.weight[-1].T + self.bias[-1]
        a_o = self.act_out(z_o)

        return a_o

    def predict(self, x):
        a_o = self.feed_forward_out(x)
        return a_o

    def predict_accuracy(self, X, t):
        pred = self.predict(X)
        accuracy = np.sum(np.where(pred < 0.5, 0, 1) == t, axis=0) / t.shape[0]
        return accuracy

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x)) 
    
    def relu(self, x):
        return np.where(x > 0, x, 0)
    
    def relu_grad(self, x):
        return np.where(x < 0, 0, 1)
    
    def l_relu(self, x):
        return np.where(x >= 0, x, 0.01*x)
    
    def l_relu_grad(self, x):
        return np.where(x >= 0, 1, 0.01)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def softmax_grad(self, x):
        return self.softmax(x) - self.softmax(x)**2

    def none(self, x):
        return x

    def none_grad(self, x):
        return 0

    def error(self, a, y):
        return (a - y)**2 / 2

    def error_grad(self, a, y):
        return a - y

    def cross_entropy(self, a, y):
        return -(y * np.log(a)) + (1-y) * np.log(1-a)#-np.sum(y* np.log(a) + (1 - y) * np.log(1 - a))

    def cross_entropy_grad(self, a, y):
        return a-y#-(y - a) / (a - a**2)


def design_matrix_1D(x, degree):
    N = len(x)
    X = np.ones((N, degree+1))

    for i in range(1, degree+1):
        X[:,i] = x**i

    return X

def test_func_1D(x, degree, noise):
    np.random.seed(100)
    a = np.random.rand(degree + 1)
    f_x = 0
    for i in range(degree + 1):
        f_x += a[i]*x**i

    return f_x + noise

def main():
    """Simple test of NN using 1D func to check for problems"""
    np.random.seed(100)
    n = 1000
    degree = 4
    x = np.linspace(0, 1, n)
    noise = np.random.normal(0, 0.1, n)
    Y = test_func_1D(x, 4, noise).reshape(n, 1) 
    Y = Y / np.max(Y)
    X = design_matrix_1D(x, degree)
    x = x.reshape(n, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    print(np.shape(X_train), np.shape(Y_train))
    neurons = np.array([300, 300, 300]) # in hidden layers 
    epochs = 30
    batch_size = 25
    eta = 0.2
    NN = NeuralNetwork(X_train, Y_train, neurons, epochs, batch_size, eta)
    NN.SGD()
    pred = (NN.predict(X_test))*np.max(Y)
    print(mean_squared_error(Y_test, pred))
    plt.plot(Y_test)
    plt.plot(pred.ravel())
    plt.show()
if __name__ == "__main__":
    main()