from NN import *
from functions import * 
from franke_compare import *
from sklearn.datasets import load_breast_cancer
from grid_search import *
from gradient_decent import *

def choose_inputs(idx, data):
    inputs = data.data 
    labels = data.feature_names
    temp = np.zeros(len(idx), dtype=object)
    for i in range(len(idx)):
        temp[i] = np.reshape(inputs[:,idx[i]],(len(inputs[:,idx[i]]),1))

    X = np.hstack((temp))
    return X

def NN_model(n_hidden_neurons, epochs, lamda, eta):
    dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lamda, learning_rate_init=eta, max_iter=epochs)
    dnn.fit(X_train, Y_train)


def main():
    data = load_breast_cancer()
    outputs = data.target 
    X = choose_inputs(range(1,30), data)
    y = outputs.reshape(len(outputs), 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5) #10
    std = np.std(X_train, axis=0)
    mean = np.mean(X_train, axis=0)

    # Standard scaler input data
    X_train = (X_train - mean) / std 
    X_test = (X_test - mean) / std 

    savename = "cancer_eta_lmb_relu"
    eta = np.array([0.001, 0.001])
    lamda = np.logspace(-2, 0, 3)
    neurons = [50]
    epochs = 200
    n_train = np.shape(X_train)
    batch_size = 25
    grid_search_eta_lambda(X_train,
                        y_train,
                        X_test,
                        y_test,
                        savename,
                        neurons,
                        epochs,
                        batch_size,
                        eta,
                        lamda,
                        mom=0,
                        seed=100,
                        init="random scaled",
                        act="relu",
                        cost="cross entropy",
                        last_act="sigmoid",
                        validate="ACC")
    plt.show()
    n_layers = 2
    n_neuron = 100
    input_size = X_train.shape[1]
    
    logreg_grid = False 
    grid_NN_lmb_eta = False
    if logreg_grid:
        eta = np.logspace(-4, 0, 5)
        lamda = np.logspace(-6, 0, 7)
        method = ["none", "ADAM", "Adagrad", "RMSprop"]
        #method = ["ADAM"]
        for m in method:
            grid_search_logreg(X_train,
                            y_train,
                            X_test,
                            y_test,
                            gradient="SGD",
                            lamda=lamda,
                            eta=eta,
                            method=m,
                            iterations=100,
                            batch_size= 10,
                            moment=0,
                            savename="logreg_%s" %(m))
        plt.show()
    #NN_keras = NN_model(input_size, n_layers, n_neuron, eta, lamda)
    #NN_keras.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    #test = NN_keras.evaluate(X_test,y_test)[1]
    #print(test)

    neurons = [40, 40, 40]
    eta = 0.1
    lamda = 1e-3
    epochs = 200
    n_train = np.shape(X_train)
    batch_size = 25

    """eta-lambda grid search own Neural Network"""
    if grid_NN_lmb_eta:
        savename = "cancer_eta_lmb"
        eta = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
        lamda = np.logspace(-2, 0, 5)
        grid_search_eta_lambda(X_train,
                            y_train,
                            X_test,
                            y_test,
                            savename,
                            neurons,
                            epochs,
                            batch_size,
                            eta,
                            lamda,
                            mom=0,
                            seed=100,
                            init="zeros",
                            act="sigmoid",
                            cost="cross entropy",
                            last_act="sigmoid",
                            validate="ACC")
        plt.show()

    """layer-neurons grid search own Neural Network"""
    savename = "cancer_L_n_test"
    neurons = np.array([10,40,60,80,100])
    n_layer = np.array([1,2,3,4,5])
    grid_search_layer_neurons(X_train,
                              y_train,
                              X_test,
                              y_test,
                              savename,
                              n_layer,
                              neurons,
                              epochs,
                              batch_size,
                              eta=0.1,
                              lamda=1e-1,
                              mom=0,
                              seed=100,
                              init="zeros",
                              act="sigmoid",
                              cost="cross entropy",
                              last_act="sigmoid",
                              validate="ACC")
    plt.show()
if __name__ == "__main__":
    main()