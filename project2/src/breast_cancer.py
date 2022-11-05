from NN import *
from functions import * 
from compare import *
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

def grid_search_eta_lambda(X_train, y_train, X_test, y_test, savename, neurons, epochs, batch_size, eta, lamda, mom, seed, init, act, cost, last_act, validate):
    val = np.zeros((len(eta), len(lamda)))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            print(lamda[j])
            NN = NeuralNetwork(X_train, y_train, neurons, epochs, batch_size, eta[i], lamda[j], moment=mom, cost=cost, seed=seed, initialize=init, activation=act, last_activation=last_act)
            NN.SGD()
            pred = NN.predict(X_test).ravel()
            if validate == "MSE":
                val[i, j] = (MSE(y_test, pred))
            elif validate == "ACC":
                pred = np.where(pred < 0.5, 0, 1) 
                val[i, j] = (accuracy(y_test, pred))

    plt.figure()
    df = pd.DataFrame(val, columns=np.log10(lamda), index=eta)
    if validate == "MSE":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.025, vmax=0.2)
    elif validate == "ACC":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$Accuracy$"}, vmin=0.8, vmax=1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')

def grid_search_layer_neurons(X_train, y_train, X_test, y_test, savename, n_layer, neurons, epochs, batch_size, eta, lamda, mom, seed, init, act, cost, last_act, validate):
    val = np.zeros((len(neurons), len(n_layer)))
    for i in range(len(neurons)):
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])
            print(neurons[i])
            NN = NeuralNetwork(X_train, y_train, neur, epochs, batch_size, eta, lamda, moment=mom, cost=cost, seed=seed, initialize=init, activation=act, last_activation=last_act)
            NN.SGD()
            pred = NN.predict(X_test).ravel()
            if validate == "MSE":
                val[i, j] = (MSE(y_test, pred))
            elif validate == "ACC":
                pred = np.where(pred < 0.5, 0, 1) 
                val[i, j] = (accuracy(y_test, pred))

    plt.figure()
    df = pd.DataFrame(val, columns=n_layer, index=neurons)

    if validate == "MSE":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.025, vmax=0.2)
    elif validate == "ACC":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$Accuracy$"}, vmin=0.8, vmax=1)
    plt.xlabel("layers")
    plt.ylabel("neurons per layer")
    plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')
      
def main():
    data = load_breast_cancer()
    outputs = data.target 
    X = choose_inputs(range(1,30), data)
    y = outputs.reshape(len(outputs), 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    std = np.std(X_train, axis=0)
    mean = np.mean(X_train, axis=0)
    X_train = (X_train - mean) / std 
    X_test = (X_test - mean) / std 

    #X_train, mn, mx = min_max_scale(X_train)
    #X_test, mn, mx = min_max_scale(X_test)
    neurons = [40, 40, 40]
    eta = 0.1
    lamda = 1e-3
    epochs = 200
    n_train = np.shape(X_train)
    print(n_train)
    batch_size = 30
    n_layers = 2
    n_neuron = 100
    input_size = X_train.shape[1]
    
    #NN_keras = NN_model(input_size, n_layers, n_neuron, eta, lamda)
    #NN_keras.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    #test = NN_keras.evaluate(X_test,y_test)[1]
    #print(test)
    """
    NN = NeuralNetwork(X_train,
                       y_train,
                       neurons,
                       epochs,
                       batch_size,
                       eta,
                       lamda=lamda,
                       moment=0,
                       seed=100,
                       initialize="zeros",
                       activation="sigmoid",
                       cost="cross entropy",
                       last_activation="sigmoid")
    NN.SGD()
    probs = NN.predict(X_test)
    pred = np.where(probs < 0.5, 0, 1) 
    print(accuracy(y_test, pred))
    """

    savename = "cancer_eta_lmb_test"
    eta = np.logspace(-3, 0, 4)
    lamda = np.logspace(-5,0, 6)
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
                           act="relu",
                           cost="cross entropy",
                           last_act="sigmoid",
                           validate="ACC")
    plt.show()
    savename = "cancer_L_n_test"
    neurons = np.array([10,20,30,40,50])
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
                              act="relu",
                              cost="cross entropy",
                              last_act="sigmoid",
                              validate="ACC")
    plt.show()

if __name__ == "__main__":
    main()