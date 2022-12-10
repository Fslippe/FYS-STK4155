from NN import *
from functions import *
from franke_compare import *
from gradient_decent import *
plt.rcParams.update({'font.size': 11})


def neuron_array(layers, neurons):
    """
    Set up neuron list of each hidden layer depending on input
    - layers    number of hidden layers 
    - neurons   number of neurons per hidden layer
    returns
    - n         list containing neurons at each layer 
    """
    n = np.zeros(layers)
    for i in range(layers):
        n[i] = int(neurons)

    return n


def grid_search_eta_lambda(X_train, y_train, X_test, y_test, savename, neurons, epochs, batch_size, eta, lamda, mom, seed, init, act, cost, last_act, validate, mn=0, mx=1):
    """
    Perform grid search for different eta and lambda and plot a heatmap
    - X_train:          train design matrix
    - y_train           train target data
    - X_test:           test design matrix
    - y_test            test target data 
    - savename          savename of heatmap produced
    - neurons           list of neurons for each hidden layer
    - epochs            iterations to perform in SGD
    - batch_size        batch size for SGD
    - eta               learning rate
    - lamda             array of lambdas to test
    - mom               add momentum to algorithm
    - seed              seed of random values computed in NN
    - init              initialization of weights in NN
    - act               Activation function of hidden layers
    - cost              cost function of NN
    - last_act          Activation function for output layer
    - validate          Validation for the heatmap, MSE, R2, ACC
    - mn                min of data before scaling
    - mx                max of data before scaling
    - validate          how to validate performance, MSE, ACC or R2
    """
    plt.rcParams.update({'font.size': 11})
    val = np.zeros((len(eta), len(lamda)))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            print(lamda[j])
            NN = NeuralNetwork(X_train, y_train, neurons, epochs, batch_size,
                               eta[i], lamda[j], moment=mom, cost=cost, seed=seed, initialize_weight=init, activation=act, last_activation=last_act)
            NN.SGD()
            pred = NN.predict(X_test).ravel()
            if validate == "MSE":
                val[i, j] = (MSE(y_test, min_max_unscale(pred, mn, mx)))
            if validate == "R2":
                val[i, j] = (R2(y_test, min_max_unscale(pred, mn, mx)))
            elif validate == "ACC":
                pred = np.where(pred < 0.5, 0, 1)
                val[i, j] = (accuracy(y_test, pred))

    plt.figure()
    df = pd.DataFrame(val, columns=np.log10(lamda), index=eta)
    if validate == "MSE":
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$MSE$"}, vmin=0.03, vmax=0.1)
    elif validate == "R2":
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$R^2$"}, vmin=0.5, vmax=1)
    elif validate == "ACC":
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" % (savename), dpi=300, bbox_inches='tight')


def grid_search_layer_neurons(X_train, y_train, X_test, y_test, savename, n_layer, neurons, epochs, batch_size, eta, lamda, mom, seed, init, act, cost, last_act, validate, mn=0, mx=1):
    """
    Perform grid search for different number of layers and neurons and plot a heatmap
    - X_train:          train design matrix
    - y_train           train target data
    - X_test:           test design matrix
    - y_test            test target data 
    - savename          savename of heatmap produced
    - n_layers          array of number of layers to test
    - neurons           array of number of neurons to test
    - epochs            iterations to perform in SGD
    - batch_size        batch size for SGD
    - eta               learning rate
    - lamda             lambda L2 norm
    - mom               add momentum to algorithm
    - seed              seed of random values computed in NN
    - init              initialization of weights in NN
    - act               Activation function of hidden layers
    - cost              cost function of NN
    - last_act          Activation function for output layer
    - validate          Validation for the heatmap, MSE, R2, ACC
    - mn                min of data before scaling
    - mx                max of data before scaling
    - validate          how to validate performance, MSE, ACC or R2
    """
    plt.rcParams.update({'font.size': 11})
    val = np.zeros((len(neurons), len(n_layer)))
    for i in range(len(neurons)):
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])
            print(neurons[i])
            NN = NeuralNetwork(X_train, y_train, neur, epochs, batch_size, eta, lamda, moment=mom,
                               cost=cost, seed=seed, initialize_weight=init, activation=act, last_activation=last_act)
            NN.SGD()
            if validate == "MSE":
                pred = NN.predict(X_test).ravel()
                val[i, j] = (MSE(y_test, min_max_unscale(pred, mn, mx)))
            if validate == "R2":
                pred = NN.predict(X_test).ravel()
                val[i, j] = (R2(y_test, min_max_unscale(pred, mn, mx)))
            elif validate == "ACC":
                val[i, j] = NN.predict_accuracy(X_test, y_test)

    plt.figure()
    df = pd.DataFrame(val, columns=n_layer, index=neurons)

    if validate == "MSE":
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$MSE$"}, vmin=0.03, vmax=0.1)
    elif validate == "R2":
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$R^2$"}, vmin=0.5, vmax=1)
    elif validate == "ACC":
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)
    plt.xlabel("layers")
    plt.ylabel("neurons per layer")
    plt.savefig("../figures/%s.png" % (savename), dpi=300, bbox_inches='tight')


def grid_search_logreg(X_train, y_train, X_test, y_test, gradient, lamda, eta, method, iterations, batch_size, mom, savename):
    """
    Perform logistic regression grid search for different eta and lambda and plot a heatmap
    - X_train:          train design matrix
    - y_train           train target data
    - X_test:           test design matrix
    - y_test            test target data 
    - gradient:         gradient descent method (GD or SGD)
    - lamda             array of lambdas to test
    - eta               array of learning rates to test
    - iterations        iterations to perform in SGD and GD
    - batch_size        batch size for SGD
    - mom               add momentum to algorithm
    - savename          savename for plotted heatmap
    """
    plt.rcParams.update({'font.size': 10})
    acc = np.zeros((len(eta), len(lamda)))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            print(lamda[j])
            if method == "none":
                mom = 0
            logreg = GradientDescent(cost="LOGREG", method=method, iterations=iterations,
                                     eta=eta[i], lamda=lamda[j], moment=mom, seed=100)
            if gradient == "SGD":
                logreg.SGD(X_train, y_train, batch_size)
            elif gradient == "GD":
                logreg.GD(X_train, y_train)

            acc[i, j] = logreg.predict_accuracy(X_test, y_test)
    plt.figure()
    plt.title(method)
    df = pd.DataFrame(acc, columns=np.log10(lamda), index=eta)
    sns.heatmap(df, annot=True, cbar_kws={
                "label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" % (savename), dpi=300, bbox_inches='tight')
