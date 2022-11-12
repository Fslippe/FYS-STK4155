from NN import *
from functions import * 
from franke_compare import *
from gradient_decent import *
plt.rcParams.update({'font.size': 11})

def neuron_array(layers, neurons):
    n = np.zeros(layers)
    for i in range(layers):
        n[i] = int(neurons)

    return n

def grid_search_eta_lambda(X_train, y_train, X_test, y_test, savename, neurons, epochs, batch_size, eta, lamda, mom, seed, init, act, cost, last_act, validate, mn=0, mx=1):
    plt.rcParams.update({'font.size': 11})
    val = np.zeros((len(eta), len(lamda)))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            print(lamda[j])
            NN = NeuralNetwork(X_train, y_train, neurons, epochs, batch_size, eta[i], lamda[j], moment=mom, cost=cost, seed=seed, initialize_weight=init, activation=act, last_activation=last_act)
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
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.03, vmax=0.1)
    elif validate == "R2":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$R^2$"}, vmin=0.5, vmax=1)
    elif validate == "ACC":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')

def grid_search_layer_neurons(X_train, y_train, X_test, y_test, savename, n_layer, neurons, epochs, batch_size, eta, lamda, mom, seed, init, act, cost, last_act, validate, mn=0, mx=1):
    plt.rcParams.update({'font.size': 11})
    val = np.zeros((len(neurons), len(n_layer)))
    for i in range(len(neurons)):
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])
            print(neurons[i])
            NN = NeuralNetwork(X_train, y_train, neur, epochs, batch_size, eta, lamda, moment=mom, cost=cost, seed=seed, initialize_weight=init, activation=act, last_activation=last_act)
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
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.03, vmax=0.1)
    elif validate == "R2":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$R^2$"}, vmin=0.5, vmax=1)
    elif validate == "ACC":
        sns.heatmap(df, annot=True, cbar_kws={"label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)
    plt.xlabel("layers")
    plt.ylabel("neurons per layer")
    plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')
      
def grid_search_logreg(X_train, y_train, X_test, y_test, gradient, lamda, eta, method, iterations, batch_size, mom, savename):
    plt.rcParams.update({'font.size': 10})
    acc = np.zeros((len(eta), len(lamda)))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            print(lamda[j])
            if method == "none":
                mom = 0
            logreg = GradientDescent(cost="LOGREG", method=method, iterations=iterations, eta=eta[i], lamda=lamda[j], moment=mom, seed=100)
            if gradient == "SGD":
                logreg.SGD(X_train, y_train, batch_size)
            elif gradient == "GD":
                logreg.GD(X_train, y_train)
            
            acc[i,j] = logreg.predict_accuracy(X_test, y_test)
    plt.figure()
    plt.title(method)
    df = pd.DataFrame(acc, columns=np.log10(lamda), index=eta)
    sns.heatmap(df, annot=True, cbar_kws={"label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')

def main():
    #Setting up data
    n = 30
    noise_std = 0.2
    degree = 6

    # train data
    x, y, z = make_data(n, noise_std, seed=100)
    x_s = x.ravel()
    y_s = y.ravel()
    z_s, mn, mx = min_max_scale(z)

    # Test data
    x, y, z = make_data(15, noise_std, seed=200)
    xy = np.array([x_s, y_s]).T
    xy_test = np.array([x.ravel(), y.ravel()]).T

    batch_size = 5
    # lambda - eta
    neurons = np.array([91, 91, 91, 91, 91]) # in hidden layers 
    eta = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 1])
    lamda = np.logspace(-9, -3, 7)
    eta = np.array([0.0001, 0.1])
    lamda = np.array([1e-6, 1e-5])
    neurons = np.array([2]) # in hidden layers 

    #grid_search_eta_lamda(xy, xy_test, z_s, z, eta, lamda, neurons, mn, mx, savename="test", epochs=50, batch_size=5)


    # layers - neurons
    eta = 0.1
    lamda=1e-0
    batch_size = n+1
    epochs = 400
    n_layer = [1, 2, 3, 4]
    neurons = [5, 10, 20, 30]
    grid_search_layer_neurons(xy, xy_test, z_s, z, n_layer, neurons, mn, mx, savename="test", epochs=epochs, eta=eta, lamda=lamda, batch_size=batch_size, init="random", act="relu", seed=100)
    """
    neurons = np.array([400, 400]) # in hidden layers 
    epochs = 100
    batch_size = 5
    eta = 0.01
    lamda=1e-5
    n_layer = [1, 2, 3, 4]
    neurons = [10, 50, 100, 200, 400]
    grid_search_layer_neurons(xy, xy_test, z_s, z, n_layer, neurons, mn, mx, savename="grid_L_n_large", epochs=50, eta=0.01, lamda=0, batch_size=5)
    n_layer = [1, 2, 3, 4, 5, 6]
    neurons = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    grid_search_layer_neurons(xy, xy_test, z_s, z, n_layer, neurons, mn, mx, savename="grid_L_n_small", epochs=50, eta=0.1, lamda=0, batch_size=5)
    n_layer = [4, 5, 6, 7, 8]
    neurons = [87, 88, 89, 90, 91, 92, 93, 94]
    grid_search_layer_neurons(xy, xy_test, z_s, z, n_layer, neurons, mn, mx, savename="grid_L_n_narrow_zoom", epochs=50, eta=0.1, lamda=0, batch_size=5)
    n_layer = [3, 4]
    neurons = [200, 250, 300, 350, 400]
    grid_search_layer_neurons(xy, xy_test, z_s, z, n_layer, neurons, mn, mx, savename="grid_L_n_wide_zoom", epochs=50, eta=0.1, lamda=0, batch_size=5)
    """

if __name__ == "__main__":
    main()