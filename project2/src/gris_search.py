from NN import *
from functions import * 
from compare import *

def grid_search_eta_lamda(xy, xy_test, z, z_test, eta, lamda, neurons, mn, mx, savename, epochs=50, batch_size=5, init="random", act="sigmoid", seed=100):
    mse = np.zeros((len(eta), len(lamda)))
    for i in range(len(eta)):
        for j in range(len(lamda)):
            print(lamda[j])
            NN = NeuralNetwork(xy, z.reshape(len(z), 1), neurons, epochs, batch_size, eta[i], lamda[j], seed=seed, initialize=init, activation=act)
            NN.SGD()
            pred = min_max_unscale(NN.predict(xy_test).ravel(), mn, mx)
            mse[i, j] = (MSE(z_test, pred))

    df = pd.DataFrame(mse, columns=np.log10(lamda), index=eta)
    plt.figure()
    sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.025, vmax=0.2)
    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" %(savename), dpi=300, bbox_inches='tight')

def neuron_array(layers, neurons):
    n = np.zeros(layers)
    for i in range(layers):
        n[i] = int(neurons)

    return n

def grid_search_layer_neurons(xy, xy_test, z, z_test, n_layer, neurons, mn, mx, savename, epochs=50, eta=0.1, lamda=0, batch_size=5, init="random", act="sigmoid", seed=100):
    mse = np.zeros((len(neurons), len(n_layer)))
    for i in range(len(neurons)):
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])
            print(neur)
            NN = NeuralNetwork(xy, z.reshape(len(z), 1), neur, epochs, batch_size, eta, lamda, seed=seed, initialize=init, activation=act)
            NN.SGD()
            pred = min_max_unscale(NN.predict(xy_test).ravel(), mn, mx)
            mse[i, j] = (MSE(z_test, pred))


    df = pd.DataFrame(mse, columns=n_layer, index=neurons)
    plt.figure()
    sns.heatmap(df, annot=True, cbar_kws={"label": r"$MSE$"}, vmin=0.025, vmax=0.2)
    plt.xlabel("layers")
    plt.ylabel("neurons per layer")
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
    grid_search_eta_lamda(xy, xy_test, z_s, z, eta, lamda, neurons, mn, mx, "grid_eta_lmb", epochs=50, batch_size=5)


    # layers - neurons
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