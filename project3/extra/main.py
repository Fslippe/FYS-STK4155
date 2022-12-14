from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from matplotlib.ticker import MaxNLocator
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style("darkgrid")


def decision_tree(tmp, depth):
    return DecisionTreeRegressor(max_depth=depth)


def random_forest(trees, depth):
    model = RandomForestRegressor(max_depth=depth, n_estimators=100)

    return model


def bootstrap(X_train, y_train, X_test, model, n_B, NN_model=False):
    y_pred = np.zeros((X_test.shape[0], n_B))
    mod_boots = model
    for i in range(n_B):
        X_, y_ = resample(X_train, y_train)
        if NN_model == False:
            mod_boots.fit(X_, y_)
        else:
            mod_boots.fit(X_, y_, epochs=200, batch_size=32, verbose=0)
        y_pred[:, i] = mod_boots.predict(X_test).ravel()
    return y_pred


def NN(layers, neurons, optimizer="adam"):
    """
    Create a Neural network in keras
    takes in:
    - neurons:      number of neurons per layer 
    - layers:       number of layers
    - epochs:       iterations in training
    returns:
    - model object to use for predictions
    """
    model = Sequential()
    model.add(Dense(neurons, input_shape=(13,), activation="relu"))
    for layer in range(layers-1):
        model.add(Dense(neurons, activation="relu"))

    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer=optimizer)
    return model


def tradeoff(X_train, X_test, y_train, y_test, model_in, complexity_1, complexity_2, n_B=100):
    bias = np.zeros(complexity_1-1)
    variance = np.zeros(complexity_1-1)
    mse = np.zeros(complexity_1-1)
    if model_in == NN:
        NN_model = True
    else:
        NN_model = False

    for comp in range(1, complexity_1):
        model = model_in(complexity_2, comp)
        y_pred = bootstrap(
            X_train, y_train, X_test, model, n_B, NN_model)

        print(np.shape(np.mean(y_pred, axis=1, keepdims=True)))

        bias[comp-1] = np.mean(
            (y_test - np.mean(y_pred, axis=1, keepdims=True).T)**2)
        variance[comp-1] = np.mean(np.var(y_pred, axis=1))
        mse[comp-1] = np.mean(
            np.mean((y_test - y_pred.T)**2, axis=1, keepdims=True))

    return bias, variance, mse


def plot_tradeoff(bias, variance, mse):
    fig = plt.figure()
    complexity = np.linspace(1, len(bias), len(bias))
    plt.plot(complexity, bias, linestyle="-", marker="o", label=r"Bias$^2$")
    plt.plot(complexity, variance, linestyle="-", marker="o", label="Variance")
    plt.plot(complexity, mse, linestyle="-", marker="o", label="MSE test")
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()


def main():
    # Load Dataset
    boston_dataset = load_boston()
    boston_dataset.keys()
    boston = pd.DataFrame(boston_dataset.data,
                          columns=boston_dataset.feature_names)
    target = boston_dataset.target

    boston, tmp1, target, tmp2 = train_test_split(
        boston, target, train_size=0.4, random_state=1)
    print("Shape data split", len(target))
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        boston, target, test_size=0.2, random_state=2)
    print("Shape train split", len(y_train))
    # Standar scale
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train = (y_train - y_mean)/y_std
    y_test = (y_test - y_mean)/y_std

    run_NN = False
    run_RF = True
    run_DT = False
    # Neural Network
    if run_NN:
        for layer in [1, 2, 3, 4]:
            bias, variance, mse = tradeoff(X_train,
                                           X_test,
                                           y_train,
                                           y_test,
                                           model_in=NN,
                                           complexity_1=20,
                                           complexity_2=layer,
                                           n_B=50)
            plot_tradeoff(bias, variance, mse)
            plt.title("Number of layers: %i" % (layer))
            #plt.title("Tree depth: %i" % (depth))

            #plt.xlabel("Number of trees")
            plt.xlabel("Number of neurons")

            plt.savefig("figures/tradeoff_NN_neurons_%s.png" %
                        (layer), dpi=300, bbox_inches="tight")

    # Random Forest
    if run_RF:
        for trees in [5, 10, 20, 30]:
            bias, variance, mse = tradeoff(X_train,
                                           X_test,
                                           y_train,
                                           y_test,
                                           model_in=random_forest,
                                           complexity_1=20,
                                           complexity_2=trees,
                                           n_B=50)
            plot_tradeoff(bias, variance, mse)
            plt.title("Number of trees: %i" % (trees))
            #plt.title("Tree depth: %i" % (depth))

            #plt.xlabel("Number of trees")
            plt.xlabel("Tree depth")

            plt.savefig("figures/tradeoff_RF_trees_%s.png" %
                        (trees), dpi=300, bbox_inches="tight")

    if run_DT:
        # Decision Tree
        bias, variance, mse = tradeoff(X_train,
                                       X_test,
                                       y_train,
                                       y_test,
                                       model_in=decision_tree,
                                       complexity_1=20,
                                       complexity_2=0,
                                       n_B=50)
        plot_tradeoff(bias, variance, mse)
        plt.xlabel("Tree depth")
        plt.savefig("figures/tradeoff_DT.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
