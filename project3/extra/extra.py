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
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style("darkgrid")
plt.rcParams.update({"font.size": 14})


def decision_tree(tmp, depth):
    """
    Returns a DecisionTreeRegressor model with a specified maximum depth.

    Parameters
    ----------
    tmp: can be anything 
    depth: int
        The maximum depth of the decision tree.

    Returns
    -------
    model: DecisionTreeRegressor
        A decision tree model with the specified maximum depth.
    """
    return DecisionTreeRegressor(max_depth=depth)


def random_forest(trees, depth):
    """
    Returns a RandomForestRegressor model with a specified maximum depth and number of trees.

    Parameters
    ----------
    depth: int
        The maximum depth of the decision tree.
    trees: int
        The number of trees in the random forest.

    Returns
    -------
    model: RandomForestRegressor
        A random forest model with the specified maximum depth and number of trees.
    """
    model = RandomForestRegressor(max_depth=depth, n_estimators=trees)
    return model


def bootstrap(X_train, y_train, X_test, model, n_B, NN_model=False):
    """
    Perform bootstrapping on a given model and return the predictions for a test set.

    Parameters
    ----------
    X_train: array-like
        The training data input.
    y_train: array-like
        The training data output.
    X_test: array-like
        The test data input.
    model: model
        The model to be bootstrapped.
    n_B: int
        The number of bootstrap samples.
    NN_model: bool, optional
        Indicates whether the model is a neural network model. Default is False.

    Returns
    -------
    y_pred: array-like
        The predictions for the test data.
    """

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


def tradeoff(X_train, X_test, y_train, y_test, model_in, complexity_1, complexity_2, n_B=100, skip=1):
    """
    Calculate the bias, variance, and mean squared error for a range of model complexities.

    Parameters
    ----------
    X_train: array-like
        The training data input.
    X_test: array-like
        The test data input.
    y_train: array-like
        The training data output.
    y_test: array-like
        The test data output.
    model_in: function
        The model to be evaluated.
    complexity_1: int
        The range of model complexities to be evaluated.
    complexity_2: int
        The second parameter for the model function.
    n_B: int, optional
        The number of bootstrap samples. Default is 100.
    skip: int, optional
        The step size for the model complexity range. Default is 1.

    Returns
    -------
    bias: array-like
        The bias of the model at each complexity.
    variance: array-like
        The variance of the model at each complexity.
    mse: array-like
        The mean squared error of the model at each complexity.
    """
    bias = np.zeros(complexity_1-1)
    variance = np.zeros(complexity_1-1)
    mse = np.zeros(complexity_1-1)
    if model_in == NN:
        NN_model = True
    else:
        NN_model = False

    for comp in range(1, complexity_1, skip):
        model = model_in(complexity_2, comp)
        y_pred = bootstrap(
            X_train, y_train, X_test, model, n_B, NN_model)

        bias[comp-1] = np.mean(
            (y_test - np.mean(y_pred, axis=1, keepdims=True).T)**2)
        variance[comp-1] = np.mean(np.var(y_pred, axis=1))
        mse[comp-1] = np.mean(
            np.mean((y_test - y_pred.T)**2, axis=1, keepdims=True))

    return bias, variance, mse


def plot_tradeoff(bias, variance, mse):
    """
    Plot the bias, variance, and mean squared error for a range of model complexities.

    Parameters
    ----------
    bias: array-like
        The bias of the model at each complexity.
    variance: array-like
        The variance of the model at each complexity.
    mse: array-like
        The mean squared error of the model at each complexity.

    Returns
    -------
    None
    """
    fig = plt.figure()
    complexity = np.linspace(1, len(bias), len(bias))
    mask = np.where(mse != 0)
    plt.plot(complexity[mask], bias[mask], linestyle="-",
             marker="o", label=r"Bias$^2$")
    plt.plot(complexity[mask], variance[mask],
             linestyle="-", marker="o", label="Variance")
    plt.plot(complexity[mask], mse[mask], linestyle="-",
             marker="o", label="MSE test")
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()


def main():
    # Load Dataset
    boston_dataset = load_boston()
    boston_dataset.keys()
    boston = pd.DataFrame(boston_dataset.data,
                          columns=boston_dataset.feature_names)
    target = boston_dataset.target

    # Split data into smaller portion including 40% of initial dataset
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

    # Run tradeoff if True
    run_NN = True
    run_RF = True
    run_DT = True

    # Neural Network
    if run_NN:
        for layer in [1, 2, 3, 4]:
            print(layer)
            start = time.time()
            bias, variance, mse = tradeoff(X_train,
                                           X_test,
                                           y_train,
                                           y_test,
                                           model_in=NN,
                                           complexity_1=60,
                                           complexity_2=layer,
                                           n_B=100,
                                           skip=3)
            plot_tradeoff(bias, variance, mse)

            plt.title("Number of layers: %i" % (layer))
            plt.xlabel("Number of neurons")

            plt.savefig("figures/tradeoff_NN_neurons_%s.png" %
                        (layer), dpi=300, bbox_inches="tight")
            print(time.time() - start)

    # Random Forest
    if run_RF:
        for trees in [5, 10, 20, 30]:  # [5, 10, 20, 30]:
            print(trees)
            start = time.time()
            bias, variance, mse = tradeoff(X_train,
                                           X_test,
                                           y_train,
                                           y_test,
                                           model_in=random_forest,
                                           complexity_1=40,
                                           complexity_2=trees,
                                           n_B=100,
                                           skip=2)
            plot_tradeoff(bias, variance, mse)
            plt.title("Number of trees: %i" % (trees))
            plt.xlabel("Tree depth")

            plt.savefig("figures/tradeoff_RF_trees_%s.png" %
                        (trees), dpi=300, bbox_inches="tight")
            print(time.time() - start)

    # Decision Tree
    if run_DT:
        bias, variance, mse = tradeoff(X_train,
                                       X_test,
                                       y_train,
                                       y_test,
                                       model_in=decision_tree,
                                       complexity_1=20,
                                       complexity_2=0,
                                       n_B=100)
        plot_tradeoff(bias, variance, mse)
        plt.xlabel("Tree depth")
        plt.savefig("figures/tradeoff_DT.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
