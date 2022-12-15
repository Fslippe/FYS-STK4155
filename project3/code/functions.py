from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
import autograd.numpy as np
import pandas as pd
from matplotlib import cm
from autograd import elementwise_grad
from autograd import grad
from random import random, seed
import logging
import os
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
plt.rcParams.update({"font.size": 14})


def accuracy(y_test, pred):
    """Accuracy score for binary prediction"""
    return np.sum(np.where(pred == y_test.ravel(), 1, 0)) / len(y_test)


def bootstrap(X_train, y_train, model, n_B, epochs=None, batch_size=None):
    for i in range(n_B):
        X_, y_ = resample(X_train, y_train)
        if epochs == None and batch_size == None:
            model.fit(X_, y_, verbose=0)

        elif epochs == None and batch_size != None:
            model.SGD(X_, y_, batch_size)

        else:
            model.fit(X_, y_, epochs=100, batch_size=32, verbose=0)

    return model


def create_neural_network_keras(neurons, optimizer="adam"):
    """
    Create a Neural network in keras
    takes in:
    - neurons:      list of neurons of hidden layers 
    - xy:           Train design matrix
    - z:            target data
    - epochs:       iterations in training
    returns:
    - model object to use for predictions
    """
    model = Sequential()
    model.add(Dense(neurons[0], activation='relu', input_shape=(20,)))
    neurons = neurons[1:]
    for layer in neurons:
        model.add(Dense(layer, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


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


def grid_search_layers(neurons, n_layer, X_train, X_test, y_train, y_test, optimizer="adam", n_B=None, epochs=100, batch_size=32, savename=None):
    scores = np.zeros((len(neurons), len(n_layer)))
    print("Total runs: ", len(neurons)*len(n_layer))
    for i in range(len(neurons)):
        print("Running ", i*len(n_layer) + 1)
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])
            model = create_neural_network_keras(neurons, optimizer)
            if n_B != None:
                model_boot = bootstrap(
                    X_train, y_train, model, n_B, epochs=100, batch_size=32)
                scores[i, j] = model_boot.evaluate(X_test, y_test)[1]
            else:
                model.fit(X_train, y_train, epochs=epochs,
                          batch_size=batch_size, verbose=0)
                scores[i, j] = model.evaluate(X_test, y_test)[1]

    plt.figure()
    df = pd.DataFrame(scores, columns=n_layer, index=neurons)

    if savename == None:
        return df
    else:
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)
        plt.xlabel("layers")
        plt.ylabel("neurons per layer")
        plt.savefig("../figures/%s.png" %
                    (savename), dpi=300, bbox_inches='tight')


def grid_search_epochs(epochs, batch_size, X_train, X_test, y_train, y_test, savename):
    scores = np.zeros((len(epochs), len(batch_size)))
    for i in range(len(epochs)):
        for j in range(len(batch_size)):
            neur = neuron_array(batch_size[j], epochs[i])
            model = create_neural_network_keras([10, 10])
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
            scores[i, j] = model.evaluate(X_test, y_test)[1]

    plt.figure()
    df = pd.DataFrame(scores, columns=batch_size, index=epochs)
    if savename == None:
        return df
    else:
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)

        plt.xlabel("batch size")
        plt.ylabel("epochs")
        plt.savefig("../figures/%s.png" %
                    (savename), dpi=300, bbox_inches='tight')


def grid_search_trees_depth(trees, depth, X_train, X_test, y_train, y_test, n_B=None, savename=None):
    scores = np.zeros((len(trees), len(depth)))
    for i in range(len(trees)):
        print("Running ", i*len(depth) + 1)
        for j in range(len(depth)):
            model = tfdf.keras.RandomForestModel(
                num_trees=trees[i], max_depth=depth[j])
            model.compile(metrics=["accuracy"])
            if n_B != None:
                model_boot = bootstrap(
                    X_train, y_train, model, n_B)
                scores[i, j] = model_boot.evaluate(X_test, y_test)[1]
            else:
                model.fit(X_train, y_train)
                scores[i, j] = model.evaluate(X_test, y_test)[1]

    plt.figure()
    df = pd.DataFrame(scores, columns=depth, index=trees)
    if savename == None:
        return df
    else:
        sns.heatmap(df, annot=True, cbar_kws={
                    "label": r"$Accuracy$"}, fmt=".3f", vmin=0.8, vmax=1)

        plt.xlabel("Depth of trees")
        plt.ylabel("Number of trees")
        plt.savefig("../figures/%s.png" %
                    (savename), dpi=300, bbox_inches='tight')


def grid_search_logreg(X_train, y_train, X_test, y_test, gradient, lamda, eta, method, iterations, batch_size, mom, savename, n_B=None):
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
    - n_B               Bootstrap iterations default None

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
                if n_B != None:
                    logreg = bootstrap(X_train, y_train, logreg, n_B,
                                       batch_size=batch_size)
                else:
                    logreg.SGD(X_train, y_train, batch_size)
            elif gradient == "GD":
                logreg.GD(X_train, y_train)

            acc[i, j] = logreg.predict_accuracy(X_test, y_test)
    plt.figure()
    plt.title(method)
    df = pd.DataFrame(acc, columns=np.log10(lamda), index=eta)
    sns.heatmap(df, annot=True, cbar_kws={
                "label": r"$Accuracy$"}, fmt=".3f", vmin=0.7, vmax=1)

    plt.xlabel(r"log$_{10}(\lambda$)")
    plt.ylabel(r"$\eta$")
    plt.savefig("../figures/%s.png" % (savename), dpi=300, bbox_inches='tight')
