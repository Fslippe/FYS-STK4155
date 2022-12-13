import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tkr
import autograd.numpy as np
import pandas as pd
from matplotlib import cm
from autograd import elementwise_grad
from autograd import grad
from random import random, seed
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample, shuffle
import logging
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# This allows appending layers to existing models
# This allows defining the characteristics of a particular layer
# This allows using whichever optimiser we want (sgd,adam,RMSprop)
# This allows using whichever regularizer we want (l1,l2,l1_l2)
# This allows using categorical cross entropy as the cost function

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
plt.rcParams.update({"font.size": 14})


def OLS(X, z):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z

    return beta


def ridge_regression(X, z, lamda):
    """
    Takes in a design matrix and actual data and returning
    an array of best beta for X and z
    """
    N = X.shape[1]
    beta = np.linalg.pinv(X.T @ X + lamda*np.eye(N)) @ X.T @ z

    return beta


def MSE(data, model):
    """
    takes in actual data and modelled data to find
    Mean Squared Error
    """
    MSE = mean_squared_error(data.ravel(), model.ravel())
    return MSE


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


def R2(data, model):
    """
    takes in actual data and modelled data to find
    R2 score
    """
    R2 = r2_score(data.ravel(), model.ravel())
    return R2


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
