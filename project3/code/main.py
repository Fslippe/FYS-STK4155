from functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow_decision_forests as tfdf
from tensorflow.keras.layers import Input
# This allows appending layers to existing models
from tensorflow.keras.models import Sequential
# This allows defining the characteristics of a particular layer
from tensorflow.keras.layers import Dense
# This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import optimizers
# This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras import regularizers
# This allows using categorical cross entropy as the cost function
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'font.size': 11})
sns.set_style("whitegrid")


def create_neural_network_keras(neurons):
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
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def assign_labels(data, words):
    for i in range(len(words)):
        data.replace(words[i], i, inplace=True)
    return data


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


def grid_search_layers(neurons, n_layer, X_train, X_test, y_train, y_test, n_B=None, savename=None):
    scores = np.zeros((len(neurons), len(n_layer)))
    print("Total runs: ", len(neurons)*len(n_layer))
    for i in range(len(neurons)):
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])
            model = create_neural_network_keras(neurons)
            if n_B != None:
                model_boot = bootstrap(
                    X_train, X_test, y_train, y_test, model, n_B)
                scores[i, j] = model_boot.evaluate(X_test, y_test)[1]
            else:
                model.fit(X_train, y_train, epochs=100,
                          batch_size=32, verbose=0)
                scores[i, j] = model.evaluate(X_test, y_test)[1]
            print("Finished run ", i*j + 1)

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


def split_grid(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


def main():
    """Import dataset"""
    df = pd.read_csv("data/weatherAUS.csv")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    wind_directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE",
                       "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    wind_directions = sorted(df["WindDir3pm"].unique())
    df["WindDir3pm"] = assign_labels(df["WindDir3pm"], wind_directions)
    df["WindDir9am"] = assign_labels(df["WindDir9am"], wind_directions)
    df["WindGustDir"] = assign_labels(df["WindGustDir"], wind_directions)
    df["RainTomorrow"] = assign_labels(df["RainTomorrow"], ["No", "Yes"])
    df["RainToday"] = assign_labels(df["RainToday"], ["No", "Yes"])

    target = assign_labels(df["RainTomorrow"], ["No", "Yes"])
    df["RainTomorrow"] = target
    mask = np.random.rand(len(df)) < 0.8
    #df.drop(columns=["RainTomorrow"], inplace=True)

    """Number of measurements for each station"""
    for loca in df["Location"].unique():
        print("%s size: " % (loca), len(df.loc[df["Location"] == loca]))

    df_train = df[mask]
    df_test = df[~mask]

    tf_dataset_train = tfdf.keras.pd_dataframe_to_tf_dataset(
        df_train.iloc[0:534, 2:], label="RainTomorrow")
    tf_dataset_test = tfdf.keras.pd_dataframe_to_tf_dataset(
        df_test.iloc[0:534, 2:], label="RainTomorrow")

    # Set up data for chosen location
    location_idx = df.index[df["Location"] == "Cobar"]
    y = target[location_idx]  # np.ravel(target.ilo)
    X = df.iloc[location_idx, 2:-1]
    corr = (df.iloc[location_idx, 2:]).corr()
    sns.heatmap(corr)
    plt.savefig("../figures/correlation_heatmap.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    corr_tomorrow = pd.DataFrame({"RainTomorrow": corr["RainTomorrow"]})
    print(corr_tomorrow)
    sns.scatterplot(x=corr_tomorrow.index, y=corr_tomorrow["RainTomorrow"])
    plt.xticks(rotation=70)
    plt.savefig("../figures/correlation_plot.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform grid search
    # grid_search_layers([10, 20, 30, 40, 50], [1, 2, 3, 4, 5], X_train,
    #                   X_test, y_train, y_test, n_B=10, savename="NN_grid_ADAM_bootstrap_cobar")
   # plt.show()

    model = create_neural_network_keras([10, 10, 10, 10, 10])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    model.summary()
    #scores = model.evaluate(X_test, y_test)
    # print(scores)
    #print(model.evaluate(X_test, y_test))
    #model = tfdf.keras.RandomForestModel()
    # model.fit(tf_dataset_train)
    # print("\n\n\nEVAL")
    # print(model.evaluate(tf_dataset_test))

    # print(model.summary())
if __name__ == "__main__":
    main()
