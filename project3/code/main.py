import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow_decision_forests as tfdf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'font.size': 11})


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
    model.add(Dense(8, activation='relu', input_shape=(20,)))
    for layer in neurons:
        model.add(Dense(layer, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])
    return model

def assign_labels(data, words):
    for i in range(len(words)):
        data.replace(words[i], i, inplace=True)
    return data

def grid_search_layers(neurons, n_layer):
    val = np.zeros((len(neurons), len(n_layer)))
    for i in range(len(neurons)):
        for j in range(len(n_layer)):
            neur = neuron_array(n_layer[j], neurons[i])


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

def main():
    """Import dataset"""
    df = pd.read_csv("data/weatherAUS.csv")
    df.dropna(inplace=True)
    wind_directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    wind_directions = sorted(df["WindDir3pm"].unique())
    df["WindDir3pm"] = assign_labels(df["WindDir3pm"], wind_directions)
    df["WindDir9am"] = assign_labels(df["WindDir9am"], wind_directions)
    df["WindGustDir"] = assign_labels(df["WindGustDir"], wind_directions)
    df["RainTomorrow"] = assign_labels(df["RainTomorrow"], ["No", "Yes"])
    df["RainToday"] = assign_labels(df["RainToday"], ["No", "Yes"])

    print(df.iloc[0:500, 15:])
    target = assign_labels(df["RainTomorrow"], ["No", "Yes"])
    df["RainTomorrow"] = target
    mask = np.random.rand(len(df)) < 0.8
    #df.drop(columns=["RainTomorrow"], inplace=True)

    """Number of measurements for each station"""
    for loca in df["Location"].unique():
        print("%s size: "%(loca), len(df.loc[df["Location"]==loca]))

    df_train = df[mask]
    df_test = df[~mask]

    tf_dataset_train = tfdf.keras.pd_dataframe_to_tf_dataset(df_train.iloc[0:534, 2:], label="RainTomorrow")
    tf_dataset_test = tfdf.keras.pd_dataframe_to_tf_dataset(df_test.iloc[0:534, 2:], label="RainTomorrow")

    y = np.ravel(target)[:500]

    X = df.iloc[:500,2:-1]
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = create_neural_network_keras([10, 10])
    #model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

    print(model.evaluate(X_test, y_test))
    model = tfdf.keras.RandomForestModel()
    model.fit(tf_dataset_train)
    print("\n\n\nEVAL") 
    print(model.evaluate(tf_dataset_test))
    #print(model.summary())
if __name__ == "__main__":
    main()