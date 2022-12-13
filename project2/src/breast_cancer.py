from NN import *
from functions import *
from franke_compare import *
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from grid_search import *
from gradient_decent import *
plt.rcParams.update({'font.size': 11})


def choose_inputs(idx, data):
    """
    Choose input features of data
    - idx:       last feature index to use
    - data:      scikit learn data set
    returns
    - input data matrix
    """
    inputs = data.data
    labels = data.feature_names
    temp = np.zeros(len(idx), dtype=object)
    for i in range(len(idx)):
        temp[i] = np.reshape(inputs[:, idx[i]], (len(inputs[:, idx[i]]), 1))

    X = np.hstack((temp))
    return X


def scikit_logreg(X_train, y_train):
    """
    scikit learn logistic regression functionality
    trained using X_train and y_train
    returns
    scikit learn logreg model object
    """
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf


def main():
    """Setup of data"""
    data = load_breast_cancer()
    outputs = data.target
    X = choose_inputs(range(1, 30), data)
    y = outputs.reshape(len(outputs), 1)

    # A random state of 10 makes it possible to gain a 100% accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5)  # =10)
    std = np.std(X_train, axis=0)
    mean = np.mean(X_train, axis=0)

    # Standard scaler input data
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    """Which grid search to run"""
    logreg_grid = True
    grid_NN_lmb_eta = True
    grid_layer_neurons = True

    """Scikit logistic regression"""
    logreg_skl = scikit_logreg(X_train, y_train.ravel())
    acc_skl = logreg_skl.score(X_test, y_test.ravel())
    print("SKlearn accuracy:", acc_skl)

    """Grid search for own logistic regression"""
    if logreg_grid:
        print(np.shape(X_train), np.shape(y_train))
        eta = np.logspace(-4, 0, 5)
        lamda = np.logspace(-6, 0, 7)
        method = ["none", "momentum", "ADAM", "AdaGrad", "RMSprop"]
        for m in method:
            grid_search_logreg(X_train,
                               y_train,
                               X_test,
                               y_test,
                               gradient="SGD",
                               lamda=lamda,
                               eta=eta,
                               method=m,
                               iterations=100,
                               batch_size=10,
                               mom=0.3,
                               savename="logreg_%s" % (m))
        plt.show()

    neurons = [80, 80, 80]
    epochs = 200
    batch_size = 45
    print(np.shape(X_train))

    """eta-lambda grid search own Neural Network"""
    if grid_NN_lmb_eta:
        savename = "cancer_eta_lmb"
        eta = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
        lamda = np.logspace(-5, 0, 6)
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
                               init="random",
                               act="sigmoid",
                               cost="cross entropy",
                               last_act="sigmoid",
                               validate="ACC")
        plt.show()

    """layer-neurons grid search own Neural Network"""
    if grid_layer_neurons:
        eta = 0.2
        lamda = 1e-1
        savename = "cancer_L_n_test"
        neurons = np.array([10, 40, 60, 80, 100])
        n_layer = np.array([1, 2, 3, 4, 5])
        grid_search_layer_neurons(X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  savename,
                                  n_layer,
                                  neurons,
                                  epochs,
                                  batch_size,
                                  eta=eta,
                                  lamda=lamda,
                                  mom=0,
                                  seed=100,
                                  init="random",
                                  act="sigmoid",
                                  cost="cross entropy",
                                  last_act="sigmoid",
                                  validate="ACC")
        plt.show()


if __name__ == "__main__":
    main()
