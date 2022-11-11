from NN import *
from functions import *
import logging, os
from grid_search import *
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from sklearn.neural_network import MLPClassifier

def min_max_scale(X):
    mx = np.max(X)
    mn = np.min(X)
    X_std = (X - mn) / (mx - mn)
    X_scaled = X_std 
    return X_scaled, mn, mx

def min_max_unscale(X, mn, mx):
    X_s = X* (mx - mn) + mn
    return X_s

def create_neural_network_keras(neurons, xy, z, lmb, epochs):
    # Tensor flow keras NN
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation="relu"))
    for layer in neurons:
        model.add(Dense(layer, activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", metrics=['mean_squared_error'], optimizer="adam")
    model.fit(xy, z, epochs=epochs, verbose=0)
    return model


def grid_search_franke(xy, xy_test, z, z_test, neurons, epochs, batch_size, mn, mx, act, validate):
    savename = "franke_L_n_test_%s_%s" %(act, validate)
    neur = np.array( [60, 80, 100, 120, 140, 160, 180, 200, 220])
    n_layer = np.array([1, 2, 3, 4, 5, 6, 7, 8])        
    #neur = np.array([120])
    #n_layer = np.array([5])
    if act == "relu":
        neurons = [120, 120, 120, 120, 120, 120, 120]
        last_act = "none"
        init = "random scaled"
        eta = 0.01
        lamda = 1e-4

    elif act == "lrelu":
        neurons = [120, 120, 120, 120, 120, 120, 120]
        last_act = "none"
        init = "random scaled"
        eta = 0.01
        lamda = 1e-4

    else:
        last_act = "sigmoid"
        init = "random"
        eta = 0.25
        lamda = 1e-4
    grid_search_layer_neurons(xy,
                              z,
                              xy_test,
                              z_test,
                              savename,
                              n_layer,
                              neur,
                              epochs,
                              batch_size,
                              eta=eta,
                              lamda=lamda,
                              mom=0,
                              seed=100,
                              init=init,
                              act=act,
                              cost="error",
                              last_act=last_act,
                              validate=validate,
                              mn=mn,
                              mx=mx)
    plt.show()
    print(neurons)
    savename = "franke_eta_lmb_%s_%s" %(act, validate)
    if act == "lrelu":
        eta = np.array([0.001, 0.05, 0.01, 0.02, 0.05, 0.1])
        lamda = np.logspace(-6, -1, 6)
    elif act == "relu":
        eta = np.array([0.001, 0.05, 0.01, 0.02, 0.05, 0.1])
        lamda = np.logspace(-6, -1, 6)
    else:
        eta = np.array([ 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
        lamda = np.logspace(-6, -1, 6)
    grid_search_eta_lambda(xy,
                        z,
                        xy_test,
                        z_test,
                        savename,
                        neurons,
                        epochs,
                        batch_size,
                        eta,
                        lamda,
                        mom=0,
                        seed=100,
                        init=init,
                        act=act,
                        cost="error",
                        last_act=last_act,
                        validate=validate,
                        mn=mn,
                        mx=mx)
    plt.show()

def main():
    #Setting up data
    n = 30
    noise_std = 0.2

    # train data
    x_train, y_train, z_train = make_data(n-1, noise_std, seed=100)
    x_train = x_train.ravel()
    y_train = y_train.ravel()
    z_train_scaled, mn, mx = min_max_scale(z_train)
    z_train_scaled = z_train_scaled.reshape(len(z_train_scaled), 1)

    # Test data
    x_test, y_test, z_test = make_data(n-1, noise_std, seed=200)
    x_test = x_test.ravel()
    y_test = y_test.ravel()
    z_test = z_test.reshape(len(z_test), 1)
    z_test_scaled = ((z_test - mn) / (mx - mn)).reshape(len(z_test), 1)
    
    # train and test data for Neural Network
    xy_train = np.array([x_train, y_train]).T
    xy_test = np.array([x_test, y_test]).T

    # OLS
    X_train = design_matrix(x_train, y_train, degree=6)
    X_test = design_matrix(x_test, y_test, degree=6)
    beta_OLS = OLS(X_train, z_train)
    pred_OLS = X_test @ beta_OLS 
    OLS_mse = MSE(z_test, pred_OLS)
    print("R2 OLS:",R2(z_test, pred_OLS))

    # Ridge
    X_train_ridge = design_matrix(x_train, y_train, degree=8)
    X_test_ridge = design_matrix(x_test, y_test, degree=8)
    beta_RIDGE = ridge_regression(X_train_ridge, z_train, lamda=1.61e-7)
    pred_RIDGE = X_test_ridge @ beta_RIDGE 
    RIDGE_mse = MSE(z_test, pred_RIDGE)
    print("R2 ridge:",R2(z_test, pred_RIDGE))

    # Own NeuralNetwork
    neurons = np.array([140, 140, 140, 140, 140, 140]) # in hidden layers 
    epochs = 100
    batch_size = 60

    # NN TF keras
    NN_tf = create_neural_network_keras([100,100], xy_train, z_train_scaled, lmb=1e-5, epochs=100)
    pred_tf = min_max_unscale(NN_tf.predict(xy_test), mn, mx)
    NN_tf_mse = MSE(z_test, pred_tf)
    print("R2:",R2(z_test, pred_tf))
    grid_search_franke(xy_train, xy_test, z_train_scaled, z_test, neurons, epochs, batch_size, mn, mx, act="lrelu", validate="MSE")
    
    act = "lrelu"
    if act == "sigmoid":
        init = "random"
        last_act = act
        eta = 0.2
        lamda=1e-5
    else:
        neurons = np.array([120, 120, 120, 120, 120, 120, 120])
        init = "random scaled"
        last_act = "none"
        eta = 0.02
        lamda=1e-2

    NN = NeuralNetwork(xy_train,
                       z_train_scaled,
                       neurons,
                       epochs,
                       batch_size,
                       eta=eta,
                       lamda=lamda,
                       moment=0,
                       cost="error",
                       seed=100,
                       initialize_weight=init,
                       activation=act,
                       last_activation=last_act)
    NN.SGD()
    pred = NN.predict(xy_test).ravel()
    pred = min_max_unscale(pred, mn, mx)
    NN_mse = (MSE(z_test, pred))
    eta = 0.1

    # plot predictions
    print(R2(z_test,pred))
    plot_3d_trisurf(x_test, y_test, pred_tf.ravel(), savename="NN_tf_franke", azim=45, title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, NN_tf_mse))
    plt.show()
    plot_3d_trisurf(x_test, y_test, z_test.ravel(), azim=45, title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, NN_mse))
    plt.show()
    plot_3d_trisurf(x_test, y_test, pred, azim=45,savename="NN_%s_franke" %(act), title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, NN_mse))
    plt.show()
    #plot_3d_trisurf(x_test, y_test, pred_tf, azim=45,savename="test_NN_tf", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, tf_NN_mse))
    #plt.show()
    
    plot_3d_trisurf(x_test, y_test, pred_OLS.ravel(), azim=45,savename="test_OLS", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, OLS_mse))
    plt.show()
    
    plot_3d_trisurf(x_test, y_test, pred_RIDGE.ravel(), azim=45,savename="test_RIDGE", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, RIDGE_mse))

    plt.show()


if __name__ == "__main__":
    main()