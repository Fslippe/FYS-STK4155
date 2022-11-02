from NN import *
from functions import *
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

def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta, lmbd):
    model = Sequential()
    model.add(Input(shape=2))
    model.add(Dense(n_neurons_layer1, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_neurons_layer2, activation='sigmoid', kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Dense(n_categories, activation='softmax'))
    
    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    
    return model

def create_neural_network_scikit(hidden_layer_sizes, lamda, eta, epochs):
    dnn = MLPClassifier(hidden_layer_sizes= hidden_layer_sizes, activation="relu",
                            alpha=lamda, learning_rate_init=eta, max_iter=epochs)
    return dnn
   
def main():
    #Setting up data
    n = 30
    noise_std = 0.2
    degree = 6

    # train data
    x, y, z = make_data(n, noise_std, seed=100)
    x_s, mn_x, mx_x = min_max_scale(x)
    y_s, mn_y, mx_y = min_max_scale(y)
    z_s, mn, mx = min_max_scale(z)

    # Test data
    x, y, z = make_data(n, noise_std, seed=200)
    x_s_test, mn_x, mx_x = min_max_scale(x)
    y_s_test, mn_y, mx_y = min_max_scale(y)
    z_s_test = z
    print(x_s_test)
    xy = np.array([x_s[0:n+1], y_s[0].reshape(n+1, 1)]).T
    xy_s_test = np.array([x_s_test[0].reshape(n+1, 1), y_s_test[0]].reshape(n+1, 1)).T

    # OLS
    X_s = design_matrix(x_s, y_s, degree)
    X_s_test = design_matrix(x_s_test, y_s_test, degree)
    beta_OLS = OLS(X_s, z_s)
    pred_OLS = X_s_test @ beta_OLS 
    pred_OLS = min_max_unscale(pred_OLS, mn, mx)
    OLS_mse = (MSE(pred_OLS, z_s_test))

    # Ridge
    X_s = design_matrix(x_s, y_s, 8)
    X_s_test = design_matrix(x_s_test, y_s_test, 8)
    beta_RIDGE = ridge_regression(X_s, z_s, lamda=1.61e-7)
    pred_RIDGE = X_s_test @ beta_RIDGE 
    pred_RIDGE = min_max_unscale(pred_RIDGE, mn, mx)
    RIDGE_mse = (MSE(pred_RIDGE, z_s_test))

    # Own NeuralNetwork
    neurons = np.array([100, 100, 100]) # in hidden layers 
    epochs = 400
    batch_size = 5
    eta = 0.1
    lamda=1e-5
    NN = NeuralNetwork(xy, z_s.reshape(len(z_s), 1), neurons, epochs, batch_size, eta)
    NN.SGD()
    pred = min_max_unscale(NN.predict(xy_s_test).ravel(), mn, mx)
    NN_mse = (MSE(z_s_test, pred))

    # Tensor flow keras NN
    #DNN = create_neural_network_keras(neurons[0], neurons[1], 1,
    #                                    eta=eta, lmbd=lamda)
    #DNN.fit(xy, z_s, epochs=epochs, batch_size=batch_size, verbose=0)
    #pred_tf = min_max_unscale(DNN.predict(xy).ravel(), mn, mx)
    #scores = DNN.evaluate(xy, min_max_scale(z_s)[0])
    gamma = 0.0
    lamda=0
    cost_func="mean_squared_error"
    model = Sequential()
    model.add(Input(shape=2))
    model.add(Dense(neurons[0], activation='sigmoid', kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(neurons[1], activation='sigmoid', kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(neurons[2], activation='sigmoid', kernel_regularizer=regularizers.l2(lamda)))

    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(learning_rate=eta,  momentum=gamma)
    model.compile(loss=cost_func, metrics=['mean_squared_error'])

    model.fit(xy, z_s.reshape(len(z_s), 1), epochs=epochs, verbose=1)
    pred_tf = min_max_unscale(model.predict(xy_s_test).ravel(), mn, mx)
    tf_NN_mse = (MSE(z_s_test, pred_tf))

    # plot predictions
    plot_3d_trisurf(x_s_test, y_s_test, z_s_test, azim=45, title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, NN_mse))
    plt.show()
    plot_3d_trisurf(x_s_test, y_s_test, pred, azim=45,savename="test_NN", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, NN_mse))
    plt.show()
    plot_3d_trisurf(x_s_test, y_s_test, pred_tf, azim=45,savename="test_NN_tf", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, tf_NN_mse))
    plt.show()
    
    plot_3d_trisurf(x_s_test, y_s_test, pred_OLS, azim=45,savename="test_OLS", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, OLS_mse))
    plt.show()
    
    plot_3d_trisurf(x_s_test, y_s_test, pred_RIDGE, azim=45,savename="test_RIDGE", title="n=%i, std=%.1f, MSE=%.5f" %(n, noise_std, RIDGE_mse))

    plt.show()


if __name__ == "__main__":
    main()