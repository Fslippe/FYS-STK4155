from NN import *
from functions import *

def main():
    n = 100
    noise_std = 0.1
    degree = 6
    x, y, z = make_data(n, noise_std)
    X = design_matrix(x, y, degree)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

if __name__ == "__main__":
    main()