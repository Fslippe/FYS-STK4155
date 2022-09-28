from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from functions import *



# Make data.
N = 22
maxdegree = 15
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x,y)
np.random.seed(2018)
sigma2 = 0.1
noise = np.random.normal(0,sigma2,(N,N))
nbootstraps = N


z = FrankeFunction(x,y)+noise
z = z.ravel()

polydegree = np.zeros(maxdegree)
MSE = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
var = np.zeros(maxdegree)

X = create_X(x, y, maxdegree)
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
X_train_scaled_all,X_test_scaled_all,z_train_scaled,z_test_scaled = mean_scaler(X_train,X_test,z_train,z_test)
z_train_scaled,z_test_scaled = z_train_scaled[:,None],z_test_scaled[:,None]



for degree in range(1,maxdegree+1):
    X_train_scaled,X_test_scaled = features(X_train_scaled_all,X_test_scaled_all,degree)
    z_pred = bootstrap_OLS(X_train_scaled,X_test_scaled,z_train_scaled,z_test_scaled,nbootstraps)


    polydegree[degree-1] = degree
    MSE[degree-1] = np.mean( np.mean((z_test_scaled - z_pred)**2, axis=1, keepdims=True) )
    bias[degree-1] = np.mean( (z_test_scaled - np.mean(z_pred, axis=1, keepdims=True))**2 )
    var[degree-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    print('Polynomial degree:', degree)
    print('Error:', MSE[degree-1])
    print('Bias^2:', bias[degree-1])
    print('Var:', var[degree-1])
    print('{} >= {} + {} = {}'.format(MSE[degree-1], bias[degree-1], var[degree-1], bias[degree-1]+var[degree-1]))
    print("_______________________________________________________________")



plt.plot(polydegree,MSE,"-o",color="r",label="MSE")
plt.plot(polydegree,bias,"-o",color="b",label="Bias")
plt.plot(polydegree,var,"-o",color="g",label="Variance")
plt.title(f"N={N}, $\sigma^2 = {sigma2}$ ")
plt.legend()
plt.show()
