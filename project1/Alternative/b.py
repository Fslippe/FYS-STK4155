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
import pandas as pd




# Make data.
N = 40
maxdegree = 5
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x,y)
np.random.seed(2018)
noise = np.random.normal(0,1,(N,N))


z = FrankeFunction(x,y)+0.2*noise
z = z.ravel()


beta_parameters = np.zeros(maxdegree,dtype=object)

MSE_test = np.zeros(maxdegree)
R2_test = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

X = create_X(x, y, maxdegree)
X_train, X_test, z_train, z_test = train_test_split(X,z,test_size=0.2)
X_train_scaled_all,X_test_scaled_all,z_train_scaled,z_test_scaled = mean_scaler(X_train,X_test,z_train,z_test)

for degree in range(1,maxdegree+1):
    print(f"Degree:{degree}/{maxdegree}")
    X_train_scaled,X_test_scaled = features(X_train_scaled_all,X_test_scaled_all,degree)
    beta = OLS(X_train_scaled,z_train_scaled)
    zpredict = X_test_scaled@beta

    beta_parameters[degree-1] = beta
    polydegree[degree-1] = degree
    MSE_test[degree-1] = MSE(z_test_scaled,zpredict)
    R2_test[degree-1] = R2(z_test_scaled,zpredict)


for i in range(maxdegree):
    plt.scatter(np.arange(len(beta_parameters[i])),beta_parameters[i],label=f"Degree: {i+1}")
    plt.plot(np.arange(len(beta_parameters[i])),beta_parameters[i],'--')



plt.xlabel(r"$j$")
plt.ylabel(r"$\beta_j$")
plt.xticks(np.arange(0, 21, 1.0))
plt.legend()
plt.show()

plt.subplot(121)
plt.plot(np.arange(1,maxdegree+1),MSE_test,color="r")
plt.xlabel("Complexity")
plt.ylabel("MSE")
plt.title("Mean Square Error")



plt.subplot(122)
plt.plot(np.arange(1,maxdegree+1),R2_test,color="r")
plt.xlabel("Complexity")
plt.ylabel(f"$R^2$")
plt.title(f"$R^2-Score$")
plt.show()
