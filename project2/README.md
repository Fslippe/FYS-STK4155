# Project 2

This Project includes 3 Folders
- figures:  Figures used in report and generated using the included python files
- latex:    Report Pdf, .tex and project description 
- src:      All python code used in this project 

Following packages need to be installed before running the commands:
- pip3 install matplotlib
- pip3 install numpy 
- pip3 install sklearn
- pip3 install seaborn

The code is structured around two classes, one functions file including most used functions, one grid search file including grid search functions, and 3 files used to analyze 3 different datasets.
- NN.py: Feed forward Neural Network class
- gradient_descent.py: gradient descent class (GD or SGD)
- functions.py: Includes functions used in most of other python files
- grid_search.py: Includes grid search functions used when analyzing data
- 1D_compare.py: Analysis of a 1-dimensional function using the gradient descent class, OLS and Ridge.
- franke_compare.py: Analysis of the franke function using the NN class, OLS and Ridge 
- breast_Cancer.py: Analysis of breast cancer data using logistic function with the gradient descent class and the neural network class.

To reproduce the results run the python files with
- python3 filename.py  


# Abstract
Neural networks are great tools in data analysis and gives us an alternative to tra-
ditional regression methods. To get a better understanding of how such a network
performs we implement a Feed Forward Neural Network. This implementation includes
normal gradient descent, stochastic gradient descent and different tuning functions for
these. To understand the performance of gradient descent we compare the different
tuning functions showing us that an implementation of these can greatly improve per-
formance, with the ADAM tuning function generally topping the chart with a mean
squared error of 0.0082 compared to ridge with 0.0098 on a simple one dimensional
function. We continue using Stochastic Gradient Descent with our Neural Network and
find that it outperforms ridge regression with a mean squared error of 0.0395 compared
to ridge with 0.407 when predicting the Franke function. We look at our a classification
case of predicting breast cancer data, and end up showing that logistic regression per-
forms just as well as our Neural Network with a prediction accuracy of 98.1%, which
ends up being better than built in functionalities from scikit learn giving an accuracy
of 97.4%. In total, we have seen that our results can greatly vary depending on initial
parameters which makes a further statistical analysis of interest.
