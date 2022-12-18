# Project 3 - Predicting rainfall tomorrow based on today’s weather using machine learning methods.

This Project includes 4 Folders

- code: All python code used in this project
- extra: Code and pdf for extra task. See folder README for more information
- figures: Figures used in report and generated using the included python files
- latex: Report Pdf, .tex and project description

Following packages need to be installed before running the commands:

```
pip3 install matplotlib
pip3 install numpy
pip3 install sklearn
pip3 install seaborn
pip3 install tensorflow
pip3 install keras
pip3 install tensorflow_decision_forests
pip3 install pandas
```

The code is structured around one main function which imports general functions from functions.py and a gradient decent class from gradient_decent.py

To reproduce the results run the main file with

- python3 main.py

# Abstract

In this article we will look into machine learning methods on predicting rain tomorrow from
freely available weather data from Australian weather stations. We will compare our results to
already established weather forecasts and look into how machine learning methods best can be
implemented to help make weather predictions. We will look into how a model’s training location
affects its test accuracy on other locations and try finding how one can optimize forecasts in
areas with a lack of weather observations. We have found an importance of training a model
on data from the same climate as the model will be tested on, with indications of sunshine and
humidity being the most important factors to be similar. Random Forest has overall been the
best performer, generally beating the accuracy of a Neural Network by around 1%. Logistic
regression has also been tested, which showed worse results than a model only predicting no
rain only reaching an accuracy of 79.4%. In total, we reached a maximum accuracy of 97.2%
using a Neural Network versus 96.3% for a Random Forest when training and testing a model
on the same weather station, and accuracies of 85.9 and 86.6% respectively when training and
testing a model on randomly chosen data from a variety of different weather stations. In total,
we have seen a disappointing performance of our models being outperformed by traditional
weather forecast models showing us the importance of interplay instead of only using either
machine learning or traditional methods in forecasting.
