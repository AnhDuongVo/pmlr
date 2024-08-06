#https://github.com/zama-ai/concrete-ml/blob/main/docs/advanced_examples/LogisticRegressionTraining.ipynb
# Import dataset libraries and util functions
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn import datasets
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import os
import glob
import random

random.seed(2024)
#pip install concrete-ml
from concrete.ml.sklearn import SGDClassifier

# Import the data
train_x = pd.read_csv("toyExample/train_x.csv")
train_y = pd.read_csv("toyExample/train_y.csv")
train_x = train_x.values
train_x = np.random.randn(train_x.shape[0], 3000)
train_y = train_y.iloc[:, 0].values

#Set parameters and permute the dataset
N_ITERATIONS = 15
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(train_x.shape[0])
train_x = train_x[perm, ::]
train_y = train_y[perm]
parameters_range = (-1.0, 1.0)

# Train on "fake encrypted data" (quantized only), fhe=simulate
sgd_clf_binary_simulate = SGDClassifier(
    random_state=RANDOM_STATE,
    max_iter=N_ITERATIONS,
    fit_encrypted=True,
    parameters_range=parameters_range,
)
print(train_x.shape, train_y.shape)
# Train with simulation on the full dataset
sgd_clf_binary_simulate.fit(train_x, train_y, fhe="simulate")


# Evaluate the decrypted weights on encrypted data
sgd_clf_binary_simulate.compile(train_x)
y_pred_fhe = sgd_clf_binary_simulate.predict(train_x, fhe="simulate")
accuracy = (y_pred_fhe == train_y).mean()
print(f"Full encrypted fit (simulated) accuracy {accuracy*100:.2f}%")