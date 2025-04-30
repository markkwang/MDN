import numpy as np
import pandas as pd

from model.mdn_model import MDN

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

plt.style.use('ggplot')

X, y = make_moons(n_samples=2500, noise=0.03)
y = X[:, 1].reshape(-1,1)
X = X[:, 0].reshape(-1,1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

EPOCHS = 10000
BATCH_SIZE=len(X)

model = MDN(n_mixtures = 2, 
            dist = 'laplace',
            input_neurons = 1000, 
            hidden_neurons = [25], 
            optimizer = 'adam',
            learning_rate = 0.001, 
            early_stopping = 250,
            input_activation = 'relu',
            hidden_activation = 'leaky_relu')

model.fit(X, y, epochs = EPOCHS, batch_size = BATCH_SIZE)

model.plot_samples_vs_true(X, y, alpha = 0.2)