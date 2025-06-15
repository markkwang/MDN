import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def to_tensor_split(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

def generate_parabola_data(n=2000, low=-2, high=2):
    x = np.random.uniform(low, high, size=(n, 1))
    y = np.array([[xi[0] ** 2 if np.random.rand() < 0.5 else -xi[0] ** 2] for xi in x])
    return to_tensor_split(x, y)

def generate_moons_like_data(n_samples=2500, noise=0.03):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    y = X[:, 1].reshape(-1, 1)
    X = X[:, 0].reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)

    return to_tensor_split(X, y)

def generate_piecewise_data(n_samples=2500):
    X = np.linspace(-4, 4, n_samples).reshape(-1, 1)
    y = np.piecewise(
        X.flatten(),
        [X.flatten() < -1, (X.flatten() >= -1) & (X.flatten() < 1), X.flatten() >= 1],
        [
            lambda x: -1 + 0.1 * np.random.randn(*x.shape),
            lambda x: 1 + 0.1 * np.random.randn(*x.shape),
            lambda x: -0.5 + 0.1 * np.random.randn(*x.shape)
        ]
    ).reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)

    return to_tensor_split(X, y)

def generate_bifurcated_spiral_data(n_samples=10000, noise_std=0.2):
    theta = np.linspace(0, 2 * np.pi, n_samples)
    r = theta + 0.01 * np.random.randn(n_samples)
    r_branch = np.where(np.random.rand(n_samples) > 0.5, r, -r)

    X_vals = r_branch * np.cos(theta) + np.random.normal(scale=noise_std, size=n_samples)
    y_vals = r_branch * np.sin(theta) + np.random.normal(scale=noise_std, size=n_samples)

    X = X_vals.reshape(-1, 1)
    y = y_vals.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)

    return to_tensor_split(X, y)

def generate_inverse_sine(n_samples=10000, noise=0.05):
    x_vals = np.linspace(-2, 2, n_samples)
    y_vals = x_vals + np.sin(np.pi * x_vals) + np.random.normal(scale=noise, size=x_vals.shape)

    X = y_vals.reshape(-1, 1)
    y = x_vals.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)

    return to_tensor_split(X, y)
