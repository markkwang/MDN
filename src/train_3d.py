import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from mdn import MDN
from loss import mdn_loss
from utils import sample_from_mdn


def generate_synthetic_3d_data(N: int = 1000):
    x_vals = np.array([(2 * i / N) - 1 for i in range(N) for j in range(N)])
    y_vals = np.array([(2 * j / N) - 1 for i in range(N) for j in range(N)])
    z_vals = x_vals**2 - y_vals**2

    X = np.stack((x_vals, y_vals), axis=1)
    Z = z_vals.reshape(-1, 1)

    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.2, random_state=42)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Z_train, dtype=torch.float32),
        torch.tensor(Z_test, dtype=torch.float32)
    )


def main():
    input_dim = 2
    output_dim = 1
    hidden_dim = 40
    k = 3  # Number of Gaussians
    epochs = 2000
    lr = 0.001

    x_train, x_test, z_train, z_test = generate_synthetic_3d_data(30)

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        MDN(hidden_dim, output_dim, k)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training on synthetic 3D surface (z = x² - y²)...")
    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        pi, sigma, mu = model(x_train)
        loss = mdn_loss(pi, sigma, mu, z_train)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    print("Training completed.\n")

    model.eval()
    with torch.no_grad():
        pi_test, sigma_test, mu_test = model(x_test)
        z_sample = sample_from_mdn(pi_test, sigma_test, mu_test).squeeze().numpy()

    plot_3d_results(x_train, z_train, x_test, z_test, z_sample)


def plot_3d_results(x_train, z_train, x_test, z_test, z_sample):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_train[:, 0], x_train[:, 1], z_train.squeeze(), c='blue', alpha=0.3, label='Train')
    ax.scatter(x_test[:, 0], x_test[:, 1], z_test.squeeze(), c='green', alpha=0.3, label='Test')
    ax.scatter(x_test[:, 0], x_test[:, 1], z_sample, c='red', alpha=0.3, label='MDN Sample')

    ax.set_title("MDN Fit for z = x² - y²")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
