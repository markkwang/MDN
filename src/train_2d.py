import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mdn import MDN
from data import (
    generate_parabola_data,
    generate_moons_like_data,
    generate_piecewise_data,
    generate_bifurcated_spiral_data,
    generate_inverse_sine
)
from loss import mdn_loss
from utils import sample_from_mdn

input_dim = 1
output_dim = 1
hidden_dim = 20
epochs = 2000
lr = 0.001

# Load data (you can switch function here)
x_train, x_test, y_train, y_test = generate_moons_like_data()

# Model
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    MDN(hidden_dim, output_dim, k=2)
)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
print("Training...")
for epoch in range(epochs):
    model.train()
    model.zero_grad()
    pi, sigma, mu = model(x_train)
    loss = mdn_loss(pi, sigma, mu, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
print("Training completed.")

# Evaluation
model.eval()
with torch.no_grad():
    pi_test, sigma_test, mu_test = model(x_test)
    y_sample = sample_from_mdn(pi_test, sigma_test, mu_test).numpy()

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.3, label="Training Data")
plt.scatter(x_test.numpy(), y_test.numpy(), alpha=0.3, label="Test Data", color="green")
plt.scatter(x_test.numpy(), y_sample, color="red", alpha=0.3, label="MDN Samples")
plt.title("Mixture Density Network Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
