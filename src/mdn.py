import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
from torch import Tensor

class MDN(nn.Module):

    def __init__(self, input_dim, output_dim, k):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k

        self.pi = nn.Sequential(
            nn.Linear(input_dim, k),
            nn.Softmax(dim=1)
        )

        self.sigma = nn.Linear(input_dim, k * output_dim)
        self.mu = nn.Linear(input_dim, k * output_dim)

    def forward(self, x):
        pi = self.pi(x)
        sigma: Tensor = torch.exp(self.sigma(x)).view(-1, self.k, self.output_dim)
        mu: Tensor = self.mu(x).view(-1, self.k, self.output_dim)

        return pi, sigma, mu
