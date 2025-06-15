import torch
import math
from torch import Tensor

def mdn_loss(pi, sigma, mu, target) -> Tensor:

    expanded_target: Tensor = target.unsqueeze(1).expand_as(mu) # (batch_size, k, output_dim)

    gaussian_likelihood: Tensor = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * ((expanded_target - mu) / sigma) ** 2) / sigma

    # for each c, calc the likelihood on the entire target
    component_likelihood: Tensor = torch.prod(gaussian_likelihood, dim=2)  # (batch_size x k)

    likelihood = torch.sum(pi * component_likelihood, dim=1) + 1e-8 # to avoid 0
    negative_log_likelihood = -torch.log(likelihood)

    return torch.mean(negative_log_likelihood)
