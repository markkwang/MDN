import torch
from torch.distributions import Categorical
from torch import Tensor

def sample_from_mdn(pi, sigma, mu) -> Tensor:
    
    batch_size, num_components, output_dim = sigma.shape

    # Sample component index for each batch item
    component_indices = Categorical(pi).sample()  # (batch_size,)

    # Gather the selected component's mu and sigma
    mu_selected = mu[torch.arange(batch_size), component_indices]         # (batch_size, output_dim)
    sigma_selected = sigma[torch.arange(batch_size), component_indices]   # (batch_size, output_dim)

    # Sample from standard normal
    eps = torch.randn_like(sigma_selected)  # (batch_size, output_dim)
    samples = mu_selected + sigma_selected * eps  # (batch_size, output_dim)

    return samples