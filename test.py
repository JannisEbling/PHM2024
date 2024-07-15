import torch
import pyro
import pyro.distributions as dist
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
from pyro.distributions import TransformedDistribution

# Generate some example multivariate time series data
# Replace this with your actual data
data = torch.randn(100, 3)  # 100 time steps, 3 variables

# Define base distribution
base_dist = dist.MultivariateNormal(torch.zeros(3), torch.eye(3))

# Define normalizing flow using affine autoregressive transforms
input_dim = data.shape[1]
hidden_dims = [10, 10]

flows = [AffineAutoregressive(AutoRegressiveNN(input_dim, hidden_dims)) for _ in range(5)]
flow_dist = TransformedDistribution(base_dist, flows)

# Train the model (simplified, actual training code will involve optimizer steps and loss calculation)
optimizer = torch.optim.Adam(flow_dist.parameters(), lr=1e-3)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    log_prob = flow_dist.log_prob(data).mean()
    loss = -log_prob
    loss.backward()
    optimizer.step()

# Transform data to latent space
with torch.no_grad():
    latent_data = flow_dist.transforms[0].inv(data)
    for transform in flow_dist.transforms[1:]:
        latent_data = transform.inv(latent_data)

# Verify independence (check covariance matrix)
latent_covariance = torch.cov(latent_data.T)
print("Covariance matrix of latent variables:\n", latent_covariance)