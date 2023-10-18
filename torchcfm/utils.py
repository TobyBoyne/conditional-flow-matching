import math

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons

# Implement some helper functions

def get_conjugate_gaussians(
        prior_mu=0.0,
        prior_sigma=1.0,
        ll_mu=0.0,
        ll_sigma=1.0,
):
    prior = torch.distributions.Normal(loc=prior_mu, scale=prior_sigma)
    mu = (prior_mu * prior_sigma ** -2) + (ll_mu * ll_sigma ** -2)
    mu /= (prior_sigma ** -2 + ll_sigma ** -2)
    var = (prior_sigma **2 * ll_sigma ** 2) / (prior_sigma **2 + ll_sigma ** 2)
    posterior = torch.distributions.Normal(loc=mu, scale=np.sqrt(var))
    return prior, posterior

def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_trajectories_1d(
        traj, 
        prior: torch.distributions.Normal=None, 
        posterior:torch.distributions.Normal=None,
        fig=None,
        ax=None
    ):
    """Plot the 1D trajectories of some selected samples."""
    n = min(2000, traj.shape[1])
    n_flows = min(100, n)
    t = np.linspace(0, 1, traj.shape[0])
    T = np.tile(t, (n, 1)).T
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    # plot start, end and trajectory
    ax.scatter(T[0, :], traj[0, :n, 0], s=10, alpha=0.5, c="black", 
                label="Prior sample z(S)")
    ax.plot(T[:, n_flows], traj[:, :n_flows, 0], alpha=0.1, c="olive",
             label = "_Flow")
    ax.scatter(T[-1, :], traj[-1, :n, 0], s=4, alpha=0.5, c="blue",
                label = "z(0)")

    # plot distributions
    if prior is not None and posterior is not None:
        h_low = np.min(traj[(0, -1), :n, 0])
        h_high = np.max(traj[(0, -1), :n, 0])
        h = torch.linspace(h_low, h_high, 100)
        prior_dist = prior.log_prob(h).exp()
        post_dist = posterior.log_prob(h).exp()
        ax.plot(-prior_dist, h, label="Prior distribution")
        ax.plot(1+post_dist, h, label="Posterior target distribution")

        # approximate pdf
        kde = gaussian_kde(traj[-1, :n, 0])
        ax.plot(1 + kde.pdf(h), h, label="Posterior sampled distribution")

    ax.legend()
    ax.set_xticks([])
    # ax.set_yticks([])
    return fig

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, noise=1.0, reverse=False):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.reverse = reverse
        self.noise = noise

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        if self.reverse:
            return -self.drift(x) + self.score(x)
        return self.drift(x) + self.score(x)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(t) * torch.ones_like(y) * self.noise
