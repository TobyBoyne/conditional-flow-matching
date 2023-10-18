"""Provides classes to sample from parameters as well as distributions."""

import torch
import matplotlib.pyplot as plt

def get_conjugate_gaussian_parameters(
        prior_mu=0.0,
        prior_sigma=1.0,
        ll_mu=0.0,
        ll_sigma=1.0,
):
    mu = (prior_mu * prior_sigma ** -2) + (ll_mu * ll_sigma ** -2)
    mu /= (prior_sigma ** -2 + ll_sigma ** -2)
    var = (prior_sigma **2 * ll_sigma ** 2) / (prior_sigma **2 + ll_sigma ** 2)
    return mu, torch.sqrt(var)

class GaussianPairContainer:
    """Contains a prior/posterior pair.
    
    Can sample from hyperparameters"""
    num_params = 2
    def __init__(self, 
                prior_mu: torch.distributions.Distribution, 
                prior_sigma: torch.distributions.Distribution,
                ll_mu: float,
                ll_sigma: float
        ):
        self.dist = torch.distributions.Normal(loc=0., scale=1.)
        self.mu_dist = prior_mu
        self.sigma_dist = prior_sigma
        self.ll_mu = ll_mu
        self.ll_sigma = ll_sigma

    def sample_prior(self, sample_shape, mu_fixed=None, sigma_fixed=None):
        x = self.dist.sample(sample_shape)
        mu, sigma = self._sample_parameters(
            sample_shape, mu_fixed, sigma_fixed
        )

        sample = x * sigma + mu

        return torch.concatenate((sample, mu, sigma), dim=-1)
    
    def sample_pair(self, sample_shape):
        """Sample the prior and posterior together"""
        x = self.dist.sample(sample_shape)
        mu, sigma = self._sample_parameters(sample_shape)

        prior_sample = x * sigma + mu

        mu_post, sigma_post = self._calc_posterior_parameters(mu, sigma)
        y = self.dist.sample(sample_shape)
        post_sample = y * sigma_post + mu_post

        prior = torch.concatenate((prior_sample, mu, sigma), dim=-1)
        post = torch.concatenate((post_sample, mu, sigma), dim=-1)

        return prior, post

    
    def _sample_parameters(self, sample_shape, mu_fixed=None, sigma_fixed=None):
        if mu_fixed is None:
            mu = self.mu_dist.sample(sample_shape)
        else:
            mu = torch.ones(sample_shape) * mu_fixed

        if sigma_fixed is None:
            sigma = self.sigma_dist.sample(sample_shape)
        else:
            sigma = torch.ones(sample_shape) * sigma_fixed
        
        return mu, sigma
    
    def _calc_posterior_parameters(self, mu, sigma):
        return get_conjugate_gaussian_parameters(
            mu, sigma, self.ll_mu, self.ll_sigma
        )
    
    def fixed_distributions(self, mu, sigma):
        """Return samples from the distributions for fixed hyperparameters"""
        prior = torch.distributions.Normal(loc=mu, scale=sigma)
        mu_post, sigma_post = self._calc_posterior_parameters(mu, sigma)
        posterior = torch.distributions.Normal(loc=mu_post, scale=sigma_post)
        return prior, posterior
    

if __name__ == "__main__":
    mu = torch.distributions.Uniform(0, 0.01)
    sigma = torch.distributions.Uniform(0.9, 1.1)
    g = GaussianPairContainer(mu, sigma, 2.0, 1.0)

    # print(g.sample((10,)).shape)
    prior, post = g.sample_pair((1000, 1))
    plt.hist(prior[:, 0], alpha=0.7)
    plt.hist(post[:, 0], alpha=0.7)
    plt.show()