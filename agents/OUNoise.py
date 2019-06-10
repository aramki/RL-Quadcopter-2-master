import numpy as np

class OUNoise:
    """
        Ornstein-Uhlenbeck process.
    """
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.state = np.copy(self.mu)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def __call__(self):
        return self.sample()

    def update_mu(self, target):
        self.mu = target