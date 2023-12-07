import torch 
import torch.nn as nn

class AndersonDarlingDistance(nn.Module):
    """
    Anderson-Darling Distance module for measuring the dissimilarity between two distributions.
    """
    def __init__(self):
        super(AndersonDarlingDistance, self).__init__()

    def marginal(self, x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
        """
        Computes the Anderson-Darling statistic for a single dimension.
        Args:
            x (torch.Tensor): Observed values.
            xhat (torch.Tensor): Sorted and observed values from another distribution.
        Returns:
            torch.Tensor: Anderson-Darling statistic for the given dimension.
        """
        size = x.size(0)
        device_ = x.device
        u = (torch.sum(xhat[:, None] >= x, dim=1) + 1) / (size + 2)
        w = -torch.mean((2 * torch.arange(1, size + 1, device=device_) - 1) * (torch.log(u) + torch.log(1 - u.flip(dims=(0,)))))
        return w - size

    def forward(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the Anderson-Darling distance for multiple dimensions.

        Args:
            X (torch.Tensor): Observed values.
            X_hat (torch.Tensor): Observed values from another distribution.
        Returns:
            torch.Tensor: Anderson-Darling distance.
        """
        dim = X.size(1)
        dist = 0.
        x, xhat = X, torch.sort(X_hat, dim=0).values
        for i in range(dim):
            w = self.marginal(x[:, i], xhat[:, i])
            dist += w
        dist /= dim
        return dist


class KendallDependenceMetric(nn.Module):
    """
    Kendall Dependence Metric module for measuring the dependence between two distributions.
    """
    def __init__(self):
        super(KendallDependenceMetric, self).__init__()

    def dependence(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the dependence values for each observation in the input.
        Args:
            X (torch.Tensor): Input values.
        Returns:
            torch.Tensor: Dependence values.
        """
        size, dim = X.size()
        device_ = X.device
        Z = torch.zeros(size)
        for i in range(size):
            u = (X[i, :] > X).all(dim=1) & (torch.arange(size, device=device_) != i)
            t = u.sum().item()
            Z[i] = t
        Z /= (size - 1)
        return Z

    def forward(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kendall dependence metric between two distributions.
        Args:
            X (torch.Tensor): Observed values.
            X_hat (torch.Tensor): Observed values from another distribution.
        Returns:
            torch.Tensor: Kendall dependence metric.
        """
        Z, Zhat = self.dependence(X), self.dependence(X_hat)
        Z, Zhat = torch.sort(Z).values, torch.sort(Zhat).values
        return torch.norm(Z - Zhat, p=1) / Z.size(0)
