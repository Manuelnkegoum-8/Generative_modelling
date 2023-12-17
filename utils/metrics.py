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

    def forward(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kendall dependence metric between two distributions.
        Args:
            X (torch.Tensor): Observed values.
            X_hat (torch.Tensor): Observed values from another distribution.
        Returns:
            torch.Tensor: Kendall dependence metric.
        """
        n = len(X)
        assert len(X_hat) == n, "Both lists have to be of equal length"

        i, j = torch.meshgrid(torch.arange(n), torch.arange(n))
        a = torch.argsort(X,axis=0)
        b = torch.argsort(X_hat,axis=0)

        ndisordered = torch.logical_or(torch.logical_and(a[i] < a[j], b[i] > b[j]), torch.logical_and(a[i] > a[j], b[i] < b[j])).sum().item()

        return torch.tensor(ndisordered / (n * (n - 1)))
