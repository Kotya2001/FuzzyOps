import torch
from typing import List


def _mk_param(val: torch.Tensor) -> torch.nn.Parameter:
    """
    Creates the PyTorch parameter from the input value

    Args:
        val (torch.Tensor): The input value for creating the parameter

    Returns:
        torch.nn.Parameter: A parameter containing the input value
    """

    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMemberFunc(torch.nn.Module):
    """
    Represents a fuzzy Gauss membership function

    Args:
        mu (float): The parameter of the center (the average value) of the function
        sigma (float): The parameter of the width (standard deviation) of the function
    """

    def __init__(self, mu: float, sigma: float):
        super(GaussMemberFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the value of the Gauss membership function for the input tensor

        Args:
            x (torch.Tensor): The input tensor for which the value needs to be calculated

        Returns:
            torch.Tensor: The value of the Gaussian function
        """

        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma ** 2))
        return val


class BellMemberFunc(torch.nn.Module):
    """
    It represents an indistinct function of belonging to a decorated bell

    Args:
        a (float): A parameter that defines the width of the function
        b (float): A parameter that defines the slope of the function
        c (float): Parameter of the function center
    """

    def __init__(self, a: float, b: float, c: float):
        super(BellMemberFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMemberFunc.b_log_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the value of the bell function for the input tensor

        Args:
            x (torch.Tensor): The input tensor for which the value needs to be calculated

        Returns:
            torch.Tensor: The value of the bell function
        """

        dist = torch.pow((x - self.c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))


def make_bell_mfs(a: float, b: float, c_list: List[float]) -> List[BellMemberFunc]:
    """
    Creates a list of bell functions

    Args:
        a (float): The width parameter for all created functions
        b (float): The tilt parameter for all created functions
        c_list (List[float]): A list of center parameters for creating functions

    Returns:
        List[BellMemberFunc]: A list of created bell functions
    """

    return [BellMemberFunc(a, b, c) for c in c_list]


def make_gauss_mfs(sigma: float, mu_list: List[float]) -> List[GaussMemberFunc]:
    """
    Creates a list of Gaussian functions

    Args:
        sigma (float): The width parameter for all created functions
        mu_list (List[float]): A list of center parameters for creating functions

    Returns:
        List[GaussMemberFunc]: A list of created Gauss functions
    """
    return [GaussMemberFunc(mu, sigma) for mu in mu_list]
