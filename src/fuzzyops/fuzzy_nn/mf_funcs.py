import torch
from typing import List


def _mk_param(val: torch.Tensor) -> torch.nn.Parameter:
    """
    Создает параметр PyTorch из входного значения.

    Args:
        val (torch.Tensor): Входное значение для создания параметра.

    Returns:
        torch.nn.Parameter: Параметр, содержащий входное значение.
    """

    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMemberFunc(torch.nn.Module):
    """
    Представляет нечеткую функцию принадлежности Гаусса.

    Args:
        mu (float): Параметр центра (среднее значение) функции.
        sigma (float): Параметр ширины (стандартное отклонение) функции.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Вычисляет значение функции Гаусса для входного тензора.
    """

    def __init__(self, mu: float, sigma: float):
        super(GaussMemberFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет значение функции принадлежности Гаусса для входного тензора.

        Args:
            x (torch.Tensor): Входной тензор, для которого необходимо вычислить значение.

        Returns:
            torch.Tensor: Значение функции Гаусса.
        """

        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma ** 2))
        return val


class BellMemberFunc(torch.nn.Module):
    """
    Представляет нечеткую функцию принадлежности обощенного колокола.

    Args:
        a (float): Параметр, определяющий ширину функции.
        b (float): Параметр, определяющий наклон функции.
        c (float): Параметр центра функции.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Вычисляет значение функции принадлежности колокола для входного тензора.
    """

    def __init__(self, a: float, b: float, c: float):
        super(BellMemberFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMemberFunc.b_log_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет значение функции колокола для входного тензора.

        Args:
            x (torch.Tensor): Входной тензор, для которого необходимо вычислить значение.

        Returns:
            torch.Tensor: Значение функции колокола.
        """

        dist = torch.pow((x - self.c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))


def make_bell_mfs(a: float, b: float, c_list: List[float]) -> List[BellMemberFunc]:
    """
    Создает список функций колокола.

    Args:
        a (float): Параметр ширины для всех создаваемых функций.
        b (float): Параметр наклона для всех создаваемых функций.
        c_list (List[float]): Список параметров центра для создания функций.

    Returns:
        List[BellMemberFunc]: Список созданных функций колокола.
    """

    return [BellMemberFunc(a, b, c) for c in c_list]


def make_gauss_mfs(sigma: float, mu_list: List[float]) -> List[GaussMemberFunc]:
    """
    Создает список функций Гаусса.

    Args:
        sigma (float): Параметр ширины для всех создаваемых функций.
        mu_list (List[float]): Список параметров центра для создания функций.

    Returns:
        List[GaussMemberFunc]: Список созданных функции Гаусса.
    """
    return [GaussMemberFunc(mu, sigma) for mu in mu_list]
