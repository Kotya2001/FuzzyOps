import torch


def _mk_param(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMemberFunc(torch.nn.Module):

    def __init__(self, mu, sigma):
        super(GaussMemberFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma ** 2))
        return val


class BellMemberFunc(torch.nn.Module):

    def __init__(self, a, b, c):
        super(BellMemberFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMemberFunc.b_log_hook)

    def forward(self, x):
        dist = torch.pow((x - self.c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))


def make_bell_mfs(a, b, c_list):
    return [BellMemberFunc(a, b, c) for c in c_list]


def make_gauss_mfs(sigma, mu_list):
    return [GaussMemberFunc(mu, sigma) for mu in mu_list]
