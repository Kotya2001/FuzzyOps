from collections import OrderedDict
import itertools

import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from mf_funcs import make_gauss_mfs

dtype = torch.float


class FuzzifyVariable(torch.nn.Module):

    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def members(self):
        return self.mfdefs.items()

    def pad_to(self, new_size):
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield mfname, yvals

    def forward(self, x):
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):

    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        self.varnames = ['x{}'.format(i) for i in range(len(varmfs))] if not varnames else list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        return len(self.varmfs)

    @property
    def max_mfs(self):
        return max([var.num_mfs for var in self.varmfs.values()])

    # def __repr__(self):
    #     r = ['Input variables']
    #     for varname, members in self.varmfs.items():
    #         r.append('Variable {}'.format(varname))
    #         for mfname, mfdef in members.mfdefs.items():
    #             r.append('- {}: {}({})'.format(mfname,
    #                                            mfdef.__class__.__name__,
    #                                            ', '.join(['{}={}'.format(n, p.item())
    #                                                       for n, p in mfdef.named_parameters()])))
    #     return '\n'.join(r)

    def forward(self, x):
        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)

        return y_pred


class AntecedentLayer(torch.nn.Module):

    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        mf_count = [var.num_mfs for var in varlist]
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))

    def num_rules(self):
        return len(self.mf_indices)

    # def extra_repr(self, varlist=None):
    #     if not varlist:
    #         return None
    #     row_ants = []
    #     mf_count = [len(fv.mfdefs) for fv in varlist.values()]
    #     for rule_idx in itertools.product(*[range(n) for n in mf_count]):
    #         thisrule = []
    #         for (varname, fv), i in zip(varlist.items(), rule_idx):
    #             thisrule.append('{} is {}'
    #                             .format(varname, list(fv.mfdefs.keys())[i]))
    #         row_ants.append(' and '.join(thisrule))
    #     return '\n'.join(row_ants)

    def forward(self, x):
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))

        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):

    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        return self.coefficients

    @coeff.setter
    def coeff(self, new_coeff):
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def forward(self, x):
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)


class AnfisNet(torch.nn.Module):

    def __init__(self, invardefs, outvarnames):
        super(AnfisNet, self).__init__()
        self.outvarnames = outvarnames
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])

        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            ('consequent', ConsequentLayer(self.num_in, self.num_rules, self.num_out)),
        ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        pass

    def input_variables(self):
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        return self.outvarnames

    # def extra_repr(self):
    #     rstr = []
    #     vardefs = self.layer['fuzzify'].varmfs
    #     rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
    #     for i, crow in enumerate(self.layer['consequent'].coeff):
    #         rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
    #         rstr.append(' ' * 9 + 'THEN {}'.format(crow.tolist()))
    #     return '\n'.join(rstr)

    def forward(self, x):
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        # self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        y_pred = torch.bmm(self.rule_tsk, self.raw_weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


# def _preprocess_data(data: pd.DataFrame, input_features: int, batch_size: int, is_class=True) -> DataLoader:
#     x = torch.Tensor(data.iloc[:, 1:input_features + 1].values)
#     le = LabelEncoder()
#     y = torch.Tensor(le.fit_transform(data.iloc[:, -1].values)).unsqueeze(
#         1) if is_class \
#         else torch.Tensor(data.iloc[:, -1].values)
#     dataset = TensorDataset(x, y)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def _preprocess_data(data: pd.DataFrame, input_features: int, batch_size: int, is_class=True) -> DataLoader:
    x = torch.Tensor(data.iloc[:, 1:input_features + 1].values)
    le = LabelEncoder()
    y = torch.Tensor(le.fit_transform(data.iloc[:, 0].values)).unsqueeze(
        1) if is_class \
        else torch.Tensor(data.iloc[:, 0].values)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _compile_model(x: torch.Tensor, member_num: int, out_num: int) -> AnfisNet:
    input_num = x.shape[1]
    min_values, _ = torch.min(x, dim=0)
    max_values, _ = torch.max(x, dim=0)
    ranges = max_values - min_values
    input_vars = []
    for i in range(input_num):
        sigma = ranges[i] / member_num
        mu_list = torch.linspace(min_values[i], max_values[i], member_num).tolist()
        name = 'x{}'.format(i)
        input_vars.append((name, make_gauss_mfs(sigma, mu_list)))
    out_vars = ['y{}'.format(i) for i in range(out_num)]
    model = AnfisNet(input_vars, out_vars)
    return model


def _class_criterion(inp, target): return torch.nn.CrossEntropyLoss()(inp, target.squeeze().long())


def _reg_criterion(inp, target): return torch.nn.MSELoss()(inp, target.squeeze())


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                                                / y_actual))
    return tot_loss, rmse, perc_loss


def _train_anfis(model, data, optimizer, criterion, epochs=500):
    errors = []

    for t in range(epochs):
        for x, y_actual in data:
            y_pred = model(x)
            loss = criterion(y_pred, y_actual)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        x, y_actual = data.dataset.tensors
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%'
                  .format(t, mse, rmse, perc_loss))


def train(data: pd.DataFrame, in_f: int, member_num: int,
          out_num: int, lr: float, is_class=True):
    train_data = _preprocess_data(data, in_f, 2, is_class)
    x, y = train_data.dataset.tensors
    model = _compile_model(x, member_num, out_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    _train_anfis(model, train_data, optimizer, _reg_criterion)
    y_pred = model(x)
    print(_reg_criterion(y_pred, y))
    # nc = torch.sum(y.squeeze().long() == torch.argmax(y_pred, dim=1))
    # tot = len(x)
    # print('{} of {} correct (={:5.2f}%)'.format(nc, tot, nc * 100 / tot))
    return model


print(type(torch.device('cuda:0')))
model = train(pd.read_csv("tests/data/sales.csv"), 2, 5, 1, 0.0003, False)
train_data = _preprocess_data(pd.read_csv("tests/data/sales.csv"), 2, 2, False)
for x, y in train_data:
    print(x)
    print(y)
    y_p = model(x)
    print(y_p)
    break
    # print(_reg_criterion(y_p, y))
    # print(y)
    # print(model(x))
    # print(y.squeeze())
    # print(torch.argmax(y_p, dim=1))
    # break
