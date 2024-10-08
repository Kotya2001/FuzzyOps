from collections import OrderedDict
import itertools
from typing import Union, Callable

import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .mf_funcs import make_gauss_mfs, GaussMemberFunc, BellMemberFunc, make_bell_mfs

dtype = torch.float

funcs = Union[GaussMemberFunc, BellMemberFunc]
task_types = {"classification": "classification", "regression": "regression"}
funcs_type = {"gauss": "gauss", "bell": "bell"}


class _FuzzyVar(torch.nn.Module):
    """
    Класс слоя для фаззификации входных переменных

    """

    def __init__(self, mfdefs: list[funcs]):
        super(_FuzzyVar, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self) -> int:
        """
        Возвращает число термов для каждой входной переменной
        :return: int
        """
        return len(self.mfdefs)

    def members(self) -> torch.nn.ModuleDict.items:
        """
        Возвращает нечеткий терм с ее функцией принадлежности в формате torch.nn.ModuleDict.items
        :return: torch.nn.ModuleDict.items
        """
        return self.mfdefs.items()

    def pad_to(self, new_size: int) -> None:
        """
        Метод устанавливает значение padding для того, чтобы выравнять матрицы после фаззификации
        (на случай, если у кол-во термов у переменных не совпадают)
        :return: None
        """
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x: torch.Tensor) -> None:
        """
        Метод для фаззикиции переданных в класс значений
        :return: None
        """
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield mfname, yvals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод для фаззикиции переданных в класс значений

        :param x: torch.Tensor
        :return: None
        """
        predictions = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            predictions = torch.cat([predictions, torch.zeros(x.shape[0], self.padding)], dim=1)
        return predictions


class _FuzzyLayer(torch.nn.Module):
    """
    Класс слоя для объединения всех нечетких термов

    """

    def __init__(self, varmfs: list[_FuzzyVar], varnames=None):
        super(_FuzzyLayer, self).__init__()
        self.varnames = ['x{}'.format(i) for i in range(len(varmfs))] if not varnames else list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self) -> int:
        """
        Свойство - возвращает число входных переменных
        :return: int
        """
        return len(self.varmfs)

    @property
    def max_mfs(self) -> int:
        """
        Свойство - возвращает максимальное число входных термов среди всех переменных
        :return: int
        """
        return max([var.num_mfs for var in self.varmfs.values()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод для конкатенации нечетких термов в один тензор

        :param x: torch.Tensor
        :return: torch.Tensor
        """
        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class _AntecedentLayer(torch.nn.Module):
    """
    Класс слоя антецедентов правил

    """

    def __init__(self, varlist: list[_FuzzyVar]):
        super(_AntecedentLayer, self).__init__()
        mf_count = [var.num_mfs for var in varlist]
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))
        print(self.mf_indices.shape)

    def num_rules(self) -> int:
        """
        Метод возвращает количество нечетких правил
        :return: int
        """
        return len(self.mf_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод фоомирует антеценденты соответствующего правила,
        в итоге формируется степень выполнения правила (произведение входных сигналом)

        :param x: torch.Tensor
        :return: torch.Tensor
        """
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1)).to(x.device)
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)
        return rules


class _ConsequentLayer(torch.nn.Module):

    def __init__(self, d_in, d_rule, d_out):
        super(_ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self) -> torch.Tensor:
        """
        Свойство, которое возвращает веса
        :return:
        """
        return self.coefficients

    @coeff.setter
    def coeff(self, new_coeff: torch.Tensor) -> None:
        """
        Сеттер для установки новых весов
        :param new_coeff: torch.Tensor
        :return: None
        """
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод дабавляет смещение входным данным, а затем производится матричное умножение весов на входные данные
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)


class _NN(torch.nn.Module):

    def __init__(self, invardefs, outvarnames):
        super(_NN, self).__init__()
        self.outvarnames = outvarnames
        varnames = [v for v, _ in invardefs]
        mfdefs = [_FuzzyVar(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])

        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', _FuzzyLayer(mfdefs, varnames)),
            ('rules', _AntecedentLayer(mfdefs)),
            ('consequent', _ConsequentLayer(self.num_in, self.num_rules, self.num_out)),
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

    def forward(self, x):
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


class Model:
    task_names = {"regression": "регрессии", "classification": "классификации"}

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 n_terms: list[int], n_out_vars: int, lr: float,
                 task_type: str, batch_size: int, member_func_type: str,
                 epochs: int,
                 verbose: bool = False,
                 device: torch.device = torch.device("cpu")):
        self.X = X
        self.Y = Y
        self.n_input_features = X.shape[1]
        self.n_terms = n_terms
        self.n_out_vars = n_out_vars
        self.lr = lr
        self.task_type = task_type
        self.batch_size = batch_size
        self.member_func_type = member_func_type
        self.device = device
        self.epochs = epochs
        self.scores = []
        self.verbose = verbose
        self.model = None

        assert self.task_type in list(task_types.keys()), \
            f"{self.task_type} некорректен для данной задачи," \
            f" корректы следующие {' '.join(list(task_types.keys()))}"

        print(f"Создание экземпляра класса для задачи {self.task_names[self.task_type]} " \
              f"со следующими гиперпараметрами\nЧисло входных признаков: {self.n_input_features}\n" \
              f"Число термов: {self.n_terms}\nЧисло выходных переменных: {self.n_out_vars}\n" \
              f"Шаг обучения: {self.lr}\nРазмер подвыборки: {self.batch_size}\n" \
              f"Тип функции принадлежности: {self.member_func_type}\n" \
              f"Размер подвыборки для обучения: {self.batch_size}\n")

    def __str__(self):
        return f"Создание экземпляра класса для задачи {self.task_names[self.task_type]} " \
               f"со следующими гиперпараметрами\nЧисло входных признаков: {self.n_input_features}\n" \
               f"Число термов: {self.n_terms}\nЧисло выходных переменных: {self.n_out_vars}\n" \
               f"Шаг обучения: {self.lr}\nРазмер подвыборки: {self.batch_size}\n" \
               f"Тип функции принадлежности: {self.member_func_type}\n" \
               f"Размер подвыборки для обучения: {self.batch_size}\n"

    def __repr__(self):
        return f"Создание экземпляра класса для задачи {self.task_names[self.task_type]} " \
               f"со следующими гиперпараметрами\nЧисло входных признаков: {self.n_input_features}\n" \
               f"Число термов: {self.n_terms}\nЧисло выходных переменных: {self.n_out_vars}\n" \
               f"Шаг обучения: {self.lr}\nРазмер подвыборки: {self.batch_size}\n" \
               f"Тип функции принадлежности: {self.member_func_type}\n" \
               f"Размер подвыборки для обучения: {self.batch_size}\n"

    def __preprocess_data(self) -> DataLoader:
        x = torch.Tensor(self.X)
        if self.device:
            x = x.to(self.device)
        le = LabelEncoder()

        y = torch.Tensor(le.fit_transform(self.Y)).unsqueeze(
            1) if self.task_type == task_types["classification"] \
            else torch.Tensor(self.Y)
        if self.device:
            y = y.to(self.device)

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def __gauss_func(self, x: torch.Tensor) -> tuple[list]:
        input_num = x.shape[1]
        min_values, _ = torch.min(x, dim=0)
        max_values, _ = torch.max(x, dim=0)
        ranges = max_values - min_values
        input_vars = []
        for i in range(input_num):
            sigma = ranges[i] / self.n_terms[i]
            mu_list = torch.linspace(min_values[i], max_values[i], self.n_terms[i]).tolist()
            name = 'x{}'.format(i)
            input_vars.append((name, make_gauss_mfs(sigma, mu_list)))
        out_vars = ['y{}'.format(i) for i in range(self.n_out_vars)]

        return input_vars, out_vars

    def __bell_func(self, x: torch.Tensor) -> tuple[list]:
        input_num = x.shape[1]
        min_values, _ = torch.min(x, dim=0)
        max_values, _ = torch.max(x, dim=0)
        input_vars = []
        for i in range(input_num):
            a, b = min_values / self.n_terms[i], max_values / self.n_terms[i]
            c_list = torch.linspace(min_values[i], max_values[i], self.n_terms[i]).tolist()
            name = 'x{}'.format(i)
            input_vars.append((name, make_bell_mfs(a, b, c_list)))
        out_vars = ['y{}'.format(i) for i in range(self.n_out_vars)]

        return input_vars, out_vars

    def __compile(self, x: torch.Tensor) -> _NN:
        input_vars, out_vars = self.__gauss_func(x) if self.member_func_type == funcs_type[
            "gauss"] else self.__bell_func(x)
        model = _NN(input_vars, out_vars)
        # Перенос модели на device
        if self.device:
            model.to(self.device)
        return model

    @staticmethod
    def __class_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
        return torch.nn.CrossEntropyLoss()(inp, target.squeeze().long())

    @staticmethod
    def __reg_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
        return torch.nn.MSELoss()(inp, target.squeeze())

    @staticmethod
    def __calc_reg_score(preds: torch.Tensor, y_actual: torch.Tensor) -> float:
        with torch.no_grad():
            tot_loss = F.mse_loss(preds, y_actual)

        return tot_loss

    @staticmethod
    def __calc_class_score(preds: torch.Tensor, y_actual: torch.Tensor, x: torch.Tensor) -> float:
        with torch.no_grad():
            corr = torch.sum(y_actual.squeeze().long() == torch.argmax(preds, dim=1))
            total = len(x)
        return corr * 100 / total

    def __train_loop(self, data: DataLoader, model: _NN,
                     criterion: Callable, calc_score: Callable,
                     optimizer: torch.optim.Adam):

        score_class = 0
        score_reg = 100000000000

        for t in range(self.epochs):
            for x, y_actual in data:
                y_pred = model(x)
                loss = criterion(y_pred, y_actual)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            x, y_actual = data.dataset.tensors
            y_pred = model(x)

            score = calc_score(y_pred, y_actual) if self.task_type == "regression" \
                else calc_score(y_pred, y_actual, x)

            if self.task_type == "regression":
                if score < score_reg:
                    self.model = model
            else:
                if score > score_class:
                    self.model = model

            self.scores.append(score)
            if self.epochs < 30 or t % 10 == 0:
                if self.verbose:
                    print(f"epoch: {t}, score: {score}")

    def train(self) -> _NN:
        train_data = self.__preprocess_data()
        x, y = train_data.dataset.tensors
        model = self.__compile(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = self.__class_criterion if self.task_type == task_types["classification"] else \
            self.__reg_criterion
        calc_error = self.__calc_class_score if self.task_type == task_types["classification"] else \
            self.__calc_reg_score

        self.__train_loop(train_data, model, criterion, calc_error, optimizer)

        return self.model

    def save_model(self, path: str):
        if self.model:
            torch.save(self.model.state_dict(), path)
        else:
            raise Exception("Модель не обучена")
