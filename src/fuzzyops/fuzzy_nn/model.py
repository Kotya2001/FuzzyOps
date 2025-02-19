from collections import OrderedDict
import itertools
from typing import Union, Callable, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .mf_funcs import make_gauss_mfs, GaussMemberFunc, BellMemberFunc, make_bell_mfs

dtype = torch.float

funcs = Union[GaussMemberFunc, BellMemberFunc]
task_types = {"classification": "classification", "regression": "regression"}
funcs_type = {"gauss": "gauss", "bell": "bell"}


class _FuzzyVar(torch.nn.Module):
    """
    Класс слоя для фаззификации входных переменных.

    Attributes:
        mfdefs (torch.nn.ModuleDict): Словарь функций принадлежности для фаззификации.
        padding (int): Значение padding для выравнивания матриц после фаззификации.

    Args:
        mfdefs (List[funcs]): Список функций принадлежности для входной переменной.

    Methods:
        num_mfs() -> int:
            Возвращает число термов для каждой входной переменной.

        members() -> torch.nn.ModuleDict.items:
            Возвращает нечеткий терм с его функцией принадлежности.

        pad_to(new_size: int) -> None:
            Устанавливает значение padding для выравнивания матриц.

        fuzzify(x: torch.Tensor) -> None:
            Фаззикация переданных значений.

        forward(x: torch.Tensor) -> torch.Tensor:
            Выполняет фаззификацию переданных значений и возвращает результаты.
    """

    def __init__(self, mfdefs: List[funcs]):
        super(_FuzzyVar, self).__init__()
        if isinstance(mfdefs, list):
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self) -> int:
        """
        Возвращает число термов для каждой входной переменной.

        Returns:
            int: Число термов.
        """

        return len(self.mfdefs)

    def members(self) -> torch.nn.ModuleDict.items:
        """
        Возвращает нечеткий терм с его функцией принадлежности.

        Returns:
            torch.nn.ModuleDict.items: Элементы словаря нечетких термов и функций принадлежности.
        """

        return self.mfdefs.items()

    def pad_to(self, new_size: int) -> None:
        """
        Метод устанавливает значение padding для выравнивания матриц после фаззификации.

        Args:
            new_size (int): Новое значение для padding.
        """

        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x: torch.Tensor) -> None:
        """
        Метод для фаззификации переданных значений.

        Args:
            x (torch.Tensor): Входные значения для фаззификации.

        Yields:
            Tuple[str, torch.Tensor]: Имя функции принадлежности и ее значения.
        """

        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield mfname, yvals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет фаззификацию переданных значений и возвращает результаты.

        Args:
            x (torch.Tensor): Входные значения для фаззификации.

        Returns:
            torch.Tensor: Результаты фаззификации, включая padding, если это необходимо.
        """

        predictions = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            predictions = torch.cat([predictions, torch.zeros(x.shape[0], self.padding)], dim=1)
        return predictions


class _FuzzyLayer(torch.nn.Module):
    """
    Класс слоя для объединения всех нечетких термов.

    Attributes:
        varmfs (torch.nn.ModuleDict): Словарь нечетких переменных.
        varnames (List[str]): Имена входных переменных.

    Args:
        varmfs (List[_FuzzyVar]): Список нечетких переменных.
        varnames (List[str], optional): Имена переменных (если не указаны, используются x0, x1 и т.д.).

    Methods:

        num_in() -> int:
            Возвращает число входных переменных.

        max_mfs() -> int:
            Возвращает максимальное число входных термов среди всех переменных.

        forward(x: torch.Tensor) -> torch.Tensor:
            Метод для конкатенации нечетких термов в один тензор.
        """

    def __init__(self, varmfs: List[_FuzzyVar], varnames=None):
        super(_FuzzyLayer, self).__init__()
        self.varnames = ['x{}'.format(i) for i in range(len(varmfs))] if not varnames else list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self) -> int:
        """
        Свойство, возвращающее число входных переменных.

        Returns:
            int: Число входных переменных.
        """

        return len(self.varmfs)

    @property
    def max_mfs(self) -> int:
        """
        Свойство, возвращающее максимальное число входных термов среди всех переменных.

        Returns:
            int: Максимальное число входных термов.
        """

        return max([var.num_mfs for var in self.varmfs.values()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод для конкатенации нечетких термов в один тензор.

        Args:
            x (torch.Tensor): Входные значения, которые должны быть обработаны.

        Returns:
            torch.Tensor: Конкатенированный тензор нечетких термов.

        Raises:
            AssertionError: Если количество входных значений не совпадает с ожидаемым.
        """

        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class _AntecedentLayer(torch.nn.Module):
    """
    Класс слоя антецедентов правил нечеткой логики.

    Этот класс отвечает за создание нечетких правил, используя антецеденты
    (функции принадлежности), которые определяются входными нечеткими переменными.
    Он генерирует правила как произведение значений функций принадлежности для
    соответствующих входных сигналов.

    Attributes:
        mf_indices (torch.Tensor): Индексы функций принадлежности для сформированных нечетких правил.

    Args:
        varlist (List[_FuzzyVar]): Список нечетких переменных, каждая из которых содержит свои функции принадлежности.

    Methods:
        num_rules() -> int:
            Возвращает количество сформированных нечетких правил.

        forward(x: torch.Tensor) -> torch.Tensor:
            Формирует антеценденты соответствующих правил и возвращает степени выполнения правил.
    """

    def __init__(self, varlist: List[_FuzzyVar]):
        super(_AntecedentLayer, self).__init__()
        mf_count = [var.num_mfs for var in varlist]
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))

    def num_rules(self) -> int:
        """
        Метод возвращает количество нечетких правил.

        Returns:
            int: Количество нечетких правил.
        """

        return len(self.mf_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Формирует антеценденты соответствующего правила и вычисляет степени выполнения правил.

        Каждое правило определяется произведением значений функций принадлежности,
        связанных с входными сигналами.

        Args:
            x (torch.Tensor): Входные значения, содержащие результаты фаззификации переменных,
                              ожидаемые размерности (batch_size, num_mfs, feature_size).

        Returns:
            torch.Tensor: Степени выполнения правил для нечетких правил,
                          размерности (batch_size, num_rules).
        """

        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1)).to(x.device)
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        rules = torch.prod(ants, dim=2)
        return rules


class _ConsequentLayer(torch.nn.Module):
    """
        Класс слоя консеквентов нечеткой логики.

        Этот класс отвечает за вычисление выходных значений нечеткой системы
        на основе заданных правил и входных данных. Он включает в себя
        коэффициенты (веса), которые используются для линейной комбинации
        входных данных для получения итоговых значений.

        Attributes:
            coefficients (torch.nn.Parameter): Параметры слоя, представляющие веса
            для линейной комбинации входных данных.

        Args:
            d_in (int): Размерность входных данных.
            d_rule (int): Количество нечетких правил.
            d_out (int): Размерность выходных данных.

        Properties:
            coeff (torch.Tensor): Возвращает коэффициенты (веса) слоя.

        Methods:
            forward(x: torch.Tensor) -> torch.Tensor:
                Вычисляет выходные значения на основе входных данных и весов.
        """

    def __init__(self, d_in: int, d_rule: int, d_out: int):
        super(_ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self) -> torch.Tensor:
        """
        Свойство, возвращающее веса слоя.

        Returns:
            torch.Tensor: Текущие коэффициенты (веса) слоя.
        """

        return self.coefficients

    @coeff.setter
    def coeff(self, new_coeff: torch.Tensor) -> None:
        """
        Сеттер для установки новых весов.

        Args:
            new_coeff (torch.Tensor): Новые коэффициенты для слоя.

        Raises:
            AssertionError: Если форма новых коэффициентов не совпадает с формой текущих коэффициентов.
        """

        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет выходные значения на основе входных данных и весов.

        Метод добавляет единичное смещение к входным данным,
        а затем выполняет матричное умножение весов на входные данные
        для получения прогнозируемых выходных значений.

        Args:
            x (torch.Tensor): Входные данные, имеющие размерность (batch_size, d_in).

        Returns:
            torch.Tensor: Выходные значения, имеющие размерность (batch_size, d_out).
        """

        x_plus = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)


class _NN(torch.nn.Module):
    """
    Класс нечеткой нейронной сети, которая комбинирует нечеткие правила и линейные модели.

    Этот класс реализует нечеткую нейронную сеть, состоящую из трех основных слоев:
    1. Слой фаззификации входных переменных.
    2. Слой антецедентов для формирования правил.
    3. Слой последствий для вычисления выходных значений на основе правил.

    Attributes:
        outvarnames (List[str]): Имена выходных переменных.
        num_in (int): Число входных переменных.
        num_rules (int): Общее количество нечетких правил.
        layer (torch.nn.ModuleDict): Словарь слоев сети, включая слои фаззификации,
                                      антецедентов и последствий.

    Args:
        invardefs (List[Tuple[str, List[funcs]]]): Список кортежей,
            где каждый кортеж состоит из имени входной переменной и
            списка функций принадлежности для этой переменной.
        outvarnames (List[str]): Список имен выходных переменных.

    Properties:
        num_out (int): Возвращает количество выходных переменных.
        coeff (torch.Tensor): Возвращает коэффициенты слоя последствий.

    Methods:
        fit_coeff(x: torch.Tensor, y_actual: torch.Tensor) -> None:
            Метод для обучения (фитинга) весов (коэффициентов) слоя последствий.
        input_variables() -> torch.nn.ModuleDict.items:
            Возвращает нечеткие входные переменные и их функции принадлежности.
        output_variables() -> List[str]:
            Возвращает имена выходных переменных.
        forward(x: torch.Tensor) -> torch.Tensor:
            Выполняет прямое распространение и возвращает предсказанные выходные значения.
    """

    def __init__(self, invardefs: List[Tuple[str, List[funcs]]],
                 outvarnames: List[str]):
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
    def num_out(self) -> int:
        """
        Возвращает количество выходных переменных.

        Returns:
            int: Количество выходных переменных.
        """

        return len(self.outvarnames)

    @property
    def coeff(self) -> torch.Tensor:
        """
        Возвращает коэффициенты слоя консеквентов.

        Returns:
            torch.Tensor: Текущие коэффициенты слоя консеквентов.
        """

        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff: torch.Tensor):
        """
        Сеттер для установки новых коэффициентов.

        Args:
            new_coeff (torch.Tensor): Новые коэффициенты для слоя консеквентов.
        """

        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x: torch.Tensor, y_actual: torch.Tensor):
        """
        Метод для обучения весов (коэффициентов) слоя последствий.

        Args:
            x (torch.Tensor): Входные данные, используемые для обучения.
            y_actual (torch.Tensor): Фактические выходные данные, с которыми необходимо сравнивать предсказания.
        """

        pass

    def input_variables(self) -> torch.nn.ModuleDict.items:
        """
        Возвращает нечеткие входные переменные и их функции принадлежности.

        Returns:
            torch.nn.ModuleDict.items: Элементы словаря нечетких переменных и функций принадлежности.
        """

        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self) -> List[str]:
        """
        Возвращает имена выходных переменных.

        Returns:
            List[str]: Имена выходных переменных.
        """

        return self.outvarnames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямое распространение и возвращает предсказанные выходные значения.

        Входные данные передаются через слой фаззификации, затем обрабатываются
        в слое антецедентов для вычисления степеней выполнения правил, и наконец,
        используются в слое последствий для получения итоговых выходных значений.

        Args:
            x (torch.Tensor): Входные значения, имеющие размерность (batch_size, num_in).

        Returns:
            torch.Tensor: Предсказанные выходные значения, имеющие размерность (batch_size, num_out).
        """

        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


class Model:
    """
    Класс для создания и обучения модели нечеткой логики.

    Этот класс предназначен для выполнения задач регрессии и классификации с использованием
    нечеткой логики. Он принимает входные данные, определяет параметры модели и
    осуществляет предварительную обработку данных.

    Attributes:
        task_names (dict): Словарь, связывающий типы задач с их текстовыми представлениями.
        X (np.ndarray): Входные данные модели.
        Y (np.ndarray): Выходные данные модели.
        n_input_features (int): Число входных признаков.
        n_terms (list[int]): Список, содержащий число термов для каждой входной переменной.
        n_out_vars (int): Количество выходных переменных.
        lr (float): Шаг обучения для оптимизации.
        task_type (str): Тип задачи ("regression" или "classification").
        batch_size (int): Размер подвыборки для обучения.
        member_func_type (str): Тип функции принадлежности.
        device (torch.device): Устройство, на котором будет выполняться модель (например, "cpu" или "cuda").
        epochs (int): Количество эпох для обучения модели.
        scores (list): Список для сохранения показателей модели.
        verbose (bool): Флаг "подробного" вывода информации о процессе обучения.
        model (torch.nn.Module): Модель для обучения, в настоящее время не определена.

    Args:
        X (np.ndarray): Входные данные для модели.
        Y (np.ndarray): Целевые значения для модели.
        n_terms (list[int]): Число термов для каждой входной переменной.
        n_out_vars (int): Количество выходных переменных.
        lr (float): Шаг обучения.
        task_type (str): Тип задачи: 'regression' или 'classification'.
        batch_size (int): Размер подвыборки для обучения.
        member_func_type (str): Тип функции принадлежности.
        epochs (int): Количество эпох для обучения модели.
        verbose (bool): Уровень подробности вывода (по умолчанию False).
        device (torch.device): Устройство для вычислений, по умолчанию "cpu".

    Methods:
        __str__() -> str:
            Строковое представление объекта модели.
        __repr__() -> str:
            Описание объекта модели для отладки.
        __preprocess_data() -> DataLoader:
            Предварительная обработка данных и создание DataLoader.
        __gauss_func(x: torch.Tensor) -> Tuple[List]:
            Генерирует параметры для гауссовских функций принадлежности на основе входных данных.
        __bell_func(x: torch.Tensor) -> tuple[list]:
            Генерирует параметры для колоколообразных функций принадлежности на основе входных данных.
        __compile(x: torch.Tensor) -> _NN:
            Компилирует модель нечеткой нейронной сети на основе выбранного типа функции принадлежности.
        __class_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
            Вычисляет значение функции потерь для задачи классификации.
        __reg_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
            Вычисляет значение функции потерь для задачи регрессии.
        __calc_reg_score(preds: torch.Tensor, y_actual: torch.Tensor) -> float:
            Вычисляет оценку модели для задачи регрессии.
        __calc_class_score(preds: torch.Tensor, y_actual: torch.Tensor, x: torch.Tensor) -> float:
            Вычисляет точность модели для задачи классификации.
        __train_loop(data: DataLoader, model: _NN, criterion: Callable, calc_score: Callable,
            optimizer: torch.optim.Adam) -> None:
            Основной цикл обучения модели.
        train() -> _NN:
            Запускает процесс обучения модели.
        save_model(path: str) -> None:
            Сохраняет состояние обученной модели в файл.

    """

    task_names = {"regression": "регрессии", "classification": "классификации"}

    def __init__(self, X: np.ndarray, Y: np.ndarray,
                 n_terms: list[int], n_out_vars: int, lr: float,
                 task_type: str, batch_size: int, member_func_type: str,
                 epochs: int,
                 verbose: bool = False,
                 device: str = "cpu"):
        self.X = X
        self.Y = Y
        self.n_input_features = X.shape[1]
        self.n_terms = n_terms
        self.n_out_vars = n_out_vars
        self.lr = lr
        self.task_type = task_type
        self.batch_size = batch_size
        self.member_func_type = member_func_type
        self.device = torch.device(device)
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
        """
        Предварительная обработка данных и создание DataLoader.

        Преобразует входные данные и целевые значения в тензоры,
        выполняет кодирование выходных переменных для классификации
        и создает объект DataLoader для предоставления данных в батчах.

        Returns:
            DataLoader: Объект DataLoader, содержащий предварительно обработанные данные.
        """

        x = torch.Tensor(self.X)
        if self.device:
            x = x.to(self.device)
        # le = LabelEncoder()

        y = torch.Tensor(self.Y)
        if self.device:
            y = y.to(self.device)

        # y = torch.Tensor(le.fit_transform(self.Y)).unsqueeze(
        #     1) if self.task_type == task_types["classification"] \
        #     else torch.Tensor(self.Y)
        # if self.device:
        #     y = y.to(self.device)

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def __gauss_func(self, x: torch.Tensor) -> Tuple[List]:
        """
        Генерирует параметры для гауссовских функций принадлежности на основе входных данных.

        Вычисляет минимумы, максимумы и диапазоны для каждой входной переменной
        и создает цента и сигмы для гауссовых функций принадлежности.

        Args:
            x (torch.Tensor): Входные данные, для которых будут созданы функции принадлежности.

        Returns:
            Tuple[List]: Список параметров входных переменных и их соответствующих функций принадлежности.
        """

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

    def __bell_func(self, x: torch.Tensor) -> Tuple[List]:
        """
        Генерирует параметры для колоколообразных функций принадлежности на основе входных данных.

        Вычисляет минимумы и максимумы для каждой входной переменной и создает параметры
        для колоколообразных функций принадлежности.

        Args:
            x (torch.Tensor): Входные данные, для которых будут созданы функции принадлежности.

        Returns:
            Tuple[List]: Кортеж, содержащий список параметров входных переменных и их
                          соответствующие функции принадлежности.
        """

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
        """
        Компилирует модель нечеткой нейронной сети на основе выбранного типа функции принадлежности.

        Вызывает методы для генерации функций принадлежности и создает экземпляр модели
        `_NN`. Переносит модель на указанное устройство (CPU или GPU).

        Args:
            x (torch.Tensor): Входные данные, на основе которых будет скомпилирована модель.

        Returns:
            _NN: Экземпляр нечеткой нейронной сети.
        """

        input_vars, out_vars = self.__gauss_func(x) if self.member_func_type == funcs_type[
            "gauss"] else self.__bell_func(x)
        model = _NN(input_vars, out_vars)
        # Перенос модели на device
        if self.device:
            model.to(self.device)
        return model

    @staticmethod
    def __class_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
        """
        Вычисляет значение функции потерь для задачи классификации.

        Использует кросс-энтропию для определения разницы между предсказанными
        и фактическими метками классов.

        Args:
            inp (torch.Tensor): Предсказанные значения модели.
            target (torch.Tensor): Фактические метки классов.

        Returns:
            float: Значение функции потерь.
        """

        return torch.nn.CrossEntropyLoss()(inp, target.squeeze().long())

    @staticmethod
    def __reg_criterion(inp: torch.Tensor, target: torch.Tensor) -> float:
        """
        Вычисляет значение функции потерь для задачи регрессии.

        Использует среднеквадратичную ошибку (MSE) для определения разницы между
        предсказанными и фактическими значениями.

        Args:
            inp (torch.Tensor): Предсказанные значения модели.
            target (torch.Tensor): Фактические значения.

        Returns:
            float: Значение функции потерь.
        """

        return torch.nn.MSELoss()(inp, target.squeeze())

    @staticmethod
    def __calc_reg_score(preds: torch.Tensor, y_actual: torch.Tensor) -> float:
        """
        Вычисляет оценку модели для задачи регрессии.

        Определяет среднеквадратичную ошибку между предсказанными и фактическими значениями.

        Args:
            preds (torch.Tensor): Предсказанные значения модели.
            y_actual (torch.Tensor): Фактические значения.

        Returns:
            float: Среднеквадратичная ошибка.
        """

        with torch.no_grad():
            tot_loss = F.mse_loss(preds, y_actual)

        return tot_loss

    @staticmethod
    def __calc_class_score(preds: torch.Tensor, y_actual: torch.Tensor, x: torch.Tensor) -> float:
        """
        Вычисляет точность модели для задачи классификации.

        Определяет процент правильных предсказаний среди всех входных данных.

        Args:
            preds (torch.Tensor): Предсказанные значения модели.
            y_actual (torch.Tensor): Фактические метки классов.
            x (torch.Tensor): Входные значения.

        Returns:
            float: Процент правильных предсказаний.
        """

        with torch.no_grad():
            corr = torch.sum(y_actual.squeeze().long() == torch.argmax(preds, dim=1))
            total = len(x)
        return corr * 100 / total

    def __train_loop(self, data: DataLoader, model: _NN,
                     criterion: Callable, calc_score: Callable,
                     optimizer: torch.optim.Adam) -> None:

        """
        Основной цикл обучения модели.

        Обучает модель на данных, обновляет веса, и отслеживает
        результативность модели во время обучения.

        Args:
            data (DataLoader): Загрузчик данных для обучения.
            model (_NN): Модель нечеткой нейронной сети.
            criterion (Callable): Функция потерь, используемая для обучения.
            calc_score (Callable): Функция для оценки модели.
            optimizer (torch.optim.Adam): Оптимизатор для обновления весов модели.

        Returns:
            None
        """

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
        """
        Запускает процесс обучения модели.

        Выполняет предварительную обработку данных, компиляцию модели и
        выполнение цикла обучения с использованием заданных критериев и оптимизатора.

        Returns:
            _NN: Обученная модель нечеткой нейронной сети.
        """

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

    def save_model(self, path: str) -> None:
        """
        Сохраняет состояние обученной модели в файл.
        Сохраняет параметры модели с использованием указанного пути.

        Args:
            path (str): Путь к файлу, в который будет сохранена модель.

        Raises:
            Exception: Если модель не была обучена.

        Returns:
            None
        """

        if self.model:
            torch.save(self.model.state_dict(), path)
        else:
            raise Exception("Модель не обучена")



def process_csv_data(path: str,
                    target_col: str,
                    n_features: int,
                    use_label_encoder: bool,
                    drop_index: bool,
                    split_size: float = 0.2,
                    use_split: bool = False):

    df = pd.read_csv(path)
    Y = df[target_col]
    X = df.drop(target_col, axis=1)

    if drop_index:
        X = X.drop(X.columns[0], axis=1)


    if use_split:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=split_size)

        new_Y_train = Y_train.values
        new_Y_test = Y_test.values

        new_X_train = X_train.values[:, :n_features]
        new_X_test = X_test.values[:, :n_features]


        if use_label_encoder:
            le = LabelEncoder()
            y_train = le.fit_transform(new_Y_train)
            y_test = le.fit_transform(new_Y_test)
        else:
            y_train = new_Y_train
            y_test = new_Y_test

        return new_X_train, new_X_test, y_train, y_test
    else:
        new_Y = Y.values
        new_X = X.values[:, :n_features]

        if use_label_encoder:
            le = LabelEncoder()
            y = le.fit_transform(new_Y)
        else:
            y = new_Y
            
        return new_X, y