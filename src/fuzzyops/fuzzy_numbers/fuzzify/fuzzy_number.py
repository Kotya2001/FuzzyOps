from .fmath import fuzzy_unite, fuzzy_difference, fuzzy_intersect
from .mf import very, neg, maybe, clip_upper, memberships
from .defuzz import DEFAULT_DEFUZZ
from typing import Callable, Union, Tuple

import torch
import matplotlib.pyplot as plt
from inspect import signature

RealNum = Union[float, int]
AnyNum = Union['FuzzyNumber', int, float]

default_dtype = "float32"


class Domain:
    """
    Класс для представления набора возможных значений чисел в нечеткой логике.

    Этот класс управляет доменом значений, предоставляет методы для создания
    нечетких чисел, изменения метода вычислений и визуализации.

    Attributes:
        _x (torch.Tensor): Набор значений в домене.
        step (RealNum): Шаг между значениями в домене.
        name (str): Имя домена.
        _method (str): Метод, используемый для нечетких операций (например, 'minimax' или 'prob').
        _vars (dict): Хранилище нечетких чисел в данном домене.
        bounds (list): Границы аргументов, используемых при создании нечетких чисел.
        membership_type (str): Тип функции принадлежности для нечетких чисел.

    Args:
        fset (Union[Tuple[RealNum, RealNum], Tuple[RealNum, RealNum, RealNum], torch.Tensor]):
            Начало, конец и шаг (или тензор значений) для создания домена.
        name (str, optional): Имя для домена (по умолчанию None).
        method (str, optional): Метод для нечетких операций ('minimax' или 'prob', по умолчанию 'minimax').

    Properties:
        method (str): Возвращает или устанавливает метод использованный для нечетких операций.
        x (torch.Tensor): Возвращает диапазон значений домена.

    Methods:
        to(device: str) -> None:
            Перемещает домен на указанное устройство (например, на GPU).
        create_number(membership: Union[str, Callable], *args: RealNum, name: str = None) -> 'FuzzyNumber':
            Создает новое нечеткое число в домене с заданной функцией принадлежности.
        get(name: str) -> 'FuzzyNumber':
            Возвращает нечеткое число с заданным именем.
        plot() -> None:
            Строит график всех нечетких чисел в домене.

    Raises:
        AssertionError: Если переданные параметры не соответствуют ожидаемым требованиям.
    """

    def __init__(self, fset: Union[Tuple[RealNum, RealNum], Tuple[RealNum, RealNum, RealNum], torch.Tensor],
                 name: str = None, method: str = 'minimax'):
        assert (isinstance(fset, Tuple) and (len(fset) == 3 or len(fset) == 2)) or isinstance(fset, torch.Tensor), \
            'set bust be given as torch.Tensor or tuple with start, end, step values'
        assert method == 'minimax' or method == 'prob', f"Unknown method. Known methods are 'minimax' and 'prob'"
        if isinstance(fset, torch.Tensor):
            self._x = fset
            self.step = self._x[1] - self._x[0]
        elif len(fset) == 3:
            start, end, step = fset
            self._x = torch.arange(start, end, step)
            self.step = step
        elif len(fset) == 2:
            start, end = fset
            self._x = torch.arange(start, end, 1)
            self.step = 1
        self.name = name
        self._method = method
        self._vars = {}
        self.bounds = []
        self.membership_type = ""

    @property
    def method(self) -> str:
        """
        Возвращает метод, использованный для нечетких операций.

        Returns:
            str: метод.
        """

        return self._method

    @method.setter
    def method(self, value: str):
        """
        Устанавливает метод для нечетких операций.

        Args:
            value (str): Новый метод ('minimax' или 'prob').

        Raises:
            AssertionError: Если указанный метод не является 'minimax' или 'prob'.
        """

        assert value == 'minimax' or value == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._method = value
        for name, var in self._vars.items():
            var._method = value

    def to(self, device: str):
        """
        Перемещает домен на указанное устройство.

        Args:
            device (str): Устройство для перемещения (например, 'cpu' или 'cuda').
        """

        self._x = self._x.to(device)

    def copy(self):
        return Domain(self._fset, self.name, self.method)

    @property
    def x(self) -> torch.Tensor:
        """
        Возвращает диапазон значений домена.

        Returns:
            torch.Tensor: диапазон значений.
        """

        return self._x

    def create_number(self, membership: Union[str, Callable], *args: RealNum, name: str = None) -> 'FuzzyNumber':
        """
        Создает новое нечеткое число в домене с заданной функцией принадлежности.

        Args:
            membership (Union[str, Callable]): Название или функция для вычисления принадлежности.
            *args (RealNum): Аргументы для функции принадлежности.
            name (str, optional): Имя для созданного нечеткого числа (по умолчанию None).

        Returns:
            FuzzyNumber: Созданное нечеткое число.

        Raises:
            AssertionError: Если membership не строка или не соответствует необходимому числу аргументов.
        """

        assert isinstance(membership, str) or (isinstance(membership, Callable) and
                                               len(args) == len(signature(membership).parameters))
        if isinstance(membership, str):
            self.membership_type = membership
            membership = memberships[membership]
        f = FuzzyNumber(self, membership(*args), self._method)
        # закинул аргументы при создании числа в память класса, нужны для оптимизации
        self.bounds = list(args)
        if name:
            self.__setattr__(name, f)
        return f

    def __setattr__(self, name: str, value: Union['FuzzyNumber', str]):
        """
        Устанавливает атрибут для домена - либо переменную, либо значение.

        Args:
            name (str): Имя атрибута.
            value (Union['FuzzyNumber', str]): Значение атрибута, должно быть FuzzyNumber.

        Raises:
            AssertionError: Если значение не является FuzzyNumber (для новых переменных).
        """

        if name in ['_x', 'step', 'name', '_method', '_vars', 'method', 'bounds', 'membership_type']:
            object.__setattr__(self, name, value)
        else:
            # assert isinstance(name, str) and name not in self._vars, 'Name must be a unique string'
            assert isinstance(value, FuzzyNumber), 'Value must be FuzzyNumber'
            self._vars[name] = value

    def __getattr__(self, name: str) -> 'FuzzyNumber':
        """
        Получает значение атрибута по имени.

        Args:
            name (str): Имя атрибута.

        Returns:
            FuzzyNumber: Значение соответствующего нечеткого числа.

        Raises:
            AttributeError: Если атрибут с заданным именем не найден.
        """

        if name in self._vars:
            return self._vars[name]
        else:
            raise AttributeError(f'{name} is not a variable in domain {self.name}')

    def get(self, name: str) -> 'FuzzyNumber':
        """
        Возвращает нечеткое число с заданным именем.

        Args:
            name (str): Имя нечеткого числа.

        Returns:
            FuzzyNumber: Нечеткое число с заданным именем.
        """
        return self._vars[name]

    def __delattr__(self, name: str) -> 'FuzzyNumber':
        """
        Удаляет нечеткое число с заданным именем из домена.

        Args:
            name (str): Имя нечеткого числа для удаления.
        """

        if name in self._vars:
            del self._vars[name]

    def plot(self):
        """
        Строит график всех нечетких чисел в домене.

        Использует matplotlib для визуализации значений нечетких чисел.
        """

        _, ax = plt.subplots()

        for name, num in self._vars.items():
            ax.plot(self.x, num.values, label=f'{name}')

        plt.title(self.name)
        ax.legend()
        plt.show()


class FuzzyNumber:
    """
    Нечеткое число, заданное в определенном домене, с функцией принадлежности, представленной торчевым тензором.

    Attributes:
        _domain Domain: Домен, на котором основано число.
        _membership Callable: Функция принадлежности нечеткого числа (например, функция, возвращающая тензор).
        _method str: Метод вычислений: 'minimax' или 'prob'. По умолчанию 'minimax'.

    Args:
        domain Domain: Домен, на котором основано число.
        membership Callable: Функция принадлежности нечеткого числа (например, функция, возвращающая тензор).
        method str:, по умолчанию 'minimax', метод для вычислений ('minimax' или 'prob').

    Properties:
        very:
            Копия числа с функцией принадлежности, возведенной в квадрат.
        negation:
            Копия числа с противоположной функцией принадлежности.
        maybe:
            Копия числа с функцией принадлежности, возведенной в степень 0.5.
        method:
            Метод, используемый для вычислений.
        membership:
            Функция принадлежности нечеткого числа.
        domain:
            Домен, в котором находится нечеткое число.
        values:
            Значения нечеткого числа на заданном домене.

    Methods:
        copy() -> FuzzyNumber:
            Создает и возвращает копию нечеткого числа.
        plot(ax=None) -> List:
            Строит график нечеткого числа.
        alpha_cut(alpha: float) -> torch.Tensor:
            Выполняет альфа-обрезку нечеткого числа.
        entropy(norm: bool = True) -> float:
            Вычисляет энтропию нечеткого числа.
        center_of_grav() -> float:
            Вычисляет центр тяжести нечеткого числа.
        left_max() -> float:
            Возвращает левый максимум нечеткого числа.
        right_max() -> float:
            Возвращает правый максимум нечеткого числа.
        center_of_max(verbose: bool = False) -> float:
            Вычисляет центр максимума нечеткого числа.
        moment_of_inertia(center: bool = None) -> float:
            Вычисляет момент инерции нечеткого числа относительно заданного центра.
    """

    def __init__(self, domain: Domain, membership: Callable, method: str = 'minimax'):
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._domain = domain
        self._membership = membership
        # self._values = membership(self._domain.x).astype(default_dtype)
        self._method = method

    @property
    def very(self) -> 'FuzzyNumber':
        """
        Копия числа с функцией принадлежности, возведенной в квадрат.

        Returns:
            FuzzyNumber: Квадрат нечеткого числа.
        """

        return FuzzyNumber(self._domain, very(self._membership), self._method)

    @property
    def negation(self) -> 'FuzzyNumber':
        """
        Копия числа с противоположной функцией принадлежности.

        Returns:
            FuzzyNumber: Квадрат нечеткого числа.
        """

        return FuzzyNumber(self._domain, neg(self._membership), self._method)

    @property
    def maybe(self) -> 'FuzzyNumber':
        """
        Копия числа с функцией принадлежности, возведенной в степень 0.5.

        Returns:
            FuzzyNumber: Квадрат нечеткого числа.
        """

        return FuzzyNumber(self._domain, maybe(self._membership), self._method)

    def copy(self) -> 'FuzzyNumber':
        """
        Создает и возвращает копию нечеткого числа.

        Returns:
            FuzzyNumber: Квадрат нечеткого числа.
        """

        return FuzzyNumber(self._domain, self._membership, self._method)

    @property
    def method(self) -> str:
        """
        Возвращает метод, используемый для вычислений.

        Returns:
            str: Возвращает метод
        """

        return self._method

    @property
    def membership(self) -> Callable:
        """
        Возвращает функцию принадлежности нечеткого числа.

        Returns:
            Callable: Возвращает функцию принадлежности нечеткого числа.
        """

        return self._membership

    @property
    def domain(self) -> Domain:
        """
        Возвращает домен, в котором находится нечеткое число.

        Returns:
            Domain: Возвращает домен.
        """

        return self._domain

    @property
    def values(self, dtype: str = default_dtype) -> Callable:
        """
        Возвращает значения нечеткого числа на заданном домене.

        Returns:
            Callable: Возвращает степени уверенности .
        """

        return self.membership(self._domain.x)  # .astype(dtype)

    def plot(self, ax=None):
        """
        Строит график нечеткого числа. Создает новый подграфик, если не указан.

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot, optional):
                Существующий график для добавления данных. Если не указан, будет создан новый график.
        """

        if self.domain.x.device.type != 'cpu':
            raise TypeError(f"can't convert {self.domain.x.device} device type tensor to numpy. Use Domain.to('cpu') "
                            f"first.")
        if not ax:
            _, ax = plt.subplots()
        out = ax.plot(self.domain.x, self.values)
        plt.show()
        return out

    def alpha_cut(self, alpha: float) -> torch.Tensor:
        """
        Выполняет альфа-обрезку нечеткого числа.

        Args:
            alpha (float): Уровень альфа для обрезки.

        Returns:
            torch.Tensor: Значения домена, для которых функция принадлежности больше или равна alpha.
        """

        return self.domain.x[self.values >= alpha]

    def entropy(self, norm: bool = True) -> float:
        """
        Вычисляет энтропию нечеткого числа.

        Args:
            norm (bool): Если True, энтропия нормируется по количеству элементов в домене.

        Returns:
            float: Значение энтропии нечеткого числа.
        """

        vals = self.values
        mask = vals != 0
        e = -torch.sum(vals[mask] * torch.log2(vals[mask]))
        if norm:
            return 2. / len(self.values) * e
        else:
            return e

    def center_of_grav(self) -> float:
        """
        Дефаззификация методом центра тяжести.

        Returns:
            float: Значение дефаззификации.
        """

        return float(torch.sum(self.domain.x * self.values) / torch.sum(self.values))

    def left_max(self) -> float:
        """
        Дефаззификация методом левого максимума.

        Returns:
            float: Значение дефаззификации.
        """
        h = torch.max(self.values)
        return float(self.domain.x[self.values == h][0])

    def right_max(self) -> float:
        """
        Дефаззификация методом правого максимума.

        Returns:
            float: Значение дефаззификации.
        """
        h = torch.max(self.values)
        return float(self.domain.x[self.values == h][1])

    def center_of_max(self, verbose: bool = False) -> float:
        """
        Дефаззификация методом центрального максимума.

        Args:
            verbose (bool):, по умолчанию False Если True, выводит информацию о максимумах.

        Returns:
            float: Значение дефаззификации.
        """
        h = torch.max(self.values)
        maxs = self.domain.x[self.values == h]
        if verbose:
            print('h:', h, 'maximums are:', maxs)
        float_tensor = maxs.to(torch.float32)
        return float(torch.mean(float_tensor))

    def moment_of_inertia(self, center: bool = None) -> float:
        """
        Дефаззификация методом момента инерции.

        Args:
            center (float):, optional Центр, относительно которого вычисляется момент инерции. Если не указан, используется центр тяжести.

        Returns:
            float: Значение дефаззификации.
        """
        if not center:
            center = self.center_of_grav()
        return float(torch.sum(self.values * torch.square(self.domain.x - center)))

    def defuzz(self, by: str = 'default') -> float:
        """
        Дефаззификация нечеткого числа конкретным методом.

        Args:
            by (str): выбор метода дефаззицикации

        Returns:
            float: Значение дефаззификации.
        """
        if by == 'default':
            by = DEFAULT_DEFUZZ
        if by == 'lmax':
            return self.left_max()
        elif by == 'rmax':
            return self.right_max()
        elif by == 'cmax':
            return self.center_of_max()
        elif by == 'cgrav':
            return self.center_of_grav()
        else:
            raise ValueError('defuzzification can be made by lmax, rmax, cmax, cgrav or default')

    def clip_upper(self, upper: RealNum) -> 'FuzzyNumber':
        """Clips the number from above.
        Parameters
        ----------
        upper : `float`
        Returns
        -------
        number : `FuzzyNumber`
        """
        return FuzzyNumber(self.domain, clip_upper(self._membership, upper), self._method)

    # magic

    def __call__(self, x: RealNum) -> Callable:
        return self._membership(torch.tensor([x], dtype=self.domain.x.dtype, device=self.domain.x.device))

    def __str__(self) -> str:
        return str(self.defuzz())

    def __repr__(self) -> str:
        return 'Fuzzy' + str(self.defuzz())

    def __add__(self, other: AnyNum) -> 'FuzzyNumber':
        if isinstance(other, int) or isinstance(other, float):
            def added(x):
                return self._membership(x + other)

            return FuzzyNumber(self.domain, added, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_unite(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only add a number (Fuzzynumber, int or float)')

    def __iadd__(self, other: AnyNum) -> 'FuzzyNumber':
        return self + other

    def __radd__(self, other: AnyNum) -> 'FuzzyNumber':
        return self.__add__(other)

    def __sub__(self, other: AnyNum) -> 'FuzzyNumber':
        if isinstance(other, int) or isinstance(other, float):
            def diff(x):
                return self._membership(x - other)

            return FuzzyNumber(self.domain, diff, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_difference(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __isub__(self, other: AnyNum) -> 'FuzzyNumber':
        return self - other

    def __mul__(self, other: AnyNum) -> 'FuzzyNumber':
        if isinstance(other, int) or isinstance(other, float):
            # raise NotImplementedError('Multiplication by a number is not implemented yet')

            def multiplied(x):
                return self._membership(x * other)

            return FuzzyNumber(self.domain, multiplied, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_intersect(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __imul__(self, other: AnyNum) -> 'FuzzyNumber':
        return self * other

    def __rmul__(self, other: AnyNum) -> 'FuzzyNumber':
        return self.__mul__(other)

    def __truediv__(self, other: RealNum) -> 'FuzzyNumber':
        # raise NotImplementedError('Division is not implemented yet')
        t_o = type(other)
        if t_o == int or t_o == float:
            def divided(x):
                return self._membership(x / other)

            return FuzzyNumber(self.domain, divided, self._method)
        else:
            raise TypeError('can only divide by a number (int or float)')

    def __idiv__(self, other: RealNum):
        return self / other

    def __int__(self) -> int:
        return int(self.defuzz())

    def __float__(self) -> float:
        return self.defuzz()
