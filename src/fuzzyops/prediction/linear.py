import torch
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber, memberships
from typing import List, Tuple


# вычисление расстояния между нечеткими числвми
def fuzzy_distance(fn0: 'TriFNum', fn1: 'TriFNum') -> float:
    """
    Вычисляет расстояние между двумя треугольными нечеткими числами.

    Args:
        fn0 (TriFNum): Первое треугольное нечеткое число.
        fn1 (TriFNum): Второе треугольное нечеткое число.

    Returns:
        float: Расстояние между двумя нечеткими числами.
    """

    a0, b0 = fn0.a, fn0.b
    a1, b1 = fn1.a, fn1.b
    left_integral = ((b1 - b0 - a1 + a0) ** 2) / 3 - 2 * a0 * (((b1 - b0 - a1 + a0)) / 2 + a1) + a1 * (
            b1 - b0 - a1 + a0) + a1 ** 2 + a0 ** 2
    a0, b0 = fn0.b, fn0.c
    a1, b1 = fn1.b, fn1.c
    right_integral = ((b1 - b0 - a1 + a0) ** 2) / 3 - 2 * a0 * (((b1 - b0 - a1 + a0)) / 2 + a1) + a1 * (
            b1 - b0 - a1 + a0) + a1 ** 2 + a0 ** 2
    dist = (left_integral + right_integral) ** 0.5
    if isinstance(dist, float):
        return dist
    return dist.item()


def integral_of_product(a_0: torch.Tensor, b_0: torch.Tensor,
                        a_1: torch.Tensor, b_1: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет интеграл произведения двух интервалов.

    Args:
        a_0 (torch.Tensor): Начало первого интервала.
        b_0 (torch.Tensor): Конец первого интервала.
        a_1 (torch.Tensor): Начало второго интервала.
        b_1 (torch.Tensor): Конец второго интервала.

    Returns:
        torch.Tensor: Результат интеграла произведения.
    """

    return (1 * (((2 * b_0 - 2 * a_0) * b_1 - 2 * a_1 * b_0 + 2 * a_0 * a_1) * 1 ** 2 + (
            3 * a_0 * b_1 + 3 * a_1 * b_0 - 6 * a_0 * a_1) * 1 + 6 * a_0 * a_1)) / 6


# вычисление интеграла суммы квадратов
def integrate_sum_squares(a_0: torch.Tensor, b_0: torch.Tensor,
                          a_1: torch.Tensor, b_1: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет интеграл суммы квадратов двух интервалов.

    Args:
        a_0 (torch.Tensor): Начало первого интервала.
        b_0 (torch.Tensor): Конец первого интервала.
        a_1 (torch.Tensor): Начало второго интервала.
        b_1 (torch.Tensor): Конец второго интервала.

    Returns:
        torch.Tensor: Результат интеграла суммы квадратов.
    """

    return ((b_1 - a_1) ** 2) / 3 + ((b_0 - a_0) ** 2) / 3 + a_1 * (b_1 - a_1) + a_0 * (b_0 - a_0) + a_1 ** 2 + a_0 ** 2


def convert_fuzzy_number_for_lreg(n: FuzzyNumber) -> 'TriFNum':
    """
    Преобразует нечеткое число класса FuzzyNumber в треугольное нечеткое число TriNum.

    Args:
        n (FuzzyNumber): Нечеткое число для преобразования.

    Returns:
        TriFNum: Преобразованное треугольное нечеткое число.
    """

    vals = n.values
    first_increasing_idx = (vals[1:] > vals[:-1]).nonzero(as_tuple=True)[0][0] + 1
    last_zero_before_increase = (vals[:first_increasing_idx] == 0).nonzero(as_tuple=True)[0][-1].item()
    max_idx = vals.argmax().item()
    first_zero_after_peak = (vals[max_idx + 1:] == 0).nonzero(as_tuple=True)[0][0] + max_idx + 1
    a, b, c = n.domain.x[last_zero_before_increase], n.domain.x[max_idx], n.domain.x[first_zero_after_peak]
    return TriFNum(n.domain, a, b, c)


class TriFNum:
    """
    Представляет треугольное нечеткое число (TriFNum) для метода нечеткой линейной регрессии.

    Attributes:
        domain (Domain): Область определения нечеткого числа.
        a (torch.Tensor): Левый конец треугольника.
        b (torch.Tensor): Пик треугольника.
        c (torch.Tensor): Правый конец треугольника.

    Methods:
        __init__(domain: Domain, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            Инициализирует треугольное нечеткое число.
        values() -> torch.Tensor:
            Вычисляет и возвращает значения нечеткого числа на заданной области определения.
        __add__(other: TriFNum | int | float) -> TriFNum:
            Определяет операцию сложения для треугольных нечетких чисел.
        __sub__(other: TriFNum) -> TriFNum:
            Определяет операцию вычитания для треугольных нечетких чисел.
        __mul__(other: int | float) -> TriFNum:
            Определяет операцию умножения для треугольных нечетких чисел.
        integrate_left() -> torch.Tensor:
            Вычисляет интеграл для левой стороны треугольного нечеткого числа.
        integrate_right() -> torch.Tensor:
            Вычисляет интеграл для правой стороны треугольного нечеткого числа.
        integrate() -> torch.Tensor:
            Вычисляет интеграл (полную площадь) под кривой треугольного нечеткого числа.
        to_fuzzy_number() -> FuzzyNumber:
            Преобразует треугольное нечеткое число в его нечеткое представление.
    """

    def __init__(self, domain: Domain, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        def left_mf(x):
            k = 1 / (b - a)
            m = -a / (b - a)
            return k * x + m

        def right_mf(x):
            k = 1 / (b - c)
            m = -c / (b - c)
            return k * x + m

        self.left_mf = left_mf
        self.right_mf = right_mf
        self.domain = domain
        self.a = a
        self.b = b
        self.c = c

    def values(self) -> torch.Tensor:
        """
        Вычисляет значения нечеткого числа на заданной области определения.

        Returns:
            torch.Tensor: Значения нечеткого числа в области определения.
        """

        left = self.left_mf(self.domain.x).clip(0, 1)
        right = self.right_mf(self.domain.x).clip(0, 1)
        values = torch.zeros_like(self.domain.x)
        p0 = torch.where(left < 1)[0]
        p1 = torch.where((left >= 1) & (right >= 1))[0]
        p2 = torch.where(right < 1)[0]
        values[p0] = left[p0]
        values[p1] = 1
        values[p2] = right[p2]
        return values

    def __add__(self, other):
        """
        Определяет операцию сложения для треугольных нечетких чисел.

        Args:
            other (TriFNum | int | float): Другой операнд для сложения.

        Returns:
             TriFNum: Результат сложения.

        Raises:
            NotImplementedError: Если другие типы операндов.
        """

        if isinstance(other, TriFNum):
            return TriFNum(self.domain, self.a + other.a, self.b + other.b, self.c + other.c)
        elif isinstance(other, int) or isinstance(other, float):
            return TriFNum(self.domain, self.a + other, self.b + other, self.c + other)
        else:
            raise NotImplementedError('can only add TriFNums or real numbers')

    def __sub__(self, other):
        """
        Определяет операцию вычитания для треугольных нечетких чисел.

        Args:
            other (TriFNum): Другой операнд для вычитания.

        Returns:
             TriFNum: Результат вычитания.

        Raises:
            NotImplementedError: Если тип операнда отличается.
        """

        if isinstance(other, TriFNum):
            return TriFNum(self.domain, self.a - other.a, self.b - other.b, self.c - other.c)
        else:
            raise NotImplementedError('can only substract TriFNums')

    def __mul__(self, other):
        """
        Определяет операцию умножения для треугольных нечетких чисел.

        Args:
            other (int | float): Другой операнд для умножения.

        Returns:
             TriFNum: Результат умножения.

        Raises:
            NotImplementedError: Если тип операнда отличается.
        """
        if isinstance(other, int) or isinstance(other, float):
            if other > 0:
                return TriFNum(self.domain, self.a * other, self.b * other, self.c * other)
            else:
                return TriFNum(self.domain, self.c * other, self.b * other, self.a * other)
        else:
            raise NotImplementedError('can only multiply by real number')

    def integrate_left(self) -> torch.Tensor:
        """
        Вычисляет интеграл для левой стороны треугольного нечеткого числа.

        Returns:
            torch.Tensor: Значение интеграла.
        """

        return (self.b - self.a) / 2 + self.a

    def integrate_right(self) -> torch.Tensor:
        """
        Вычисляет интеграл для правой стороны треугольного нечеткого числа.

        Returns:
            torch.Tensor: Значение интеграла.
        """

        return (self.b - self.c) / 2 + self.b

    def integrate(self) -> torch.Tensor:
        """
        Вычисляет интеграл (полную площадь) под кривой треугольного нечеткого числа.

        Returns:
            torch.Tensor: Значение интеграла.
        """

        return self.integrate_right() - self.integrate_left()

    def to_fuzzy_number(self) -> FuzzyNumber:
        """
        Преобразует треугольное нечеткое число в его нечеткое представление.

        Returns:
            FuzzyNumber: Нечеткое число, созданное из треугольного нечеткого числа.
        """

        return self.domain.create_number("triangular", self.a, self.b, self.c)


def fit_fuzzy_linear_regression(X: List[TriFNum], Y: List[TriFNum]) -> Tuple[float, float]:
    """
    Реализует нечеткую линейную регрессию с использованием треугольных нечетких чисел.

    Эта функция находит коэффициенты a и b для линейной регрессии,
    которые минимизируют расстояние между предсказанными нечеткими значениями
    и фактическими нечеткими значениями.
    Реализовано на основе материалов
    https://ej.hse.ru/data/2014/09/03/1316474700/%D0%A8%D0%B2%D0%B5%D0%B4%D0%BE%D0%B2.pdf

    Args:
        X (List[TriFNum]): Список треугольных нечетких чисел, представляющих
                           независимые переменные (фичи).
        Y (List[TriFNum]: Список треугольных нечетких чисел, представляющих
                           зависимую переменную (целевую переменную).

    Returns:
        Tuple[float, float]: Коэффициенты a и b линейной регрессии,
                             где a - угловой коэффициент, а b - свободный член.

    Raises:
        ValueError: Если длины списков X и Y не совпадают.

    Notes:
        Для выполнения расчетов используется интегрирование и вычисление
        различных моментов на основе нечетких чисел.
    """

    n = len(X)
    if isinstance(X[0], FuzzyNumber):
        # X = deepcopy(X)
        for i in range(n):
            X[i] = convert_fuzzy_number_for_lreg(X[i])
    if isinstance(Y[0], FuzzyNumber):
        # Y = deepcopy(Y)
        for i in range(n):
            Y[i] = convert_fuzzy_number_for_lreg(Y[i])

    I1 = sum([integral_of_product(Y[i].a, Y[i].b, X[i].a, X[i].b) for i in range(n)])
    I2 = sum([integral_of_product(Y[i].b, Y[i].c, X[i].b, X[i].c) for i in range(n)])
    J1 = sum([integral_of_product(Y[i].a, Y[i].b, X[i].b, X[i].c) for i in range(n)])
    J2 = sum([integral_of_product(Y[i].b, Y[i].c, X[i].a, X[i].b) for i in range(n)])
    K1 = sum([integral_of_product(X[i].a, X[i].b, X[i].a, X[i].b) for i in range(n)])
    K2 = sum([integral_of_product(X[i].b, X[i].c, X[i].b, X[i].c) for i in range(n)])
    L1 = sum([X[i].integrate_left() for i in range(n)])
    L2 = sum([X[i].integrate_right() for i in range(n)])
    M1 = sum([Y[i].integrate_left() for i in range(n)])
    M2 = sum([Y[i].integrate_right() for i in range(n)])
    N = sum([integrate_sum_squares(Y[i].a, Y[i].b, Y[i].b, Y[i].c) for i in range(n)])
    a_pos = max(0, (2 * n * (I1 + I2) - (L1 + L2) * (M1 + M2)) / (2 * n * (K1 + K2) - (L1 + L2) ** 2))
    a_neg = min(0, (2 * n * (J1 + J2) - (L1 + L2) * (M1 + M2)) / (2 * n * (K1 + K2) - (L1 + L2) ** 2))
    b_pos = 1 / 2 / n * (M1 + M2) - 1 / 2 / n * a_pos * (L1 + L2)
    b_neg = 1 / 2 / n * (M1 + M2) - 1 / 2 / n * a_neg * (L1 + L2)

    # Вычисление функционала H в зависимости от коэффициента а
    H_pos = a_pos ** 2 * (K1 + K2) + 2 * a_pos * b_pos * (L1 + L2) + 2 * n * b_pos ** 2 - 2 * a_pos * (
            I1 + I2) - 2 * b_pos * (M1 + M2) + N
    H_neg = a_neg ** 2 * (K1 + K2) + 2 * a_neg * b_neg * (L1 + L2) + 2 * n * b_neg ** 2 - 2 * a_neg * (
            J1 + J2) - 2 * b_neg * (M1 + M2) + N

    if H_pos < H_neg:
        a, b = a_pos, b_pos
    else:
        a, b = a_neg, b_neg
    a, b = a.item(), b.item()
    errors = []
    for i in range(len(X)):
        errors.append(fuzzy_distance(X[i] * a + b, Y[i]))
    return a, b, (sum(errors) / n) ** 0.5
