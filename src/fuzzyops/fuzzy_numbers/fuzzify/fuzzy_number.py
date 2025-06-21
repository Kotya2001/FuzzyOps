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
    A class for representing a set of possible values of fuzzy numbers

    This class manages the domain of values (it adds a universal set on which fuzzy numbers are built),
    provides methods for creating
    fuzzy numbers, changing the calculation method and visualization

    Attributes:
        _x (torch.Tensor): A set of values in the domain
        step (RealNum): The step between the values in the domain
        name (str): The domain name
        _method (str): The method used for fuzzy operations (for example, 'minimax' or 'prob')
        _vars (dict): Storage of fuzzy numbers in this domain
        bounds (list): The boundaries of the arguments used when creating fuzzy numbers
        membership_type (str): Type of membership function for fuzzy numbers

    Args:
        fset (Union[Tuple[RealNum, RealNum], Tuple[RealNum, RealNum, RealNum], torch.Tensor]):
            The beginning, the end, and the step (or tensor of values) to create the domain
        name (str, optional): The domain name (None by default)
        method (str, optional): Method for fuzzy operations ('minimax' or 'prob', default is 'minimax')

    Properties:
        method (str): Returns or sets the method used for fuzzy operations
        x (torch.Tensor): Returns a range of domain values

    Raises:
        AssertionError: If the passed parameters do not meet the expected requirements
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
        Returns the method used for fuzzy operations

        Returns:
            str: method
        """

        return self._method

    @method.setter
    def method(self, value: str):
        """
        Sets the method for fuzzy operations

        Args:
            value (str): New method ('minimax' or 'prob')

        Raises:
            AssertionError: If the specified method is not 'minimax' or 'prob'
        """

        assert value == 'minimax' or value == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._method = value
        for name, var in self._vars.items():
            var._method = value

    def to(self, device: str):
        """
        Moves the domain to the specified device ('cpu' or 'cuda')

        Args:
            device (str): The device to move (for example, 'cpu' or 'cuda')
        """

        self._x = self._x.to(device)

    def copy(self) -> 'Domain':
        """
        Returns a range of domain values

        Returns:
            Domain: Fuzzy numbers domain
        """
        return Domain(self._fset, self.name, self.method)

    @property
    def x(self) -> torch.Tensor:
        """
        Returns a range of domain values

        Returns:
            torch.Tensor: Range of values
        """

        return self._x

    def create_number(self, membership: Union[str, Callable], *args: RealNum, name: str = None) -> 'FuzzyNumber':
        """
        Creates a new fuzzy number in the domain with the specified membership function

        Args:
            membership (Union[str, Callable]): Name or function for calculating membership
            *args (RealNum): Arguments for the membership function
            name (str, optional): Name for the created fuzzy number (None by default)

        Returns:
            FuzzyNumber: The created fuzzy number

        Raises:
            AssertionError: If membership is not a string or does not match the required number of arguments
        """

        assert isinstance(membership, str) or (isinstance(membership, Callable) and
                                               len(args) == len(signature(membership).parameters))
        if isinstance(membership, str):
            self.membership_type = membership
            membership = memberships[membership]
        f = FuzzyNumber(self, membership(*args), self._method)
        self.bounds = list(args)
        if name:
            self.__setattr__(name, f)
        return f

    def __setattr__(self, name: str, value: 'FuzzyNumber'):
        """
        Sets an attribute for a domain, either a variable or a value

        Args:
            name (str): Attribute Name
            value ('FuzzyNumber'): The attribute value must be FuzzyNumber or str

        Raises:
            AssertionError: If the value is not a FuzzyNumber (for new variables)
        """

        if name in ['_x', 'step', 'name', '_method', '_vars', 'method', 'bounds', 'membership_type']:
            object.__setattr__(self, name, value)
        else:
            assert isinstance(value, FuzzyNumber), 'Value must be FuzzyNumber'
            self._vars[name] = value

    def __getattr__(self, name: str) -> 'FuzzyNumber':
        """
        Gets the attribute value by name

        Args:
            name (str): Attribute name

        Returns:
            FuzzyNumber: The value of the corresponding fuzzy number

        Raises:
            AttributeError: If the attribute with the specified name is not found
        """

        if name in self._vars:
            return self._vars[name]
        else:
            raise AttributeError(f'{name} is not a variable in domain {self.name}')

    def get(self, name: str) -> 'FuzzyNumber':
        """
        Returns a fuzzy number with the specified name

        Args:
            name (str): The name of the fuzzy number

        Returns:
            FuzzyNumber: A fuzzy number with a given name
        """
        return self._vars[name]

    def __delattr__(self, name: str) -> 'FuzzyNumber':
        """
        Deletes a fuzzy number with the specified name from the domain

        Args:
            name (str): Name of the fuzzy number to delete
        """

        if name in self._vars:
            del self._vars[name]

    def plot(self):
        """
        Plots all the fuzzy numbers in the domain

        Uses matplotlib to visualize the values of fuzzy numbers
        """

        _, ax = plt.subplots()

        for name, num in self._vars.items():
            ax.plot(self.x, num.values, label=f'{name}')

        plt.title(self.name)
        ax.legend()
        plt.show()


class FuzzyNumber:
    """
    A fuzzy number defined in a specific domain with a membership function

    Attributes:
        _domain Domain: The domain on which the number is based
        _membership Callable: The membership function of a fuzzy number
        _method str: Calculation method 'minimax' or 'prob'. The default is 'minimax'

    Args:
        domain Domain: The domain on which the number is based
        membership Callable: The membership function of a fuzzy number (for example, a function that returns a tensor)
        method str: Calculation method 'minimax' or 'prob'. The default is 'minimax'

    Properties:
        very:
            A copy of a number with a membership function squared
        negation:
            A copy of a number with the opposite membership function
        maybe:
            A copy of a number with a membership function raised to the power of 0.5
        method:
            The method used for calculations
        membership:
            Fuzzy number membership function
        domain:
            The domain where the fuzzy number is located
        values:
            Fuzzy number values on a given domain
    """

    def __init__(self, domain: Domain, membership: Callable, method: str = 'minimax'):
        assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minmax' and 'prob'"
        self._domain = domain
        self._membership = membership
        self._method = method

    @property
    def very(self) -> 'FuzzyNumber':
        """
        A copy of a number with a membership function squared

        Returns:
            FuzzyNumber: The square of a fuzzy number
        """

        return FuzzyNumber(self._domain, very(self._membership), self._method)

    @property
    def negation(self) -> 'FuzzyNumber':
        """
        A copy of a number with the opposite membership function

        Returns:
            FuzzyNumber: The square of a fuzzy number
        """

        return FuzzyNumber(self._domain, neg(self._membership), self._method)

    @property
    def maybe(self) -> 'FuzzyNumber':
        """
        A copy of a number with a membership function raised to the power of 0.5

        Returns:
            FuzzyNumber: The square of a fuzzy number
        """

        return FuzzyNumber(self._domain, maybe(self._membership), self._method)

    def copy(self) -> 'FuzzyNumber':
        """
        Creates and returns a copy of the fuzzy number

        Returns:
            FuzzyNumber: The square of a fuzzy number
        """

        return FuzzyNumber(self._domain, self._membership, self._method)

    @property
    def method(self) -> str:
        """
        Returns the method used for calculations

        Returns:
            str: Returns the method
        """

        return self._method

    @property
    def membership(self) -> Callable:
        """
        Returns the membership function of a fuzzy number

        Returns:
            Callable: Returns the membership function of a fuzzy number
        """

        return self._membership

    @property
    def domain(self) -> Domain:
        """
        Returns the domain where the fuzzy number is located

        Returns:
            Domain: Returns the domain
        """

        return self._domain

    @property
    def values(self, dtype: str = default_dtype) -> Callable:
        """
        Returns the values of a fuzzy number on the specified domain

        Returns:
            Callable: Returns the confidence level
        """

        return self.membership(self._domain.x)  # .astype(dtype)

    def plot(self, ax=None):
        """
        Plots a graph of a fuzzy number. Creates a new subgraph if not specified

        Args:
            ax (matplotlib.axes._subplots.AxesSubplot, optional):
                An existing graph for adding data. If not specified, a new schedule will be created.
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
        Performs alpha cropping of a fuzzy number

        Args:
            alpha (float): The alpha level for cutting

        Returns:
            torch.Tensor: Domain values for which the membership function is greater than or equal to alpha
        """

        return self.domain.x[self.values >= alpha]

    def entropy(self, norm: bool = True) -> float:
        """
        Calculates the entropy of a fuzzy number

        Args:
            norm (bool): If True, the entropy is normalized by the number of elements in the domain

        Returns:
            float: The value of the entropy of a fuzzy number
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
        Center of gravity defuzzification

        Returns:
            float: The meaning of defazzification
        """
        weights_sum = torch.sum(self.values)
        if weights_sum == 0:
            return 0.0
        return float(torch.sum(self.domain.x * self.values) / weights_sum)

    def left_max(self) -> float:
        """
        Defuzzification by the left maximum method

        Returns:
            float: The meaning of defazzification
        """
        h = torch.max(self.values)
        return float(self.domain.x[self.values == h][0])

    def right_max(self) -> float:
        """
        Defuzzification by the right maximum method

        Returns:
            float: The meaning of defuzzification
        """
        h = torch.max(self.values)
        return float(self.domain.x[self.values == h][1])

    def center_of_max(self, verbose: bool = False) -> float:
        """
        Defuzzification by the central maximum method

        Args:
            verbose (bool): By default, False, If True, displays information about the maxima.

        Returns:
            float: The meaning of defuzzification
        """
        h = torch.max(self.values)
        maxs = self.domain.x[self.values == h]
        if verbose:
            print('h:', h, 'maximums are:', maxs)
        float_tensor = maxs.to(torch.float32)
        return float(torch.mean(float_tensor))

    def moment_of_inertia(self, center: bool = None) -> float:
        """
        Defuzzification by the moment of inertia method

        Args:
            center (float): The center relative to which the moment of inertia is calculated. If not specified, the center of gravity is used

        Returns:
            float: Значение дефаззификации.
        """
        if not center:
            center = self.center_of_grav()
        return float(torch.sum(self.values * torch.square(self.domain.x - center)))

    def defuzz(self, by: str = 'default') -> float:
        """
        Defuzzification of a fuzzy number by a specific method

        Args:
            by (str): Choosing a defazzification method

        Returns:
            float: The meaning of defuzzification
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
        """
        A method for slicing a fuzzy number along the boundary of the degree of confidence of a given clear value
        from a universal set

        Args:
            upper (RealNum): Clear values for the slice

        Returns:
            FuzzyNumber: Limited fuzzy number
        """
        return FuzzyNumber(self.domain, clip_upper(self._membership, upper), self._method)

    # magic

    def __call__(self, x: RealNum) -> torch.Tensor:
        """
        A magical method for obtaining the degree of certainty of a specific clear meaning from a universal set

        Args:
            x (RealNum): Clear values for the slice

        Returns:
            torch.Tensor: The degree of confidence for a value from a universal set
        """
        return self._membership(torch.tensor([x], dtype=self.domain.x.dtype, device=self.domain.x.device))

    def __str__(self) -> str:
        """
        String value of a fuzzy number

        Returns:
            str: A string value of a fuzzy number
        """
        return str(self.defuzz())

    def __repr__(self) -> str:
        """
        String value of a fuzzy number (console output)

        Returns:
            str: The string value of a fuzzy number (console output)
        """
        return 'Fuzzy' + str(self.defuzz())

    def __add__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for summing clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
        if isinstance(other, int) or isinstance(other, float):
            def added(x):
                return self._membership(x - other)

            return FuzzyNumber(self.domain, added, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_unite(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only add a number (Fuzzynumber, int or float)')

    def __iadd__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for summing clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
        return self + other

    def __radd__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for summing clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
        return self.__add__(other)

    def __sub__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for subtracting clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
        if isinstance(other, int) or isinstance(other, float):
            def diff(x):
                return self._membership(x + other)

            return FuzzyNumber(self.domain, diff, self._method)
        elif isinstance(other, FuzzyNumber):
            new_mf = fuzzy_difference(self, other)
            return FuzzyNumber(self.domain, new_mf, self._method)
        else:
            raise TypeError('can only substract a number (Fuzzynumber, int or float)')

    def __isub__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for subtracting clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
        return self - other

    def __mul__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for multiplying clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
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
        """
        An overloaded method for multiplying clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
        return self * other

    def __rmul__(self, other: AnyNum) -> 'FuzzyNumber':
        """
        An overloaded method for multiplying clear and fuzzy numbers with an instance of the fuzzy number class

        Args:
            other (AnyNum): A clear or fuzzy number

        Returns:
            FuzzyNumber: The result of the operation is a fuzzy number
        """
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
        """
        Integer defuzz  value

        Returns:
            int: Integer defuzz  value
        """
        return int(self.defuzz())

    def __float__(self) -> float:
        """
        Float defuzz  value

        Returns:
            float: Float defuzz  value
        """
        return self.defuzz()
