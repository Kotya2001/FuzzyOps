"""
file with "triangle" number class for use in graph and
with functions

Triangle number have structure: [a, b, c], where
triangle number is: (a-b; a; a+c)

a - center of fuzzy triangle number
b - left delta
c - right delta

------------------------------------------------------
can be created from:

- list with len=3:
    [1, 2, 3]

------------------------------------------------------
compare functions:

- __eq__:

checks if all three values of number are equal:
    a1 == a2 and b1 == b2 and c1 == c2

- __gt__:

depends on 'eq_types' variable, that is passed on object creating:

    - 'base' (default value):
        checks if base of first triangle number is greater
        than base of second triangle number:
           a1 > a2

    - 'min':
        checks if lower border of first triangle number is
        greater than base of second triangle number:
            (a1 - b1) > (a2 - b2)

    - 'max':
        checks if higher border of first triangle number is
        greater than base of second triangle number:
            (a1 + c1) > (a2 + c2)

    if equality is not met - return False

------------------------------------------------------
math functions:

bases of triangle fuzzy numbers are summed, divided, e.t.c.,
depending on given math operation.

difference is in deltas calculations. approach to
delta calculation should be described in 'math_type'
variable:

    - 'mean' (default):
        new delta is mean value of same deltas from given numbers
        F1 - F2 = (a1, b1, c1) - (a2, b2, c2) = (a1-a2, mean(b1, b2), mean(c1, c2))

    - 'min'
        new delta is minimum value of same deltas from given numbers
        F1 - F2 = (a1, b1, c1) - (a2, b2, c2) = (a1-a2, min(b1, b2), min(c1, c2))

    - 'max'
        new delta is maximum value of same deltas from given numbers
        F1 - F2 = (a1, b1, c1) - (a2, b2, c2) = (a1-a2, max(b1, b2), max(c1, c2))

    - 'sum'
        new delta is sum of same deltas from given numbers
        F1 - F2 = (a1, b1, c1) - (a2, b2, c2) = (a1-a2, b1+b2, c1+c2)

type of approach is taken from left value (F1 in examples).

"""

# todo(exceptions!!!)

from .base_number import BaseGraphNumber

_equal_allowed_types = [
    'base',
    'min',
    'max',
]
_math_allowed_types = [
    'mean',
    'min',
    'max',
    'sum',
]

class GraphTriangleFuzzyNumber(BaseGraphNumber):
    def _set_value(
        self,
        value,
        eq_type=None,
        math_type=None,
    ):
        is_good = True
        if type(value) is list:
            if len(value) != 3:
                is_good = False
                raise Exception(f'array for triangle number should have len 3; given: {len(value)}')
            if not((type(value[0]) in [int, float]) and (type(value[1]) in [int, float]) and (type(value[2]) in [int, float])):
                is_good = False
                raise Exception(f'values of triangle number array should be int or float')
        else:
            is_good = False
            raise Exception(f'cannot create "triangle numbers" object from {type(value)} object')

        if eq_type is None:
            eq_type = 'base'
        elif not(eq_type in _equal_allowed_types):
            is_good = False
            raise Exception(f'equal type {eq_type} is not allowed')

        if math_type is None:
            math_type = 'mean'
        elif not(math_type in _math_allowed_types):
            is_good = False
            raise Exception(f'math type {math_type} is not allowed')


        if is_good:
            self._value = value
            self._funcs = _GraphTriangleFuzzyNumberFunctions
            self._eq_type = eq_type
            self._math_type = math_type
        else:
            raise Exception(f'creating of object failed')

    # todo
    def __repr__(self):
        return str(self._value)

    def __str__(self):
        return str(self._value)



class _GraphTriangleFuzzyNumberFunctions:
    def is_equal(
        self,
        other
    ):
        # todo(add check or not???)
        return (self._value[0] == other._value[0]) and (self._value[1] == other._value[1]) and (self._value[2] == other._value[2])


    def is_greater(
        self,
        other
    ):
        if self._eq_type == 'base':
            return self._value[0] > other._value[0]

        if self._eq_type == 'min':
            return (self._value[0] - self._value[1]) > (other._value[0] - other._value[1])

        if self._eq_type == 'max':
            return (self._value[0] + self._value[2]) > (other._value[0] + other._value[2])

        raise Exception(f'wrong _eq_type: {self._eq_type}')


    def add(
        self,
        other
    ):
        if self._math_type == 'mean':
            arr = [
                self._value[0] + other._value[0],
                (self._value[1] + other._value[1]) / 2,
                (self._value[2] + other._value[2]) / 2
            ]
        if self._math_type == 'min':
            arr = [
                self._value[0] + other._value[0],
                min(self._value[1], other._value[1]),
                min(self._value[2], other._value[2])
            ]
        if self._math_type == 'max':
            arr = [
                self._value[0] + other._value[0],
                max(self._value[1], other._value[1]),
                max(self._value[2], other._value[2])
            ]
        if self._math_type == 'sum':
            arr = [
                self._value[0] + other._value[0],
                self._value[1] + other._value[1],
                self._value[2] + other._value[2]
            ]
        to_ret = GraphTriangleFuzzyNumber(arr, self._eq_type, self._math_type)
        return to_ret


    def sub(
        self,
        other
    ):
        if self._math_type == 'mean':
            arr = [
                self._value[0] - other._value[0],
                (self._value[1] + other._value[1]) / 2,
                (self._value[2] + other._value[2]) / 2
            ]
        if self._math_type == 'min':
            arr = [
                self._value[0] - other._value[0],
                min(self._value[1], other._value[1]),
                min(self._value[2], other._value[2])
            ]
        if self._math_type == 'max':
            arr = [
                self._value[0] - other._value[0],
                max(self._value[1], other._value[1]),
                max(self._value[2], other._value[2])
            ]
        if self._math_type == 'sum':
            arr = [
                self._value[0] - other._value[0],
                self._value[1] + other._value[1],
                self._value[2] + other._value[2]
            ]
        to_ret = GraphTriangleFuzzyNumber(arr, self._eq_type, self._math_type)
        return to_ret
