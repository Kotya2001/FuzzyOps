"""
master-class for any class of number that could be used in graph
"""


# todo(better exceptions)

class BaseGraphNumber:
    def __init__(
        self,
        value,
        eq_type=None,
        math_type=None,
    ):
        self._value = None
        self._funcs = None
        self._eq_type = None
        self._math_type = None
        self._set_value(value, eq_type, math_type)

    def _set_value(
        self,
        value,
        eq_type=None,
        math_type=None,
    ):
        raise Exception('no _set_value_overwritten!!!')

    def __eq__(self, other):
        if not(type(self) == type(other)):
            raise Exception(f'cannot compare "{type(self).__name__}" with "{type(other).__name__}"')

        is_equal = getattr(self._funcs, 'is_equal', None)

        if is_equal:
            return is_equal(self, other)
        else:
            raise Exception('this class has not "is_equal" function written')


    def __gt__(self, other):
        if not(type(self) == type(other)):
            raise Exception(f'cannot compare "{type(self).__name__}" with "{type(other).__name__}"')

        is_greater = getattr(self._funcs, 'is_greater', None)

        if is_greater:
            return is_greater(self, other)
        else:
            raise Exception('this class has not "is_greater" function written')


    def __add__(self, other):
        if other is None:
            return type(self)(self._value, self._eq_type, self._math_type)

        if not(type(self) == type(other)):
            raise Exception(f'cannot add "{type(other).__name__}" to "{type(self).__name__}"')

        add = getattr(self._funcs, 'add', None)

        if add:
            return add(self, other)
        else:
            raise Exception('this class has not "add" function written')


    def __sub__(self, other):
        if other is None:
            return type(self)(self._value, self._eq_type, self._math_type)

        if not(type(self) == type(other)):
            raise Exception(f'cannot subtract "{type(other).__name__}" from "{type(self).__name__}"')

        sub = getattr(self._funcs, 'sub', None)

        if sub:
            return sub(self, other)
        else:
            raise Exception('this class has not "sub" function written')