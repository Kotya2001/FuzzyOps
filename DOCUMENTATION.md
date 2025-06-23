### Classes

```python
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
```

### Functions

```python
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
```