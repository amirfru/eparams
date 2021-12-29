"""Constraint class and some common constraints."""
import os
from typing import Any, Callable, NamedTuple


class Constraint(NamedTuple):
    """Constraint function wrapper."""

    func: Callable[[Any], bool]
    description: str

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __invert__(self):
        return Constraint(lambda x: not self.func(x), 'NOT ' + self.description)

    def __and__(self, other):
        if isinstance(other, Constraint):
            return Constraint(lambda x: self.func(x) and other.func(x), f'{self.description} AND {other.description}')
        else:
            return Constraint(lambda x: self.func(x) and other(x), f'{self.description} AND {other}')

    def __or__(self, other):
        if isinstance(other, Constraint):
            return Constraint(lambda x: self.func(x) or other.func(x), f'{self.description} OR {other.description}')
        else:
            return Constraint(lambda x: self.func(x) or other(x), f'{self.description} OR {other}')


def in_range(xmin: float, xmax: float) -> Constraint:
    return Constraint(lambda x: xmin <= x <= xmax, f'value must be between {xmin}..{xmax}')


def max_len(maxlen: int) -> Constraint:
    return Constraint(lambda x: len(x) < maxlen, f'len() must be <= {maxlen}')


def len_is(lenx : int) -> Constraint:
    return Constraint(lambda x: len(x) == lenx, f'len() must be == {lenx}')


def min_len(minlen: int) -> Constraint:
    return Constraint(lambda x: len(x) >= minlen, f'len() must be >= {minlen}')


def max_val(xmax: float) -> Constraint:
    return Constraint(lambda x: x <= xmax, f'value must be <= {xmax}')


def min_val(xmin: float) -> Constraint:
    return Constraint(lambda x: x >= xmin, f'value must be >= {xmin}')


def choice(*opts: Any) -> Constraint:
    return Constraint(lambda x: x in opts, f'value must be one of: {opts}')


isfile = Constraint(os.path.isfile, f'path is not a file')
isdir = Constraint(os.path.isdir, f'path is not a dir')
exists = Constraint(os.path.exists, f'path does not exist')
