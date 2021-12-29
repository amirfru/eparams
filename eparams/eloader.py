"""Register and map delta-configs."""
import runpy
import sys
from pathlib import Path
from typing import TypeVar, Any, Dict, Callable

__all__ = ['register', 'scan_for_registered', 'mapping']

mapping = {}
EXECUTED = set()  # saves a list of executed files so we don't run the same file twice.
F = TypeVar('F')


def _same_func(f1, f2) -> bool:
    if f1 is f2:
        return True
    return f1.__code__ == f2.__code__


def register(func: F = None, *, key: Any = None) -> F:
    """Return the same function after registering in the global `mapping`.

    Args:
        func: the function to be mapped
        key: the key in the global mapping to use. If empty, use function's name

    Returns:
        func (The original function).
    """

    # See if we're being called as @register or @register().
    if func is None:
        return lambda f: register(f, key=key)
    key = key or func.__name__
    if key in mapping:
        if not _same_func(mapping[key], func):
            raise ValueError(f'Already registered function {key}')
    else:
        mapping[key] = func
    return func


def scan_for_registered(pattern: str) -> Dict[Any, Callable]:
    """Scan for @register functions using globbing

    Args:
        pattern: glob pattern

    Returns:
        The global `mapping` after scan.
    """
    loaded_modules = set(getattr(m, '__file__', None) for m in sys.modules.values())
    for file in Path().glob(pattern):
        abs_loc = str(file.absolute())
        if abs_loc not in loaded_modules and abs_loc not in EXECUTED:
            runpy.run_path(file)
            EXECUTED.add(abs_loc)
    return mapping
