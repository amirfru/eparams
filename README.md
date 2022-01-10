# eparams
_Simple_ and _configurable_ parameters/configuration class. No manual schema needed!
<br/> Use python3.6+ type hinting to declare the param type, or let eparams to infer it from the assignment, and keep you from assigning the wrong type down the line!

At its core it is designed as a drop-in replacement for `dataclass`, but with added features to make life easier.

## Installation 
$`pip install eparams`

## Features
- Automatically type-check parameters using [typeguard](https://github.com/agronholm/typeguard).
- Auto typing - doesn't *force* you to add type hints everywhere.
- Define a custom / use our default preprocess function to cast your input to the correct type.
- Use mutable types (`list` `dict` etc) at class definition with no worries and no awkward mandatory factory objects.
- Protect from typos that would normally lead to assigning undeclared parameters.
- Define custom constraints.
- freeze / unfreeze class.
- History tracking.
  - Track assignment to the class to see which attributes were assigned and from what values.
- Added helper methods: 
  - `_to_json()` `_to_yaml()` `_to_dict()` `_from_json()` `_from_yaml()` `_from_dict()`
  - `copy()` `_freeze()` `_unfreeze()` `_history()`
  - `__contains__()` `__eq__()` `__getitem__()` `__setitem__()`
- All above features are fully configurable per parameter and globally.
- Built to be nested, with `conf.sub_conf.some_val` structure, all methods work recursively. Plays well with `dataclass`.
- Helper functions:
  - Recursive comparison: compare configs from different versions of your code / accross experiments
  - Register partial-config functions in a dictionary for easy calls using cli and external configs. 


## Example usage:
See **[sample_project](sample_project)** for an example usage in a "complete" project which includes:
- cli integration with `argparse`.
- delta-config registration.
- constraints.
- frozen class.
- yaml/dict export.

#### Basic usage:
```python
from eparams import params

# Basic usage
@params
class OptimizerParams:
    learning_rate = 1e-2
    weight_decay = 0.001
    batch_size = 1024

@params
class Params:
  name = 'default name'
  tags = ['no', 'problem', 'with', 'list', 'here']
  optim = OptimizerParams()
  verbose = False

config = Params(name='new name')
config.optim.batch_size = 2048  # ok
config.optim.learning_rate = '0.001' # ok, cast to float
config.optim.batch_size = '32' # ok, cast to int
config.verbose = 'True' # ok, cast to bool
config['optim.weight_decay'] = 0  # we can set nested attributes like this as-well
print(config)
# prints: 
# name='new name'
# tags=['no', 'problem', 'with', 'list', 'here']
# optim.learning_rate=0.001
# optim.weight_decay=0
# optim.batch_size=32
# verbose=True
config._to_yaml('/path/to/config.yaml')  # save as yaml

# The following lines will raise an exception
config.optim.batch_size = 'string'  # raises ValueError: invalid literal for int() with base 10: 'string'
config.optim.batch_size = 1.3  # raises TypeError: type of batch_size must be int; got float instead
config.verrrbose = False  # raises ValueError: Cannot assign <verrrbose> to class <<class '__main__.Params'>>, missing from class definition (allow_dynamic_attribute=False)
```

#### Debug and compare configs from old/different runs:
```python
from eparams import params, params_compare

# type_verify=False does not check for typing errors.
# default_preprocessor=None removes the preprocessing step (a custom function is also valid here)
# allow_dynamic_attribute=True allows the class to dynamically set new attributes.
@params(type_verify=False, default_preprocessor=None, allow_dynamic_attribute=True)
class Params:
  name = 'default name'
  my_int = 0

old_config = Params()._from_yaml('/path/to/old/config.yaml', strict=False)  # load some old config
print(old_config._history())  # show parameters that were loaded and their pre-loaded value
params_not_in_old_yaml = [k for k, hist in old_config._history(full=True).items() if not hist]
modified, added, removed = params_compare(old_config, Params())  # compare two versions of configs
```

#### Preprocessing example:
```python
import enum
from pathlib import Path
from eparams import params, Var


class MyEnum(enum.Enum):
    a = 'a'
    b = 'b'

@params
class Example:
    num: int = 3
    type = MyEnum.b
    path: Path = '/user/home'  # cast to Path
    some_list = [1, 2, 3]  # we copy the list before assignment
    some_string = Var('yo_yo', preprocess_fn=str.lower)  # lower string

ex = Example()
ex.num = '4'  # cast to int
ex.type = 'a'  # cast to MyEnum
ex.some_string = 'HELLO'  # .lower()

assert ex.some_list is not Example.some_list
print(ex)
# prints:
# num=4
# type=<MyEnum.a: 'a'>
# path=PosixPath('/user/home')
# some_list=[1, 2, 3]
# some_string='hello'
```

### Other popular approaches/libraries to define parameters:
- Simply use a `dict`
- Simply use a `dataclass`
- [pydantic](https://pydantic-docs.helpmanual.io/)
- [attrs](https://www.attrs.org/en/stable/index.html)



