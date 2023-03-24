"""Main eparams logic."""
import ast
import datetime
import enum
import sys
import copy
from dataclasses import is_dataclass
from pathlib import Path
from typing import List, Dict, NamedTuple, Optional, Callable, Any, Iterator, Tuple, ClassVar, Sequence, \
    NoReturn, Union, Set, IO
import typeguard

from .constraints import Constraint

if sys.version_info >= (3, 8):
    from typing import get_origin, get_args
else:
    def get_origin(tp):
        return getattr(tp,'__origin__', None)
    def get_args(tp):
        return getattr(tp, '__args__', None)


# typeguard had some non-backwards-compatible changes in V3.0
import inspect
check_type_sig = inspect.signature(typeguard.check_type)
if 'argname' in check_type_sig.parameters:
    check_type = typeguard.check_type
else:
    def check_type(argname, value, expected_type):
        try:
            typeguard.check_type(value, expected_type)
        except Exception as e:
            e.args = (' '.join([f'"{argname}"', *e.args]),)
            raise e


__all__ = ['params',
           'Var',
           'ParamsClass',
           'EparamsConf',
           'preprocess_fn',

           # Helper functions.
           'params_compare',
           'params_dumps',
           'is_eparmas_instance',
           ]

_EPARAMS = '__eparams__'
_POST_INIT = '__post_init__'
MISSING = object()
AUTO = object()
NOINIT = object()
_DICT_NAME = 'initargs_dict'


def _isclass(obj) -> bool:
    try:
        return issubclass(obj, obj)
    except TypeError:
        return False


def is_eparmas_instance(obj) -> bool:
    """Returns True if obj is an instance of a dataclass."""
    return not _isclass(obj) and hasattr(obj, _EPARAMS)


def _to_dict(obj, follow_anyclass = False, follow_dataclass = True) -> Dict[str, Any]:
    if is_eparmas_instance(obj):
        return {name: _to_dict(getattr(obj, name), follow_anyclass, follow_dataclass)
                for name in getattr(obj, _EPARAMS).fields}
    if (follow_anyclass and hasattr(obj, '__dict__') and not _isclass(obj)) or (follow_dataclass and is_dataclass(obj)):
        return {name: _to_dict(val, follow_anyclass, follow_dataclass) for name, val in obj.__dict__.items()}
    else:
        return obj


def _flatten_dict(dct: Dict) -> Iterator[Tuple[List[str], Any]]:
    for key, val in dct.items():
        if isinstance(val, dict):
            for child_keys, child_val in _flatten_dict(val):
                yield [key] + child_keys, child_val
        else:
            yield [key], val


def _flatten_params(obj, follow_anyclass=False, follow_dataclass=True) -> Iterator[Tuple[List[str], Any]]:
    for key, val in obj.__dict__.items():
        if (follow_anyclass and hasattr(val, '__dict__') and not _isclass(val)) \
                or is_eparmas_instance(val) \
                or (follow_dataclass and is_dataclass(val)):
            for child_keys, child_val in _flatten_params(val, follow_anyclass, follow_dataclass):
                yield [key] + child_keys, child_val
        else:
            yield [key], val


def _dict_compare(d1: Dict, d2: Dict) -> Tuple[Dict, Set, Set]:
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    # same = set(o for o in shared_keys if d1[o] == d2[o])
    return modified, added, removed


def params_compare(obj1, obj2, follow_anyclass=False, follow_dataclass=True) -> Tuple[Dict, Set, Set]:
    """Compare two params instances recursively.

    Args:
        obj1: first instance
        obj2: second instance
        follow_anyclass: compare any object with a __dict__
        follow_dataclass: compare dataclasses as-well

    Returns:
        A 3-tuple of:
        - dict of different values between params
        - params that only appear in obj1
        - params that only appear in obj2

    """
    assert is_eparmas_instance(obj1)
    assert is_eparmas_instance(obj2)
    d1 = {tuple(k): val for k, val in _flatten_params(obj1, follow_anyclass=follow_anyclass,
                                                      follow_dataclass=follow_dataclass)}
    d2 = {tuple(k): val for k, val in _flatten_params(obj2, follow_anyclass=follow_anyclass,
                                                      follow_dataclass=follow_dataclass)}
    return _dict_compare(d1, d2)  # modified, added, removed


def params_dumps(obj, key_sep: str = '.', value_sep: str = '=', params_sep = '\n') -> str:
    """Dump params instance to string

    Args:
        obj: params instance
        key_sep: separator between key path
        value_sep: separator before value
        params_sep: separator between different params

    Returns:
        Generated string.

    """
    return params_sep.join(f'{key_sep.join(k)}{value_sep}{v.__repr__()}' for k, v in _flatten_params(obj))


def _try_set_attr(obj, name, value, strict=True):
    try:
        setattr(obj, name, value)
    except Exception as E:
        if strict:
            raise E
        else:
            print(E)


def _from_flatten(obj, flatten_list: Iterator[Tuple[List[str], Any]], strict=True) -> None:
    for path, value in flatten_list:
        _obj = obj
        for p in path[:-1]:
            _obj = getattr(obj, p)
        _try_set_attr(_obj, path[-1], value, strict=strict)


def _from_dict(obj, dct: Dict[str, Any], strict=True) -> None:
    for key, val in dct.items():
        if isinstance(val, dict):
            cur_val = getattr(obj, key)
            if not isinstance(cur_val, dict):
                _from_dict(cur_val, val, strict=strict)
                continue
        _try_set_attr(obj, key, val, strict=strict)


def _to_yaml(obj, path_or_buf: Optional[Union[Path, str, IO]] = None, **kwargs) -> Optional[str]:
    import yaml
    dumper = kwargs.pop('dumper', yaml.Dumper)
    if path_or_buf is None or hasattr(path_or_buf, 'write'):
        return yaml.dump(_to_dict(obj), path_or_buf, Dumper=dumper, **kwargs)
    else:
        with open(path_or_buf, 'w') as f:
            yaml.dump(_to_dict(obj), f, Dumper=dumper, **kwargs)


def _from_yaml(obj, path_or_buf: Union[Path, str, IO], Loader=None, strict=True) -> None:
    import yaml
    Loader = Loader or yaml.Loader
    if hasattr(path_or_buf, 'read'):
        dct = yaml.load(path_or_buf, Loader=Loader)
    else:
        with open(path_or_buf, 'r') as f:
            dct = yaml.load(f, Loader=Loader)
    _from_dict(obj, dct, strict=strict)


def _to_json(obj, path_or_buf: Optional[Union[Path, str, IO]] = None, **kwargs) -> Optional[str]:
    import json
    if path_or_buf is None:
        return json.dumps(_to_dict(obj), **kwargs)
    if hasattr(path_or_buf, 'write'):
        json.dump(_to_dict(obj), path_or_buf, **kwargs)
    else:
        with open(path_or_buf, 'w') as f:
            json.dump(_to_dict(obj), f, **kwargs)


def _from_json(obj, path_or_buf: Union[Path, str, IO], strict=True, **kwargs) -> None:
    import json
    if hasattr(path_or_buf, 'read'):
        dct = json.load(path_or_buf, **kwargs)
    else:
        with open(path_or_buf, 'r') as f:
            dct = json.load(f, **kwargs)
    _from_dict(obj, dct, strict=strict)


def _history(obj, *, full=False, flat=True) -> Union[Dict[str, List[Any]]]:
    if not is_eparmas_instance(obj):
        return {}
    hist: Dict[str, List[Any]] = getattr(obj, _EPARAMS).history.copy()
    for key, val in obj.__dict__.items():
        if is_eparmas_instance(val):
            _hist = _history(val, full=full, flat=False)
            if full or _hist:
                hist.update({key: _hist})
        elif full and not key in hist:
            hist[key] = []
    if flat:
        hist = {'.'.join(p): h for p, h in _flatten_dict(hist)}
    return hist


def _set_runtime_recursive(**kwargs) -> Callable[[Any], None]:
    def _rec(obj):
        if not is_eparmas_instance(obj):
            return
        setattr(obj, _EPARAMS, getattr(obj, _EPARAMS)._replace(**kwargs))
        for key, val in obj.__dict__.items():
            if is_eparmas_instance(val):
                _rec(val)
    return _rec


def preprocess_fn(var: 'Var', value: Any):
    """Preprocess function before assignment."""
    origin = get_origin(var.type)
    if origin and origin in (tuple, list, dict):
        try:
            value = origin(value)
        except (TypeError, ValueError):
            return value
        args = get_args(var.type)
        if not args:
            return value
        if origin == tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                args = [args[0]] * len(value)
            elif len(args) < len(value):
                args = list(args) + [args[-1]] * (len(value) - len(args))
            return tuple(preprocess_fn(Var(tp=t), value=v) for v, t in zip(value, args))
        elif origin == list:
            return [preprocess_fn(Var(tp=args[0]), value=v) for v in value]
        elif origin == dict:
            return {preprocess_fn(Var(tp=args[0]), value=k): preprocess_fn(Var(tp=args[1]), value=v) for k, v in value.items()}
    if isinstance(var.type, enum.EnumMeta):
        return var.type(value)
    if var.type in (int, float, Path) and isinstance(value, str):
        return var.type(value)
    if isinstance(value, Path) and var.type == str:
        return str(value)
    if var.type == bool:
        if value in (0, 1):
            return bool(value)
        elif isinstance(value, str):
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
    if var.type == datetime.datetime and isinstance(value, str):
        return datetime.datetime.fromisoformat(value)
    if var.type != str and isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return copy.deepcopy(value)


attr_overriding_set = {
    '__eq__',
    '__repr__',
    '__contains__',
    '__getitem__',
    '__setitem__',
    '__str__',
    '_flatten_params',
    '_freeze',
    '_from_dict',
    '_from_flatten',
    '_from_json',
    '_from_yaml',
    '_history',
    '_history_reset',
    '_to_dict',
    '_to_json',
    '_to_yaml',
    '_unfreeze',
    'copy',
}


class Var:
    """param meta information."""

    def __init__(self, value: Any = NOINIT, name: str = '', tp: Any = AUTO,
                 preprocess_fn: Optional[Callable] = MISSING,
                 type_verify: bool = MISSING,
                 constraints: Union[Callable[..., bool], Sequence[Callable[..., bool]]] = ()):
        self.name = name
        self.preprocess_fn = preprocess_fn
        self.value = value
        self.type = self._guess_type(tp)
        self.constraints = [constraints] if callable(constraints) else constraints
        self.type_verify = type_verify

    def _guess_type(self, tp: Any) -> Any:
        if tp is AUTO:
            if callable(self.preprocess_fn) \
                    and hasattr(self.preprocess_fn, '__annotations__') \
                    and 'return' in self.preprocess_fn.__annotations__:
                return self.preprocess_fn.__annotations__['return']
            if self.value is NOINIT:
                return MISSING
            if callable(self.preprocess_fn) and self.preprocess_fn is not preprocess_fn:
                return MISSING
            return type(self.value)
        return tp

    def func_sig(self) -> str:
        self_str = f'{_DICT_NAME}["{self.name}"]'
        return self.name + (f': {self_str}.type' if self.type is not MISSING else '') \
                         + (f'={self_str}.value' if self.value is not MISSING else '')

    def body_sig(self) -> str:
        ret = f' self.{self.name} = {self.name}'
        if self.value is NOINIT or self.value is Var:
            ret = f" if {self.name} is not NOINIT and {self.name} is not Var:\n" + \
                  "\n".join(" " + line for line in ret.split('\n'))

        return ret

    def validate(self, value: Any) -> NoReturn:
        if self.constraints:
            msgs = []
            for c in self.constraints:
                if not c(value):
                    msgs.append(c.description if isinstance(c, Constraint) else 'failed constraint test')
            if msgs:
                raise ValueError(f'{self.name}={value}', *msgs)
        if not self.type_verify or self.type is MISSING:
            return
        check_type(self.name, value, self.type)

    def preprocess(self, value: Any) -> Any:
        fn = self.preprocess_fn
        if callable(fn):
            # check if one or more args are needed
            if hasattr(fn, '__code__') and fn.__code__.co_argcount - fn.__code__.co_kwonlyargcount > 1:
                return fn(self, value)
            else:
                return fn(value)
        return value

    @classmethod
    def from_val(cls, val: Any, name: str, params: 'EparamsConf', cls_annotations: Dict[str, Any]) -> 'Var':
        if isinstance(val, cls):
            return cls(val.value, name, tp=cls_annotations.get(name, val.type),
                       preprocess_fn=val.preprocess_fn if val.preprocess_fn is not MISSING \
                           else params.default_preprocessor,
                       constraints=val.constraints,
                       type_verify=val.type_verify if val.type_verify is not MISSING else params.type_verify)
        else:
            default_type = AUTO if params.auto_typing else MISSING
            return cls(val, name, tp=cls_annotations.get(name, default_type), preprocess_fn=params.default_preprocessor,
                       type_verify=params.type_verify)

    def __repr__(self):
        return f'Var({self.__dict__})'


class EparamsConf(NamedTuple):
    """Config for the eparams wrapper."""

    type_verify: bool = True
    auto_typing: bool = True
    frozen: bool = False
    track_history: bool = True
    default_preprocessor: Optional[Callable] = preprocess_fn
    validate_on_setattr: bool = True
    attr_overriding: Set[str] = attr_overriding_set
    allow_dynamic_attribute: bool = False


class _EasyParamsRuntime(NamedTuple):
    params: EparamsConf
    fields: Dict[str, Var]
    history: Optional[Dict[str, List[Any]]] = None
    frozen: bool = False
    track_history: bool = False
    dict_orig: Optional[Dict] = None

    def validate(self, obj):
        _cls_dict = self.dict_orig
        for name in _cls_dict:
            if name == _EPARAMS:
                continue
            if name not in self.fields:
                raise ValueError(f'attribute <{name}> exists in self, but missing from class definition <{obj.__class__}>')
        for name in self.fields:
            if name not in _cls_dict:
                raise ValueError(f'<{name}> is not initialized.')

            val = getattr(obj, name)
            self.fields[name].validate(val)

            if is_eparmas_instance(val):
                getattr(val, _EPARAMS).validate(val)

    def new(self, **kwargs):
        return self._replace(**{'history': {}, 'frozen': self.params.frozen, 'track_history': self.params.track_history,
                                **kwargs})


def _create_init_fn(cls: type, args: Dict[str, Var], params: EparamsConf, *, globals=None):
    # Note that we mutate locals when exec() is called.  Caller
    # beware!  The only callers are internal to this module, so no
    # worries about external callers.
    if not args:
        return lambda self: None

    name = '__init__'
    _locals = {_DICT_NAME: args, 'cls': cls}
    arglist = [arg for arg in args.values() if arg.value is not MISSING]
    if not hasattr(cls, _POST_INIT):
        arglist = [arg for arg in args.values() if arg.value is MISSING] + arglist
    args_signature = 'self, *, ' + ','.join(arg.func_sig() for arg in arglist) + ', **kwargs'
    body = '\n'.join(f'{arg.body_sig()}' for arg in arglist)
    # body += f'\n print(kwargs)'
    body += f'\n for k, v in kwargs.items(): setattr(self, k, v)'
    if hasattr(cls, _POST_INIT):
        body += f'\n self.{_POST_INIT}()'
    body += f'\n self.{_EPARAMS} = self.{_EPARAMS}.new(dict_orig=super(self.__class__, self).__getattribute__("__dict__"))'
    body += f'\n self.{_EPARAMS}.validate(self)'

    # Compute the text of the entire function.
    txt = f'def {name}({args_signature}):\n{body}'

    exec(txt, globals, _locals)
    return _locals[name]


def _get_init_args(cls, params: EparamsConf) -> Dict[str, Var]:
    cls_annotations = cls.__dict__.get('__annotations__', {})

    def valid_param(name: str, val: Any):
        if name.startswith('__'):
            return False
        if (name, val) in (('copy', copy.deepcopy), ('_to_dict', _to_dict), ('_flatten_params', _flatten_params)):
            return False
        tp = cls_annotations.get(name, None)
        if tp is ClassVar or (get_origin(tp) is ClassVar):
            return False
        return True

    init_args = {k: Var.from_val(v, k, params=params, cls_annotations=cls_annotations) for k, v in cls.__dict__.items()
                 if valid_param(k, v)}
    init_args_no_value = {k: Var(MISSING, k, tp=tp)
                          for k, tp in cls_annotations.items() if k not in init_args and valid_param(k, MISSING)}
    init_args.update(init_args_no_value)
    return init_args


def _process_class(cls: type, params: EparamsConf) -> type:
    fields: Dict[str, Var] = {}
    for b in cls.__mro__[-1:0:-1]:
        if hasattr(b, _EPARAMS) and isinstance(getattr(b, _EPARAMS), _EasyParamsRuntime):
            fields.update(getattr(b, _EPARAMS).fields)
        else:
            fields.update(_get_init_args(b, params))
    fields.update(_get_init_args(cls, params))

    def __setattr__(self, name: str, val):
        if isinstance(self, type) or name == _EPARAMS:  # not an instance
            return super(cls, self).__setattr__(name, val)
        _runtime: _EasyParamsRuntime = getattr(self, _EPARAMS)
        if _runtime.frozen:
            raise ValueError(f'class is frozen, cannot modify values ({name}={val}).')
        if name not in _runtime.fields:
            if _runtime.params.allow_dynamic_attribute:
                _runtime.fields[name] = Var.from_val(val, name, params=_runtime.params, cls_annotations={})
            else:
                raise ValueError(f'Cannot assign <{name}> to class <{cls}>, missing from class definition '
                                 f'(allow_dynamic_attribute={_runtime.params.allow_dynamic_attribute})')
        val = _runtime.fields[name].preprocess(val)
        if _runtime.fields[name].type is MISSING and _runtime.params.auto_typing:
            _runtime.fields[name].type = type(val)
        if params.validate_on_setattr:
            _runtime.fields[name].validate(val)
        if _runtime.track_history:
            if not name in _runtime.history:
                if hasattr(self, name):  # can be missing if allow_dynamic_attribute is True and it is first assignment
                    _runtime.history[name] = [getattr(self, name)]
            else:
                _runtime.history[name].append(getattr(self, name))
        super(cls, self).__setattr__(name, val)

    def __repr__(self):
        rpr = f'{cls.__name__} <{self.__dict__}>'
        if len(rpr) > 200:
            rpr = rpr[:197] + '...'
        if getattr(self, _EPARAMS).frozen:
            rpr += ' FROZEN'
        return rpr

    def __getattribute__(self, item):
        if item == '__dict__':
            return {name: getattr(self, name) for name in getattr(self, _EPARAMS).fields}
        return super(cls, self).__getattribute__(item)

    def __getstate__(self):
        return getattr(self, _EPARAMS).dict_orig or self.__dict__

    def __setstate__(self, state):
        __dict__ = super(self.__class__, self).__getattribute__("__dict__")
        for k, v in state.items():
            __dict__[k] = v
        runtime: _EasyParamsRuntime = getattr(self, _EPARAMS)
        __dict__[_EPARAMS] = runtime.new(dict_orig=__dict__)

    def __getitem__(self, key):
        """Get attributes using dot-separated string path"""
        obj = self
        for p in key.split('.'):
            obj = getattr(obj, p)
        return obj

    def __contains__(self, item):
        """Check if dot-separated string path exists"""
        try:
            __getitem__(self, item)
            return True
        except:
            return False

    def __setitem__(self, key, value):
        """Set attributes using dot-separated string path"""
        *path, name = key.split('.')
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        obj = self
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, name, value)

    def __eq__(self, other):
        if not is_eparmas_instance(other):
            return  False
        return not any(params_compare(self, other))

    setattr(cls, _EPARAMS, _EasyParamsRuntime(params, fields))
    setattr(cls, '__getattribute__', __getattribute__)
    setattr(cls, '__getstate__', __getstate__)
    setattr(cls, '__init__', _create_init_fn(cls, fields, params))
    setattr(cls, '__setattr__', __setattr__)
    setattr(cls, '__setstate__', __setstate__)

    def setattrif(name, func):
        if name in params.attr_overriding:
            if not name.startswith('__') and hasattr(cls, name):  # already user defined.
                return
            setattr(cls, name, func)

    setattrif('__eq__', __eq__)
    setattrif('__contains__', __contains__)
    setattrif('__getitem__', __getitem__)
    setattrif('__setitem__', __setitem__)
    setattrif('__repr__', __repr__)
    setattrif('__str__', params_dumps)
    setattrif('_flatten_params', _flatten_params)
    setattrif('_freeze', _set_runtime_recursive(frozen=True))
    setattrif('_from_dict', _from_dict)
    setattrif('_from_flatten', _from_flatten)
    setattrif('_from_json', _from_json)
    setattrif('_from_yaml', _from_yaml)
    setattrif('_history', _history)
    setattrif('_history_reset', _set_runtime_recursive(history={}))
    setattrif('_to_dict', _to_dict)
    setattrif('_to_json', _to_json)
    setattrif('_to_yaml', _to_yaml)
    setattrif('_unfreeze', _set_runtime_recursive(frozen=False))
    setattrif('copy', copy.deepcopy)

    return cls


_defaults = EparamsConf()


def params(_cls=None, *,
           type_verify=_defaults.type_verify,
           auto_typing=_defaults.auto_typing,
           frozen=_defaults.frozen,
           track_history=_defaults.track_history,
           default_preprocessor=_defaults.default_preprocessor,
           validate_on_setattr=_defaults.validate_on_setattr,
           attr_overriding=None,
           allow_dynamic_attribute=_defaults.allow_dynamic_attribute,
           ):
    """Returns the same class as was passed in, with extra methods.

    Args:
        _cls: the class to wrap.
        type_verify: use `typeguard` to verify type is consistent.
        auto_typing: guess param's type when no annotation is given.
        frozen: do not allow changes to params after init.
        track_history: track history of params values.
        default_preprocessor: default preprocessor function to execute before changing param's value
        validate_on_setattr: run validation (=constraints + type checking) whenever value is changed
        attr_overriding: the new class methods to add, defaults to attr_overriding_set
        allow_dynamic_attribute: whether to allow for a new attribute to be introduced at runtime.

    Returns:
        The same class as was passed in, with extra methods.

    """
    if attr_overriding is None:
        attr_overriding = _defaults.attr_overriding

    def wrap(cls):
        return _process_class(cls, EparamsConf(type_verify=type_verify,
                                               auto_typing=auto_typing,
                                               frozen=frozen,
                                               track_history=track_history,
                                               default_preprocessor=default_preprocessor,
                                               validate_on_setattr=validate_on_setattr,
                                               attr_overriding=attr_overriding,
                                               allow_dynamic_attribute=allow_dynamic_attribute
                                               )
                              )

    # See if we're being called as @params or @params().
    if _cls is None:
        return wrap

    return wrap(_cls)


class ParamsClass:
    """Base params class for alternative syntax/style."""
    def __init_subclass__(cls, **kwargs):
        eparams = getattr(cls, _EPARAMS)
        if isinstance(eparams, _EasyParamsRuntime):
            eparams = eparams.params
        x = params(cls, **eparams._asdict())
        return x

    __eparams__ = EparamsConf()
