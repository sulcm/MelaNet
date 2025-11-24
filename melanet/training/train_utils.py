import ast
import inspect

from typing import get_origin, get_args, Union, Any
from enum import Enum

from torch.optim.optimizer import Optimizer, ParamsT
from transformers import TrainerCallback


class TrainerPhaseDetectorCallback(TrainerCallback):
    def __init__(self):
        self.__trainer_phase: str = "train"

    def __phase_from_trainer_control(self, control):
        self._set_trainer_phase("eval" if control.should_evaluate else "train")
        return self.__trainer_phase

    def get_trainer_phase(self) -> str:
        return self.__trainer_phase

    def _set_trainer_phase(self, phase: str) -> None:
        self.__trainer_phase = phase

    def on_step_begin(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_step_end(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_evaluate(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)


INTERPRET_TRUE = {"1", "true", "yes", "on", "y", "t"}
INTERPRET_NONE = {"", "None", "none", "null", None}


def equal_types(a, b) -> bool:
    """Return True if type annotations a and b represent the same type."""
    # Handle aliases transparently
    if isinstance(a, type) and isinstance(b, type):
        return a is b

    # Unwrap unions
    if get_origin(a) is Union and get_origin(b) is Union:
        return set(get_args(a)) == set(get_args(b))

    # Unwrap Optional
    if get_origin(a) is Union and type(None) in get_args(a):
        return equal_types(a, Union[b, None])

    if get_origin(b) is Union and type(None) in get_args(b):
        return equal_types(Union[a, None], b)

    # Parametric generics
    if get_origin(a) and get_origin(b):
        if get_origin(a) != get_origin(b):
            return False
        return all(equal_types(x, y) for x, y in zip(get_args(a), get_args(b)))

    # Fallback identity
    return a == b


# Boolean parser that handles many formats
def parse_bool(s: str) -> bool:
    return s.lower() in INTERPRET_TRUE

# Convert a string to a Python literal safely
def parse_literal(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except Exception:
        return s  # fallback to raw string

def coerce_value(value: str, annotation):
    """Convert string → annotated type, even with complex annotations."""
    # No annotation → keep raw string
    if annotation is inspect._empty or annotation is str:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Optional[T], Union[T, NoneType]
    if origin is Union and type(None) in args:
        non_none_type = next(t for t in args if t is not type(None))
        if value in INTERPRET_NONE:
            return None
        return coerce_value(value, non_none_type)

    # Handle Union types
    if origin is Union:
        last_error = None
        for t in args:
            try:
                return coerce_value(value, t)
            except Exception as e:
                last_error = e
        raise last_error

    # Enum support
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        try:
            return annotation[value]  # by name
        except KeyError:
            try:
                return annotation(value)  # by value
            except Exception:
                raise ValueError(f"{value!r} is not valid for enum {annotation}")

    # Built-in basic types
    if annotation is bool:
        return parse_bool(value)
    if annotation in (int, float):
        return annotation(value)

    # List, Tuple, Set
    if origin in (list, tuple, set):
        element_type = args[0] if args else str
        elems = parse_literal(value)
        if not isinstance(elems, (list, tuple, set)):
            raise ValueError(f"Expected a list-like literal for {annotation}, got {value}")
        coerced = [coerce_value(str(x), element_type) for x in elems]
        return origin(coerced)

    # Dict
    if origin is dict:
        key_type, val_type = args if args else (str, str)
        parsed = parse_literal(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a dict literal for {annotation}, got {value}")
        return {
            coerce_value(str(k), key_type): coerce_value(str(v), val_type)
            for k, v in parsed.items()
        }

    # Custom class with from_str()
    if hasattr(annotation, "from_str") and callable(getattr(annotation, "from_str")):
        return annotation.from_str(value)

    # Custom class with string constructor
    try:
        return annotation(value)
    except Exception:
        pass

    # Fallback to literal
    literal = parse_literal(value)
    if isinstance(literal, annotation):
        return literal
    else:
        raise ValueError(f"Do not know how to convert {value!r} to {annotation}")

def infer_typed_kwargs(func, str_kwargs: dict[str, str]) -> dict[str, Any]:
    """Call a function with kwargs converted according to its annotation types."""
    sig = inspect.signature(func)
    typed_args = {}

    for name, param in sig.parameters.items():
        if name not in str_kwargs:
            if param.default is inspect._empty:
                if issubclass(func, Optimizer) and param.annotation is ParamsT:
                    # Model parameters (passed in runtime)
                    continue
                raise TypeError(f"Missing required argument: {name}")
            continue  # use default

        annotation = param.annotation
        typed_args[name] = coerce_value(str_kwargs[name], annotation)

    return typed_args


def parse_kwargs_from_cli(str_kwargs: str, func = None) -> dict[str, Any]:
    _str_kwargs = {}
    for mapping in str_kwargs.replace(" ", "").split(","):
        key, value = mapping.split("=")
        _str_kwargs[key] = value

    if _str_kwargs and func is not None:
        return infer_typed_kwargs(func, _str_kwargs)
    else:
        return _str_kwargs