from typing import Callable, Dict, Iterable, Optional, Union, Type, Tuple
import numpy as np
from .utils import read_tif_array, as_ndarray  # add more as needed
from neurozip.types.nzloader import NzLoad, NzImageLoad, NzNPLoad, Parameters

Reader = Callable[[Union[str, np.ndarray]], np.ndarray]
Container = Type[NzLoad]

# Map kind -> callable
# Loader signature: (payload, *, parameters=None, **kwargs) -> NzLoad
_REGISTRY: Dict[str, Tuple[Reader, Container]] = {
    "widefield": (read_tif_array, NzImageLoad),
    "tif": (read_tif_array, NzImageLoad),
    "tiff": (read_tif_array, NzImageLoad),
    "ndarray": (as_ndarray, NzLoad),   # or something else?
}


def register_loader(kind: str, reader: Reader, container: Container) -> None:
    k = kind.lower()
    if k in _REGISTRY:
        raise ValueError(f"Loader for kind '{k}' already exists.")
    _REGISTRY[k] = (reader, container)


def list_kinds() -> Iterable[str]:
    return _REGISTRY.keys()


def _infer_kind_from_path(path: str) -> Optional[str]:
    low = path.lower()
    if low.endswith((".tif", ".tiff")):  # This should be extended
        return "widefield"
    return None


def dispatch_load(
    payload: Union[str, np.ndarray],
    *,
    kind: Optional[str] = None,
    parameters: Optional[Parameters] = None,
    **kwargs
) -> NzLoad:
    # Case 1: kind is unspecified
    if kind is None:
        if isinstance(payload, str):
            # infer from path when possible
            low = payload.lower()
            if low.endswith((".tif", ".tiff")):
                kind = "widefield"
            else:
                raise ValueError(
                    "Could not infer loader kind from path. "
                    "Pass `kind=...` explicitly"
                    "(e.g., 'widefield', 'ndarray')."
                )
        elif isinstance(payload, np.ndarray):
            # Array + no kind → plain NzLoad
            return NzLoad(data=payload, parameters=parameters)
        else:
            raise TypeError(
                "payload must be a file path (str) or a NumPy ndarray."
            )

    # Case 2: kind is specified (or inferred above)
    try:
        reader, container = _REGISTRY[kind.lower()]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown kind='{kind}'. Available: {available}")

    # Readers always return an ndarray
    # For arrays this is identity via `as_ndarray`
    if isinstance(payload, np.ndarray):
        # Directly use the array
        arr = payload
        return container(data=arr, parameters=parameters)
    arr = reader(payload, **kwargs)
    return container(data=arr, parameters=parameters)
