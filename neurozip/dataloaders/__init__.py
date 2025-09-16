from typing import Callable, Dict, Iterable, Optional, Union
import numpy as np
from .utils import read_tif, read_ndarray  # add more functions later
from neurozip.types.nzloader import NzLoad, Parameters

# Map kind -> callable
# Loader signature: (payload, *, parameters=None, **kwargs) -> NzLoad
_LOADERS: Dict[str, Callable[..., NzLoad]] = {
    "widefield": read_tif,
    "tiff": read_tif,      # aliases are handy
    "tif": read_tif,
    "ndarray": read_ndarray,  # more to be added later
}


def register_loader(kind: str, fn: Callable[..., NzLoad]) -> None:
    """Allow extensions to register new loaders later."""
    kind = kind.lower()
    if kind in _LOADERS:
        raise ValueError(f"Loader for kind '{kind}' already exists.")
    _LOADERS[kind] = fn


def get_loader(kind: str) -> Callable[..., NzLoad]:
    try:
        return _LOADERS[kind.lower()]
    except KeyError:
        available = ", ".join(sorted(_LOADERS))
        raise ValueError(f"Unknown kind='{kind}'. Available: {available}")


def list_kinds() -> Iterable[str]:
    return _LOADERS.keys()


def dispatch_load(
    payload: Union[str, np.ndarray],
    *,
    kind: Optional[str] = None,
    parameters: Optional[Parameters] = None,
    **kwargs
) -> NzLoad:
    """
    Internal dispatcher used by nz.load().
    - If kind is provided, use it.
    - If kind is None and payload is a str path, infer from extension.
    - If kind is None and payload is an ndarray, assume 'ndarray'.
    """
    if kind is None:
        if isinstance(payload, str):
            low = payload.lower()
            if low.endswith((".tif", ".tiff")):
                kind = "widefield"
            else:
                raise ValueError(
                    "Could not infer loader kind from path. "
                    "Pass `kind=...` explicitly (e.g., 'widefield')."
                )
        elif isinstance(payload, np.ndarray):
            kind = "ndarray"
        else:
            raise TypeError(
                "payload must be a file path (str) or a NumPy ndarray.")

    loader = get_loader(kind)
    return loader(payload, parameters=parameters, **kwargs)
