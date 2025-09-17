from typing import Optional, Union
import numpy as np
from .types.nzloader import NzLoad, Parameters
from .dataloaders import dispatch_load


def load(
    payload: Union[str, np.ndarray],
    *,
    kind: Optional[str] = None,
    parameters: Optional[Parameters] = None,
    **kwargs,
) -> NzLoad:
    """
    Public loader: nz.load(...)

    Examples:
      nz.load("file.tif", kind="widefield")
      nz.load(arr, kind="ndarray")
      nz.load("file.tif")  # kind inferred from extension
    """
    return dispatch_load(payload, kind=kind, parameters=parameters, **kwargs)
