import numpy as np

from neurozip.types.nzloader import NzLoad


def read_tif(filepath: str) -> NzLoad:
    # Read Tif
    # Convert it to ndarray
    return read_ndarray()


def read_ndarray(data: np.ndarray) -> NzLoad:
    return NzLoad(data)