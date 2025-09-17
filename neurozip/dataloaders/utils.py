import numpy as np
from PIL import Image
from neurozip.types.nzloader import NzLoad, Parameters


def read_tif_array(
    filepath: str, *, parameters: Parameters | None = None
) -> NzLoad:
    """
    Properly load a multi-page TIFF stack using Pillow (PIL).
    Returns NzLoad with data shaped (frames, H, W).
    """
    with Image.open(filepath) as img:
        frames = []
        try:
            while True:
                # Convert each frame to a NumPy array and append to the list
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # End of file reached
        # Stack all frames into a 3D array
        return np.stack(frames, axis=0)


def as_ndarray(data: np.ndarray) -> np.ndarray:
    return data


def read_ndarray(
    data: np.ndarray, *, parameters: Parameters | None = None
) -> NzLoad:
    return NzLoad(data=data, parameters=parameters)
