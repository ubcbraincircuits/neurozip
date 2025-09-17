from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Parameters:
    rate: Optional[float] = None
    events: Optional[np.ndarray] = None
    # add more later as needed


class NzLoad:

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        parameters: Optional[Parameters] = None,
    ):
        super().__setattr__("data", data)
        super().__setattr__("parameters", parameters or Parameters())

    def __getattr__(self, name):
        """Delegate unknown attribute access to parameters."""
        if hasattr(self.parameters, name):
            return getattr(self.parameters, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name, value):
        """Delegate unknown attribute assignment to parameters."""
        if name in ("data", "parameters"):
            super().__setattr__(name, value)
        elif hasattr(self.parameters, name):
            setattr(self.parameters, name, value)
        else:
            super().__setattr__(name, value)


class NzImageLoad(NzLoad):
    def __init__(self, data=None, parameters=None):
        super().__init__(data, parameters)
        print("Hi Image")


class NzNPLoad(NzLoad):
    def __init__(self, data=None, parameters=None):
        super().__init__(data, parameters)
        print("Hi NP")
