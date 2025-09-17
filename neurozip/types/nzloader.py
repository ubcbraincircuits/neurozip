from dataclasses import dataclass, fields
from typing import Optional
import numpy as np
import difflib


@dataclass(slots=True)
class Parameters:
    rate: Optional[float] = None
    events: Optional[np.ndarray] = None
    # add more later as needed


class NzLoad:
    _CORE_ATTRS = {f.name for f in fields(Parameters)} | {"data", "parameters"}

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        parameters: Optional[Parameters] = None,
    ):
        super().__setattr__("data", data)
        super().__setattr__("parameters", parameters or Parameters())
        super().__setattr__(
            "_param_names", {f.name for f in fields(self.parameters)}
        )

    def _valid_names(self):
        return self._CORE_ATTRS | self._param_names

    def __getattr__(self, name):
        """Delegate unknown attribute access to parameters."""
        if name in self._param_names:
            return getattr(self.parameters, name)
        # build suggestion for reading
        suggestion = difflib.get_close_matches(name, self._valid_names(), n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'.{hint}"
        )

    def __setattr__(self, name, value):
        """Delegate unknown attribute assignment to parameters."""
        if name in self._CORE_ATTRS:
            super().__setattr__(name, value)
            if name == "parameters":
                super().__setattr__(
                    "_param_names", {f.name for f in fields(self.parameters)}
                )
        elif name in self._param_names:
            setattr(self.parameters, name, value)
        else:
            # compute suggestion
            suggestion = difflib.get_close_matches(
                name, self._valid_names(), n=1
            )
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise AttributeError(
                f"Cannot set unknown attribute '{name}'. "
                f"Allowed: {sorted(self._valid_names())}.{hint}"
            )


class NzImageLoad(NzLoad):
    def __init__(self, data=None, parameters=None):
        super().__init__(data, parameters)
        print("Hi Image")


class NzNPLoad(NzLoad):
    def __init__(self, data=None, parameters=None):
        super().__init__(data, parameters)
        print("Hi NP")
