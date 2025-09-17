from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from .nzloader import NzLoad, Parameters


@dataclass(slots=True)
class HistoryItem:
    name: str
    params: Dict[str, Any]


class NzPreprocessed(NzLoad):
    """Generic post-preprocessing container."""
    def __init__(self, data: np.ndarray, *,
                 parameters: Optional[Parameters] = None,
                 history: Optional[List[HistoryItem]] = None):
        super().__init__(data=data, parameters=parameters)
        self.history: List[HistoryItem] = history or []


class NzImagePreprocessed(NzPreprocessed):
    """Preprocessed widefield/imaging data."""
    pass


class NzEphysPreprocessed(NzPreprocessed):
    """Preprocessed Ephys/NP data."""
    pass
