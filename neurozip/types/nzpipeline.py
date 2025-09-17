from __future__ import annotations
from typing import (
    Protocol,
    runtime_checkable,
    Iterable,
    List,
    Any,
    Optional,
    Type,
    Dict,
)
import warnings
from .nzloader import NzLoad, NzImageLoad
from .nzpreprocessor import NzPreprocessed, NzImagePreprocessed, HistoryItem


@runtime_checkable
class Transformer(Protocol):
    """Minimal sklearn-like API."""
    def fit(self, X: NzLoad) -> "Transformer":
        ...

    def transform(self, X: NzLoad) -> NzLoad:
        ...

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)

    def get_params(self) -> Dict[str, Any]:
        ...

    def name(self) -> str:
        ...


class Pipeline:
    """Ordered composition of transformers (preprocessing)."""
    def __init__(self, steps: Iterable[Transformer], *,
                 expect: Optional[Type[NzLoad]] = None):
        self.steps: List[Transformer] = list(steps)
        self.expect = expect  # optional input type guard

    def _check_type(self, X: NzLoad) -> None:
        if self.expect is not None and not isinstance(X, self.expect):
            warnings.warn(
                f"Pipeline expects {self.expect.__name__},"
                f"received {type(X).__name__}. "
                "Proceeding, but transformer(s) may raise errors.",
                stacklevel=2,
            )

    def fit(self, X: NzLoad) -> "Pipeline":
        self._check_type(X)
        cur = X
        for t in self.steps:
            # cheap & simple; many fits need the transformed result
            cur = t.fit_transform(cur)
        return self

    def transform(self, X: NzLoad) -> NzPreprocessed:
        self._check_type(X)
        cur = X
        history: List[HistoryItem] = []
        for t in self.steps:
            cur = t.transform(cur)
            history.append(HistoryItem(name=t.name(), params=t.get_params()))
        # choose preprocessed container class by input kind
        if isinstance(cur, NzImageLoad):
            return NzImagePreprocessed(cur.data,
                                       parameters=cur.parameters,
                                       history=history)
        return NzPreprocessed(cur.data,
                              parameters=cur.parameters,
                              history=history)

    def fit_transform(self, X: NzLoad) -> NzPreprocessed:
        # Fit the pipeline once on X, collecting fitted params;
        # then run a second pass to ensure `transform` logic
        # (and history recording) is identical to `.transform`.
        self._check_type(X)
        cur = X
        for t in self.steps:
            cur = t.fit_transform(cur)
        return self.transform(X)  # builds proper history and final container


def build_pipeline(*, steps: Iterable[Transformer],
                   expect: Optional[Type[NzLoad]] = None) -> Pipeline:
    """Public helper, inspired by sklearn.pipeline.Pipeline."""
    return Pipeline(steps=steps, expect=expect)
