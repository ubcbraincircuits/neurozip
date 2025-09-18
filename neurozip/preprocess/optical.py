from __future__ import annotations
from typing import Any, Dict, Optional, Literal
import numpy as np
import warnings
from ..types.nzloader import NzLoad, NzImageLoad

# from ..types.nzpreprocessor import NzPreprocessed # maybe needed later?

# ---------- helpers ----------


def _as_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (T,H,W) array; got shape {x.shape}.")


def _time_axis_stats(x: np.ndarray):
    # mean/std over time per pixel
    mu = x.mean(axis=0, dtype=np.float64)
    sigma = x.std(axis=0, ddof=0, dtype=np.float64)
    return mu, sigma


# ---------- Base mixin ----------


class _BaseTransformer:
    def get_params(self) -> Dict[str, Any]:
        # expose simple serializable params (avoiding to store arrays here)
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not isinstance(v, np.ndarray)
        }

    def name(self) -> str:
        return type(self).__name__


class DFF(_BaseTransformer):
    """
    ΔF/F transformer.
    type: 'movmean' or 'mean'
      - 'movmean': moving-average baseline with window (odd int recommended)
      - 'mean': global mean over time per pixel
    window: int (only for 'movmean')
    eps: small float for stability
    """

    def __init__(
        self,
        *,
        type: Literal["movmean", "mean"] = "movmean",
        window: int = 300,
        eps: float = 1e-8,
    ):
        self.type = type
        self.window = int(window)  # 10*rate is suggested
        self.eps = float(eps)
        # fitted artifacts
        self._F0: Optional[np.ndarray] = None  # (H,W) or (T,H,W) if movmean

    def fit(self, X: NzLoad) -> "DFF":
        if not isinstance(X, NzImageLoad):
            warnings.warn(
                f"DFF is optimized for NzImageLoad; got"
                f"{type(X).__name__}.",
                stacklevel=2,
            )
        arr = _as_3d(X.data)
        if self.type == "mean":
            self._F0 = arr.mean(axis=0, dtype=np.float64)  # (H,W)
        elif self.type == "movmean":
            # 1D moving average along time, same length (T)
            T, H, W = arr.shape
            k = max(1, self.window)
            # Asymmetric pad ensures output length == T for any k
            pad_left = (k - 1) // 2
            pad_right = k // 2
            kernel = np.ones(k, dtype=np.float64) / k
            # pad along time
            padded = np.pad(
                arr.astype(np.float64),
                ((pad_left, pad_right), (0, 0), (0, 0)),
                mode="edge",
            )
            # conv per pixel via FFT-friendly stride trick
            mv = np.apply_along_axis(
                lambda v: np.convolve(v, kernel, mode="valid"), 0, padded
            )
            self._F0 = mv  # (T,H,W)
        else:
            raise ValueError("DFF.type must be 'movmean' or 'mean'")
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        if self._F0 is None:
            raise RuntimeError("DFF must be fitted before transform.")
        arr = _as_3d(X.data).astype(np.float64, copy=False)
        if self.type == "mean":
            out = (arr - self._F0) / (self._F0 + self.eps)
        else:
            out = (arr - self._F0) / (self._F0 + self.eps)
        return type(X)(
            data=out.astype(np.float32, copy=False), parameters=X.parameters
        )

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- ZScore ----------
class ZScore(_BaseTransformer):
    """
    Z-score over time for each pixel: (x - mean_t) / std_t
    """

    def __init__(self, *, eps: float = 1e-8):
        self.eps = float(eps)
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def fit(self, X: NzLoad) -> "ZScore":
        arr = _as_3d(X.data).astype(np.float64, copy=False)
        mu, sigma = arr.mean(axis=0), arr.std(axis=0, ddof=0)
        self._mu, self._sigma = mu, sigma
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        if self._mu is None or self._sigma is None:
            raise RuntimeError("ZScore must be fitted before transform.")
        arr = _as_3d(X.data).astype(np.float64, copy=False)
        out = (arr - self._mu) / (self._sigma + self.eps)
        return type(X)(
            data=out.astype(np.float32, copy=False), parameters=X.parameters
        )

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Mask ----------
class ApplyMask(_BaseTransformer):  # This needs to be better implemented
    """
    Apply a binary mask (H,W). If mask is None, tries X.parameters.mask.
    Keeps original shape; masked pixels set to 0.
    """

    def __init__(self, mask: Optional[np.ndarray] = None):
        self.mask = mask

    def fit(self, X: NzLoad) -> "ApplyMask":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data)
        mask = self.mask
        if mask is None and hasattr(X.parameters, "mask"):
            mask = getattr(X.parameters, "mask")
        if mask is None:
            warnings.warn(
                "ApplyMask: no mask provided/found;"
                "returning input unchanged.",
                stacklevel=2,
            )
            return X
        if mask.shape != arr.shape[1:]:  # Maybe implement mask reshaping later
            raise ValueError(
                f"Mask shape {mask.shape} must match (H,W)={arr.shape[1:]}."
            )
        out = arr.copy()
        out[:, ~mask.astype(bool)] = 0  # set outside mask to 0; or np.nan?
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Downsample ----------
class Downsample(_BaseTransformer):
    """
    Downsample spatially by integer factor (nearest-neighbor).
    """

    def __init__(self, factor: int = 2):
        self.factor = int(factor)
        if self.factor < 1:
            raise ValueError("Downsample.factor must be >= 1.")

    def fit(self, X: NzLoad) -> "Downsample":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data)
        if self.factor == 1:
            return X  # no-op
        # Maybe implement better resampling later
        # (e.g., skimage.transform.resize)
        out = arr[:, :: self.factor, :: self.factor]
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Clip ----------
class Clip(_BaseTransformer):  # Maybe add percentile-based option later
    """
    Clip values to [min_val, max_val]. Use None for no clipping on that end.
    """

    def __init__(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ):
        if min_val is None and max_val is None:
            raise ValueError(
                "Clip requires at least one of" "min_val or max_val to be set."
            )
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X: NzLoad) -> "Clip":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data)
        out = arr.copy()
        if self.min_val is not None:
            out = np.maximum(out, self.min_val)
        if self.max_val is not None:
            out = np.minimum(out, self.max_val)
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Normalize ----------
class Normalize(_BaseTransformer):
    """
    Normalize entire array to [0,1] based on global min/max.
    Recommended to do Clip first if outliers present (most likely they are!).
    """

    def __init__(self):
        # nothing to fit
        pass

    def fit(self, X: NzLoad) -> "Normalize":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data).astype(np.float64, copy=False)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-12:
            warnings.warn(
                "Normalize: data has near-zero dynamic range;"
                "returning zeros.",
                stacklevel=2,
            )
            out = np.zeros_like(arr, dtype=np.float32)
        else:
            out = (arr - min_val) / (max_val - min_val)
            out = out.astype(np.float32, copy=False)
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Rescale ----------
class Rescale(_BaseTransformer):
    """
    Rescale entire array to [new_min, new_max] based on global min/max.
    """

    def __init__(self, new_min: float = 0.0, new_max: float = 1.0):
        if new_max <= new_min:
            raise ValueError("Rescale requires new_max > new_min.")
        self.new_min = float(new_min)
        self.new_max = float(new_max)

    def fit(self, X: NzLoad) -> "Rescale":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data).astype(np.float64, copy=False)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-12:
            warnings.warn(
                "Rescale: data has near-zero dynamic range;"
                "returning constant array.",
                stacklevel=2,
            )
            out = np.full_like(
                arr,
                fill_value=(self.new_min + self.new_max) / 2,
                dtype=np.float32,
            )
        else:
            scale = (self.new_max - self.new_min) / (max_val - min_val)
            out = (arr - min_val) * scale + self.new_min
            out = out.astype(np.float32, copy=False)
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ----------- Trim ----------
class Trim(_BaseTransformer):
    """
    Trim frames from start and/or end: (T,H,W) -> (T-trim_start-trim_end,H,W)
    """

    def __init__(self, trim_start: int = 0, trim_end: int = 0):
        self.trim_start = int(trim_start)
        self.trim_end = int(trim_end)
        if self.trim_start < 0 or self.trim_end < 0:
            raise ValueError("Trim values must be non-negative.")

    def fit(self, X: NzLoad) -> "Trim":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data)
        T = arr.shape[0]
        start = self.trim_start
        end = T - self.trim_end if self.trim_end > 0 else T
        if start >= end:
            raise ValueError(
                f"Trim values too large for data with {T} frames."
            )
        out = arr[start:end]
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ----------- Remove Artifacts ----------
class RemoveArtifacts(_BaseTransformer):
    """
    Remove artifacts by linear interpolation around given time indices.
    """

    def __init__(self, *, indices: np.ndarray, pad: int = 2):
        self.indices = np.asarray(indices, dtype=int)
        if self.indices.ndim != 1:
            raise ValueError("indices must be 1D array.")
        self.pad = int(pad)
        if self.pad < 0:
            raise ValueError("pad must be non-negative.")

    def fit(self, X: NzLoad) -> "RemoveArtifacts":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data).astype(np.float64, copy=False)
        T, H, W = arr.shape
        for idx in self.indices:
            start = max(0, idx - self.pad)
            end = min(T - 1, idx + self.pad)
            if start >= end:
                continue  # nothing to interpolate
            before_idx = start - 1
            after_idx = end + 1
            if before_idx < 0 or after_idx >= T:
                continue  # cannot interpolate at edges
            num_to_fill = end - start + 1
            before_frame = arr[before_idx]
            after_frame = arr[after_idx]
            for i in range(num_to_fill):
                alpha = (i + 1) / (num_to_fill + 1)
                arr[start + i] = (
                    1 - alpha
                ) * before_frame + alpha * after_frame
        return type(X)(
            data=arr.astype(np.float32, copy=False), parameters=X.parameters
        )

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ----------- Temporal Filter ----------
class TemporalFilter(_BaseTransformer):
    """
    Simple temporal filtering: 'lowpass', 'highpass', or 'bandpass',
    using a butterworth filter.
    """

    def __init__(
        self,
        *,
        filter_type: Literal["lowpass", "highpass", "bandpass"] = "bandpass",
        lowcut: Optional[float] = 0.01,
        highcut: Optional[float] = 10.0,
        fs: float = 30.0,
        order: int = 5,
    ):
        self.filter_type = filter_type
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        # fitted artifacts
        self._b: Optional[np.ndarray] = None
        self._a: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None

    def fit(self, X: NzLoad) -> "TemporalFilter":
        from scipy.signal import butter, lfilter_zi

        if not isinstance(X, NzImageLoad):
            warnings.warn(
                f"TemporalFilter is optimized for NzImageLoad; got"
                f"{type(X).__name__}.",
                stacklevel=2,
            )
        arr = _as_3d(X.data)
        nyq = 0.5 * self.fs
        if self.filter_type == "lowpass":
            if self.highcut is None:
                raise ValueError(
                    "highcut must be specified for lowpass filter."
                )
            normal_cutoff = self.highcut / nyq
            self._b, self._a = butter(
                self.order, normal_cutoff, btype="low", analog=False
            )
        elif self.filter_type == "highpass":
            if self.lowcut is None:
                raise ValueError(
                    "lowcut must be specified for highpass filter."
                )
            normal_cutoff = self.lowcut / nyq
            self._b, self._a = butter(
                self.order, normal_cutoff, btype="high", analog=False
            )
        elif self.filter_type == "bandpass":
            if self.lowcut is None or self.highcut is None:
                raise ValueError(
                    "Both lowcut and highcut must be specified"
                    "for bandpass filter."
                )
            low = self.lowcut / nyq
            high = self.highcut / nyq
            self._b, self._a = butter(
                self.order, [low, high], btype="band", analog=False
            )
        else:
            raise ValueError(
                "filter_type must be 'lowpass', 'highpass', or 'bandpass'."
            )
        # Initialize zi for each pixel
        T, H, W = arr.shape
        self._zi = np.array(
            [
                lfilter_zi(self._b, self._a) * arr[0, h, w]
                for h in range(H)
                for w in range(W)
            ]
        )
        self._zi = self._zi.reshape(H, W, -1)  # shape (H,W,len(a)-1)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        from scipy.signal import lfilter

        if self._b is None or self._a is None or self._zi is None:
            raise RuntimeError(
                "TemporalFilter must be fitted before transform."
            )
        arr = _as_3d(X.data)
        T, H, W = arr.shape
        out = np.zeros_like(arr, dtype=np.float32)
        for h in range(H):
            for w in range(W):
                out[:, h, w], self._zi[h, w] = lfilter(
                    self._b, self._a, arr[:, h, w], zi=self._zi[h, w]
                )
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ----------- Spatial Filter ----------
class SpatialFilter(_BaseTransformer):
    """
    Simple gaussian spatial filtering with given sigma (in pixels).
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)

    def fit(self, X: NzLoad) -> "SpatialFilter":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        from scipy.ndimage import gaussian_filter

        arr = _as_3d(X.data)
        out = np.zeros_like(arr, dtype=np.float32)
        for t in range(arr.shape[0]):
            out[t] = gaussian_filter(arr[t], sigma=self.sigma)
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Threshold ----------
class Threshold(_BaseTransformer):
    """
    Threshold values: set values below threshold to zero.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def fit(self, X: NzLoad) -> "Threshold":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data)
        out = arr.copy()
        out[out < self.threshold] = 0
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)


# ---------- Flatten ----------
class Flatten(_BaseTransformer):  # Not sure if this is useful. Maybe later
    """
    Flatten spatial dimensions: (T,H,W) -> (T, H*W)
    """

    def __init__(self):
        # nothing to fit
        pass

    def fit(self, X: NzLoad) -> "Flatten":
        # nothing to learn; but validate
        _ = _as_3d(X.data)
        return self

    def transform(self, X: NzLoad) -> NzLoad:
        arr = _as_3d(X.data)
        T, H, W = arr.shape
        out = arr.reshape(T, H * W)
        return type(X)(data=out, parameters=X.parameters)

    def fit_transform(self, X: NzLoad) -> NzLoad:
        self.fit(X)
        return self.transform(X)
