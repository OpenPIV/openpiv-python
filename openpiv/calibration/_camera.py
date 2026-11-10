import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import DTypeLike


__all__ = [
    'BaseCamera'
]


class BaseCamera(ABC):
    def __init__(
        self,
        name: str,
        resolution: tuple[int, int],
        dtype: DTypeLike=np.float64
    ):
        self.name = name
        self.resolution = np.array(resolution, dtype=int)
        self.dtype = np.dtype(dtype)

    @abstractmethod
    def minimize_params(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray
    ):
        msg = 'Camera model must implement `minimize_params`'
        raise NotImplementedError(msg)

    @abstractmethod
    def project_points(
        self,
        object_points: np.ndarray
    ):
        msg = 'Camera model must implement `project_points`'
        raise NotImplementedError(msg)

    @abstractmethod
    def project_to_z(
        self,
        image_points: np.ndarray,
        z: float | np.ndarray
    ):
        msg = 'Camera model must implement `project_to_z`'
        raise NotImplementedError(msg)

    @abstractmethod
    def save_parameters(
        self,
        file_path: str,
        file_name: str | None=None
    ):
        msg = 'Camera model must implement `save_parameters'
        raise NotImplementedError(msg)

    @abstractmethod
    def load_parameters(
        self,
        file_path: str,
        file_name: str
    ):
        msg = 'Camera model must implement `load_parameters'
        raise NotImplementedError(msg)