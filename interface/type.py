import numpy as np
from dataclasses import dataclass, field


@dataclass
class GearProfileDir:
    gear_poly_dir: str
    relief_poly_dir: str


@dataclass
class SidePlateState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class FilmParam:
    h_cells: np.ndarray
    h_grad: tuple[float, float]
    U_cells: np.ndarray
    ht_cells: np.ndarray
    bc_lst: list[float]


@dataclass(frozen=True)
class FluidProp:
    mu: float


@dataclass
class Pressure:
    p: np.ndarray
    F: float
    center: tuple[float, float]


@dataclass(frozen=True)
class SidePlateMassProp:
    m: float
    Ic: np.ndarray
    barycenter: np.ndarray
    g_vec: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))


@dataclass
class ForceTorque:
    F: np.ndarray = field(default_factory=lambda: np.zeros(3))
    M: np.ndarray = field(default_factory=lambda: np.zeros(3))
