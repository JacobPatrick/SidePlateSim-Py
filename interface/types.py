import numpy as np
from dataclasses import dataclass, field


<<<<<<< HEAD
<<<<<<< HEAD
@dataclass(frozen=True)
=======
@dataclass
>>>>>>> 1907e6c ((feat) build a basic structure for the Purdue method)
=======
@dataclass(frozen=True)
>>>>>>> a1d2c1a ((feat) use non-linear root-finding method to solve for the speed that balances the side plate)
class FluidProperties:
    density: float
    viscosity: float


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a1d2c1a ((feat) use non-linear root-finding method to solve for the speed that balances the side plate)
@dataclass(frozen=True)
class GearParams:
    inner_radius: float
    rotation_speed: float


<<<<<<< HEAD
@dataclass
class IterationParameters:
    base_step_size: float
    max_step_size: float
    min_step_size: float
=======
=======
>>>>>>> a1d2c1a ((feat) use non-linear root-finding method to solve for the speed that balances the side plate)
@dataclass
class IterationParameters:
    step_size: float
>>>>>>> 1907e6c ((feat) build a basic structure for the Purdue method)
    total_time: float


@dataclass(frozen=True)
class GearProfilePath:
    gear_poly_path: str
    relief_poly_path: str


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
