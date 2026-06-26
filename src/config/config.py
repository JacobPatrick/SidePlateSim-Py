import yaml
from dataclasses import dataclass


@dataclass
class FluidProperties:
    density: float
    viscosity: float


@dataclass
class GearParameters:
    module: float
    num_teeth: int
    pressure_angle: float
    inner_radius: float
    rotation_speed: float
    status_vec: tuple


@dataclass
class SidePlateMassProp:
    m: float
    Ic: list[list[float]]
    barycenter: list[float]
    g_vec: list[float] = (0.0, 0.0, -9.81)


@dataclass
class FilmParameters:
    p_lst: list


@dataclass
class IterationParameters:
    step_size: float
    total_time: float


@dataclass
class SimulationConfig:
    fluid: FluidProperties
    gear: GearParameters
    side_plate: SidePlateMassProp
    film: FilmParameters
    iteration: IterationParameters

    @staticmethod
    def load_from_yaml(file_path: str) -> "SimulationConfig":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        fluid_prop = FluidProperties(**data["fluid"])
        gear_params = GearParameters(**data["gear"])
        side_plate_params = SidePlateMassProp(**data["side_plate"])
        film_params = FilmParameters(**data["film"])
        iter_params = IterationParameters(**data["iteration"])

        return SimulationConfig(
            fluid=fluid_prop,
            gear=gear_params,
            side_plate=side_plate_params,
            film=film_params,
            iteration=iter_params,
        )


def load_config(file_name: str) -> SimulationConfig:
    return SimulationConfig.load_from_yaml(f"config/{file_name}.yml")


if __name__ == "__main__":
    config = load_config("SimParams_1")
    print(config)
