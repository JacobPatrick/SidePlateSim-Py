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


@dataclass
class FilmParameters:
    h_base: float
    h_tilt: tuple
    p_0: float
    U_vec: tuple
    ht: float


@dataclass
class SimulationConfig:
    fluid: FluidProperties
    gear: GearParameters
    film: FilmParameters

    @staticmethod
    def load_from_yaml(file_path: str) -> 'SimulationConfig':
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        fluid_prop = FluidProperties(**data['fluid'])
        gear_params = GearParameters(**data['gear'])
        film_params = FilmParameters(**data['film'])

        return SimulationConfig(
            fluid=fluid_prop, gear=gear_params, film=film_params
        )


def load_config(file_name: str) -> SimulationConfig:
    return SimulationConfig.load_from_yaml(f'config/{file_name}.yml')


if __name__ == '__main__':
    config = load_config('SimParams_1')
    print(config)
