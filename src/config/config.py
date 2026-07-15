import yaml
from dataclasses import dataclass
from interface.types import FluidProperties, SidePlateMassProp, IterationParameters


@dataclass
class SimulationConfig:
    fluid: FluidProperties
    side_plate: SidePlateMassProp
    iteration: IterationParameters

    @staticmethod
    def load_from_yaml(
        file_path: str,
    ) -> "SimulationConfig":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        fluid_prop = FluidProperties(**data["fluid"])
        side_plate_params = SidePlateMassProp(**data["side_plate"])
        iter_params = IterationParameters(**data["iteration"])

        return SimulationConfig(
            fluid=fluid_prop,
            side_plate=side_plate_params,
            iteration=iter_params,
        )


def load_config(file_name: str) -> SimulationConfig:
    return SimulationConfig.load_from_yaml(f"config/{file_name}.yml")
