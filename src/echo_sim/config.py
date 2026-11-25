from pathlib import Path

from echo_sim.models import Skin, WorldConfig


def load_world(path: str | Path) -> WorldConfig:
    return WorldConfig.model_validate_json(Path(path).read_text())


def load_skin(path: str | Path, world: WorldConfig) -> Skin:
    skin = Skin.model_validate_json(Path(path).read_text())
    skin.check_covers(world)
    return skin
