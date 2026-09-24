"""Load the sized vehicle saved by examples/size_vehicle.py."""
import json
from pathlib import Path

from .sizing import LaunchSite
from .vehicle import Vehicle, VehicleSpec, build_vehicle

SIZED = Path(__file__).resolve().parents[1] / "data" / "sized_vehicle.json"


def load_sized(path: Path = SIZED) -> tuple[VehicleSpec, Vehicle, LaunchSite]:
    d = json.loads(path.read_text())
    spec = VehicleSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["spec"].items()})
    return spec, build_vehicle(spec), LaunchSite(**d["site"])
