from .run import run as default_run
from .icm_run import run as icm_run
from .get_trajectory_run import run as trajectory_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["icm"] = icm_run
REGISTRY["trajectory"] = trajectory_run