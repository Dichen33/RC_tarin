from pathlib import Path


MJLAB_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_ROOT = MJLAB_ROOT.parent
PROJECT_ROOT = MUJOCO_ROOT.parent
ROBOT_ROOT = MUJOCO_ROOT / "htdw_4438"
MODEL_PATH = ROBOT_ROOT / "scene.xml"
IK_CONTROL_PATH = ROBOT_ROOT / "ik_control.py"
RUNS_ROOT = MJLAB_ROOT / "runs"
MODEL_ROOT = MJLAB_ROOT / "model"
LEGACY_LOGS_ROOT = MJLAB_ROOT / "logs"
GENERATED_ROOT = MJLAB_ROOT / "generated"
TERRAIN_ROOT = GENERATED_ROOT / "terrains"
