"""Containing Settings for local things."""
from pathlib import Path

# Assuming this file is in the 'config' directory
_config_file_path = Path(__file__).resolve()
PROJECT_ROOT = _config_file_path.parent.parent  # This will navigate two levels up

WANDB_DIR = PROJECT_ROOT / ".runs"
WANDB_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DATA_DIR = WANDB_DIR / "checkpoints"
RUNS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# wandb
wandb_kwargs = dict(
    entity="mxmn",
    #project='concat_moons',
    project='concat_mnist',
    #project="test",
    dir=WANDB_DIR,
)

# test
if __name__ == "__main__":
    print(PROJECT_ROOT)
