from pathlib import Path

# Repo root = .../alz-ftd-ctl-reve
REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOW_DIR   = REPO_ROOT / "cache" / "windows"
RESULTS_ROOT = REPO_ROOT / "results"

# Dataset-specific path — update this for your machine / environment
PARTICIPANTS_TXT = Path(
    r"D:\ACADEMICS\datasets\alz-ftd-ctl\ds004504\derivatives\participants.txt"
)
