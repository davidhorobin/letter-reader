from pathlib import Path

ROOT = Path(__file__).parent.resolve().parents[1]
DATA_DIR = ROOT / "data"

NUM_TRAIN_IMAGES = 112800
NUM_TEST_IMAGES = 18800

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28