from pathlib import Path

ROOT = Path(__file__).parent.resolve().parents[1]
DATA_DIR = ROOT / "data"

MAPPING_PATH = DATA_DIR / "emnist-balanced-mapping.txt"
TRAINING_IMG_PATH = DATA_DIR / "emnist-balanced-train-images-idx3-ubyte.gz"
TRAINING_LABEL_PATH = DATA_DIR / "emnist-balanced-train-labels-idx1-ubyte.gz"
TEST_IMG_PATH = DATA_DIR / "emnist-balanced-test-images-idx3-ubyte.gz"
TEST_LABEL_PATH = DATA_DIR / "emnist-balanced-test-labels-idx1-ubyte.gz"

NUM_TRAIN_IMAGES = 112800
NUM_TEST_IMAGES = 18800

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28