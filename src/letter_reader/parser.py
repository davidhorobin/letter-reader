import numpy as np
import gzip
import matplotlib.pyplot as plt
from config import *

path = DATA_DIR / "emnist-balanced-train-images-idx3-ubyte.gz"
f = gzip.open(path, 'r')
f.read(16)
buffer = f.read(NUM_TRAIN_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT)
data = np.frombuffer(buffer, dtype=np.uint8)
data = data.reshape(NUM_TRAIN_IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT)
data = data.transpose(0, 2, 1)
image = np.asarray(data[3478]).squeeze()
plt.imshow(image, cmap='gray')
plt.show()

f.close()