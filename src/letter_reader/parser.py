import numpy as np
import gzip
import matplotlib.pyplot as plt
from config import *

def images_to_array(filename):
    imgs = gzip.open(filename, 'r')
    imgs.seek(0, 2)
    size = int((imgs.tell() - 16) / (IMAGE_WIDTH * IMAGE_HEIGHT))
    imgs.seek(0)

    imgs.read(16)
    buffer = imgs.read(size * IMAGE_WIDTH * IMAGE_HEIGHT)
    data = np.frombuffer(buffer, dtype=np.uint8)
    data = data.reshape(size, IMAGE_WIDTH, IMAGE_HEIGHT)
    data = data.transpose(0, 2, 1)

    imgs.close()

    return data

def labels_to_dict(filename):
    labels = open(filename, 'r')
    label_dict = {}
    for row in labels:
        split = row.split(' ')
        label_dict[int(split[0])] = int(split[1])
    labels.close()
    return label_dict

def show_image(image_matrix):
    image = np.asarray(image_matrix).squeeze()
    plt.imshow(image_matrix, cmap='gray')
    plt.show()

label_map = labels_to_dict(MAPPING_PATH)
data = images_to_array(TRAINING_IMG_PATH)
show_image(data[10])