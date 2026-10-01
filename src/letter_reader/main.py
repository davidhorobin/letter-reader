from init import init_network
from config import *
from parser import *
from graphics import show_char
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

layer_sizes = [784, 256, 128, 47]
network = init_network(layer_sizes)

labels_map = labels_dict(MAPPING_PATH)
images_arr = images_to_array(TRAINING_IMG_PATH)
logging.info("Images loaded")
labels_arr = labels_to_array(TRAINING_LABEL_PATH)
logging.info("Labels loaded")

print(network.forward(np.concatenate(images_arr[0])))
print(np.argmax(network.forward(np.concatenate(images_arr[0]))))
