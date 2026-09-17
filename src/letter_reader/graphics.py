import matplotlib.pyplot as plt
import numpy as np


def show_char(image_matrix):
    image = np.asarray(image_matrix).squeeze()
    plt.imshow(image_matrix, cmap='gray')
    plt.show()
