import numpy as np
from activation_functions import *


class Layer:
    def __init__(self, n_in, n_out, activate=relu):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        self.b = np.zeros(n_out)
        self.activate = activate

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        self.z = z
        return self.activate(z)
