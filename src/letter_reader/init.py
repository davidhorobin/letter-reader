from layers import Layer
from network import Network


def init_network(layer_sizes):
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
    layers.append(Layer(layer_sizes[-2], layer_sizes[-1]))
    network = Network(layers)
    return network
