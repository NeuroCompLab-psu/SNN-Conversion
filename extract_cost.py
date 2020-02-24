import collections
import torch.nn as nn
import torch
from alexnet import alexnet
from torchsummary import summary

def flatten_network(network, all_layers, count):
    for layer in network.children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, nn.DataParallel):
            flatten_network(layer, all_layers, count)
        if not list(layer.children()):  # if leaf node, add it to list
            if isinstance(layer, nn.Conv2d):
                all_layers['Conv2d.' + str(count['conv'])] = layer
                count['conv'] += 1
            elif isinstance(layer, nn.ReLU):
                all_layers['ReLU.' + str(count['relu'])] = layer
                count['relu'] += 1
            elif isinstance(layer, nn.AvgPool2d):
                all_layers['AvgPool.' + str(count['pooling'])] = layer
                count['pooling'] += 1
            elif isinstance(layer, nn.Linear):
                all_layers['Linear.' + str(count['linear'])] = layer
                count['linear'] += 1
            elif isinstance(layer, nn.Dropout):
                all_layers['Dropout.' + str(count['dropout'])] = layer
                count['dropout'] += 1
            else:
                raise KeyError('Unknown layer')
    return all_layers

def cost(layers, input_channel):
    costs = []
    for name, layer in layers.items():
        if isinstance(layer, nn.Conv2d):
            if '.0' in name:
                layer.kernel_size[0] * layer.kernel_size[1] * input_channel
            else:
                layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
            costs.append()
        elif isinstance(layer, nn.ReLU):
            pass
device = torch.device("cuda")
ann = alexnet().to(device)
summary(ann, input_size=(3,224,224))
all_layers = collections.OrderedDict()
count = {'conv': 0, 'linear': 0, 'pooling': 0, 'relu': 0, 'dropout': 0}
all_layers = flatten_network(ann, all_layers, count)
# all_layers.popitem(last=True)
# all_layers.popitem(last=False)


# all_layers.popitem()
print(count)