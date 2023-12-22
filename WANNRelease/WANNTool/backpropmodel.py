import ann
import torch.nn as nn
import torch
import numpy as np

class SelectiveActivation(nn.Module):
  def __init__(self, indices,activation, bias=False):
    super().__init__()
    self.indices = indices
    self.activation=activation
    if len(indices) == 0:
      self.linear = nn.Parameter(torch.ones(1))
    else:
      self.linear = nn.Linear(len(indices), 1 , bias)
  
  def forward(self, x):
    if len(self.indices) == 0:
      return torch.cat([x,self.activation(torch.ones((x.shape[0], 1)) * self.linear)], 1)
    else:
      return torch.cat([x,self.activation(self.linear(x[:,self.indices]))], 1)

class WANNOutput(nn.Module):
    def __init__(self,outputSize):
        super().__init__()
        self.outputSize = outputSize

    def forward(self, x):
        return x[:,-self.outputSize:]
    
def importNetAsTorchModel(fileName,inputSize, outputSize):
    wVec, aVec, wKey = ann.importNet(fileName)
    wSize = int(np.sqrt(wVec.shape[0]))
    w = wVec.reshape((wSize,wSize))
    
    # Create a torch model
    model = nn.Sequential()

    for i in range(inputSize,wSize):
        indices = w[:,i].nonzero()[0]
        activation = getActivationFunction(aVec[i])
        name = f'{actName(aVec[i])}_{i}'
        model.add_module(name, SelectiveActivation(indices, activation))
    model.add_module('output', WANNOutput(outputSize))
    return model


def linear(x):
    return x

def unsigned_step_function(x):
    return 1.0 * (x > 0)

def sin(x):
    return torch.sin(np.pi * x)

def gaussian(x):
    return torch.exp(-x ** 2)

def tanh(x):
    return torch.tanh(x)

def sigmoid(x):
    return (torch.tanh(x / 2.0) + 1.0) / 2.0

def inverse(x):
    return -x

def absolute_value(x):
    return torch.abs(x)

def relu(x):
    return torch.relu(x)

def cosine(x):
    return torch.cos(np.pi * x)

def actName(actId):
    return {
        1: 'linear',
        2: 'unsigned step function',
        3: 'sin',
        4: 'gaussian',
        5: 'tanh',
        6: 'sigmoid',
        7: 'inverse',
        8: 'absolute value',
        9: 'ReLU',
        10: 'cosine',
    }.get(actId, 'unknown activation function')

def getActivationFunction(actId):
    return {
        # 1: linear
        1: linear,
        # 2: unsigned step function
        2: unsigned_step_function,
        # 3: sin
        3: sin,
        # 4: gaussian with mean 0 and sigma 1
        4: gaussian,
        # 5: tanh
        5: tanh,
        # 6: sigmoid (unsigned)
        6: sigmoid,
        # 7: Inverse
        7: inverse,
        # 8: Absolute value
        8: absolute_value,
        # 9: ReLU
        9: relu,
        # 10: cosine
        10: cosine,
    }.get(actId, lambda x: x)