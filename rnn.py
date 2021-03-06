import numpy as np
import torch
from torch import nn
from torch.nn import ReLU
from torch.nn import Tanh
from abc import ABC, abstractmethod

input_dim = 2
hidden_dim = 2
output_dim = 2

class RNNConfiguration:
    """
    Storage class for configuring RNNs
    """
    def __init__(self):
        """ Default configuration """
        # A matrix representing the input weights
        # These should be randomly initialized.
        self.input_weights = np.random.rand(input_dim, hidden_dim)
        self.input_bias = np.random.rand(hidden_dim)
        # A matrix representing the recurrent weights
        self.recurrent_weights = np.random.rand(hidden_dim, hidden_dim)
        self.recurrent_bias = np.random.rand(hidden_dim)
        # A matrix representing the output weights
        self.output_weights = np.ones((hidden_dim, output_dim))
        self.output_bias = np.ones((output_dim))
        self.T = 2
        self.activation = ReLU()

def make_parameter(matrix):
    return nn.Parameter(torch.tensor(matrix, dtype=torch.double, requires_grad=True))

class UpdatePolicy(ABC):
    """ Abstract base class for updating a weight given a firing history. """
    @abstractmethod
    def update(self, weight, firing_history):
        """ Should return weight update given firing history. """
        pass
class HebbianRule(UpdatePolicy):
    """
        An update policy that manually implements a Hebb's rule type update,
        that increases a weight based on the covariance between their firing
        histories.
    """
    def __init__(self, step_size):
        self.step_size = step_size

    def update(self, weight, firing_history):
        covar = sum(x * y for x, y in zip(*firing_history))
        variances = [sum(x ** 2 for x in history) for history in firing_history]
        return covar / sum(variances) * self.step_size


class UpdateNetwork(nn.Module, UpdatePolicy):
    """ A small neural network used to update the network weights. """
    def __init__(self, input_size, hidden_size): 
        super(UpdateNetwork, self).__init__()
        self.layer1 = make_parameter(np.ones((hidden_size, input_size+1)))
        self.layer2 = make_parameter(np.ones((hidden_size)))
        self.activation = Tanh()

    def update(self, weight, firing_history):
        concat_history = torch.tensor(sum(firing_history, []) + [1])
        y = self.activation(self.layer1 @ concat_history)

        return self.activation(self.layer2 @ y)

class RNNSimple(nn.Module):
    """ A simple recurrrent neural network """
    def __init__(self, config): 
        super(RNNSimple, self).__init__()
        # A matrix representing the input weights
        self.input_weights = make_parameter(config.input_weights)
        self.input_bias = make_parameter(config.input_bias)
        # A matrix representing the recurrent weights
        self.recurrent_weights = make_parameter(config.recurrent_weights)
        self.recurrent_bias = make_parameter(config.recurrent_bias)
        # A matrix representing the output weights
        self.output_weights = make_parameter(config.output_weights)
        self.output_bias = make_parameter(config.output_bias)
        # Number of rounds of recurrent activity
        self.T = config.T
        # ACtivation function
        self.activation = config.activation

    def forward(self, x):
        y = self.activation(self.input_weights @ x + self.input_bias)
        intermediate = [y]
        for i in range(self.T):
            intermediate.append(self.activation(self.recurrent_weights @ intermediate[-1] + self.recurrent_bias))
        return self.activation(self.output_weights @ intermediate[-1] + self.output_bias), intermediate

T = 10
c = RNNConfiguration()
c.T = T
m = RNNSimple(c)
m.forward(torch.DoubleTensor([1,1]))[0].sum().backward()
updater = UpdateNetwork(2 * (T+1), 6)

data = [torch.tensor([0.1, 0.03], dtype=torch.double)]
print("before:")
print(list(m.parameters()))
for x in data:
    output, firing = m.forward(x)
    print(firing)
    # Update recurrent weights
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            with torch.no_grad():
                m.recurrent_weights[i][j] += updater.update(m.recurrent_weights[i][j],
                                                            [[firing[t][i]  for t in range(len(firing))], [firing[t][j] for t in range(len(firing))]])
print("after:")
print(list(m.parameters()))
