import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    A class used to represent deep q network.

    Attributes
    ----------
    input_size: int
        size of input layer
    hidden_size : int
        size of hidden layer
    output_size : int
        size of output layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_input_layer = nn.Linear(input_size, hidden_size)
        self.linear_hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Feed forward function.

        Parameters:
            x (tensor): input state to the network

        Returns
            x (tensor): output action of the network
        """
        x = F.relu(self.linear_input_layer(x))
        x = F.relu(self.linear_hidden_layer(x))
        x = self.linear_output_layer(x)

        return x
