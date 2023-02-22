import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act_func=nn.ReLU()):
        super(NeuralNet, self).__init__()
        layers = []
        # This the first layer [in_channels, hidden_channels[0]]. For example in this exercise: in 784, out 300
        layers.append(nn.Linear(in_channels, hidden_channels[0]))
        layers.append(act_func)  # activation function
        # Hidden layers
        for i in range(1, len(hidden_channels)):
            layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
            layers.append(act_func)  # activation function
        # This is the last layers [hidden_channels[-1], out_channels]. For example in this exercise: in 100, out 10
        layers.append(nn.Linear(hidden_channels[-1], out_channels))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        out = self.classifier(x)
        return out
