import numpy as np
from torch import nn


class PropertiesEncoder(nn.Module):
    def __init__(self, num_layers=12, properties_length=10, hidden_dim=128):
        super(PropertiesEncoder, self).__init__()

        self.num_layers = num_layers
        self.properties_length = properties_length
        self.hidden_dim = hidden_dim

        #self.linear_mlp = nn.ModuleList()
        selu_mlp = []
        in_features = properties_length
        for _ in range(self.num_layers):
            layer = nn.Linear(in_features=in_features,
                              out_features=self.hidden_dim)
            layer.weight.data.normal_(0.0, np.sqrt(1. / np.prod(layer.weight.shape[1:])))
            selu_mlp.append(layer)
            # Add selu activation module to list of modules
            selu_mlp.append(nn.SELU())
            # linear_mlp.append(nn.AlphaDropout(p=0.2))
            in_features = hidden_dim
        self.selu_mlp = nn.Sequential(*selu_mlp)


    def forward(self, x):
        x = self.selu_mlp(x)
        return x

