import numpy as np
import torch.nn as nn
import torch
from .shifted_softplus import ShiftedSoftplus
from .swish import Swish


class PropertiesEncoderSN(nn.Module):
    def __init__(
            self,
            num_layers=12,
            properties_length=10,
            hidden_dim=128
    ):
        super(PropertiesEncoderSN, self).__init__()

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


class PropertiesEncoder(nn.Module):
    def __init__(
        self,
        num_layers=3,
        properties_length=10,
        hidden_dim=128,
        activation: str = "swish",
    ):

        super(PropertiesEncoder, self).__init__()
        # initialize attributes
        if activation == "ssp":
            Activation = ShiftedSoftplus
        elif activation == "swish":
            Activation = Swish
        else:
            raise ValueError(
                "Argument 'activation' may only take the "
                "values 'ssp', or 'swish' but received '" + str(activation) + "'."
            )

        self.init_linear = nn.Linear(properties_length, hidden_dim)

        residual_mlp = []
        for _ in range(num_layers):
            residual_mlp.append(Activation(hidden_dim))
            layer = nn.Linear(in_features=hidden_dim,
                              out_features=hidden_dim)
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
            residual_mlp.append(layer)
            residual_mlp.append(Activation(hidden_dim))

        self.residual_mlp = nn.Sequential(*residual_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_linear(x)
        y = self.residual_mlp(x)
        return x + y

