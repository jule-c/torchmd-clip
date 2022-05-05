from typing import Optional, Tuple
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    Distance,
    rbf_class_mapping,
    act_class_mapping,
)


class TorchMD_GN(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        neighbor_embedding=True,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
        aggr="add",
    ):
        super(TorchMD_GN, self).__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert aggr in [
            "add",
            "mean",
            "max",
        ], 'Argument aggr must be one of: "add", "mean", or "max"'

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.neighbor_embedding = neighbor_embedding
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z
        self.aggr = aggr

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance = Distance(
            cutoff_lower, cutoff_upper, max_num_neighbors=max_num_neighbors
        )
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = (
            NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
            ).jittable()
            if neighbor_embedding
            else None
        )

        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(
                hidden_channels,
                num_rbf,
                num_filters,
                act_class,
                cutoff_lower,
                cutoff_upper,
                aggr=self.aggr,
            )
            self.interactions.append(block)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        if self.neighbor_embedding is not None:
            self.neighbor_embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(self,
                z: Tensor,
                pos: Tensor,
                batch: Tensor,
                s: Optional[Tensor] = None,
                q: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        x = self.embedding(z)

        edge_index, edge_weight, _ = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_weight, edge_attr)

        return x, None, z, pos, batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper}, "
            f"aggr={self.aggr})"
        )


class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        num_filters,
        activation,
        cutoff_lower,
        cutoff_upper,
        aggr="add",
    ):
        super(InteractionBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            activation(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels,
            hidden_channels,
            num_filters,
            self.mlp,
            cutoff_lower,
            cutoff_upper,
            aggr=aggr,
        ).jittable()
        self.act = activation()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_filters,
        net,
        cutoff_lower,
        cutoff_upper,
        aggr="add",
    ):
        super(CFConv, self).__init__(aggr=aggr)
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.net = net
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        W = self.net(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        # propagate_type: (x: Tensor, W: Tensor)
        x = self.propagate(edge_index, x=x, W=W, size=None)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W