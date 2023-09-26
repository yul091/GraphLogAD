from torch_geometric.nn import GATConv
from typing import Union, Callable
from .graph_base import Adj, Tensor, torch, nn, F


class StructureAE(nn.Module):
    """
    Structure Autoencoder in AnomalyDAE model: the encoder
    transforms the node attribute X into the latent representation 
    with the linear layer, and a graph attention layer produces 
    an embedding with weight importance of node neighbors. 
    Finally, the decoder reconstructs the final embedding
    to the original.

    Parameters
    ----------
    in_dim (int): input dimension of node data
    embed_dim (int): the latent representation dimension of node
       (after the first linear layer)
    out_dim (int): the output dim after the graph attention layer
    dropout (float): dropout probability for the linear layer
    act (F, optional): Choice of activation function

    Returns
    -------
    x (torch.Tensor): Reconstructed attribute (feature) of nodes
    embed_x (torch.Tensor): Embed nodes after the attention layer
    """

    def __init__(self, in_dim: int, embed_dim: int, out_dim: int, dropout: float, act: Union[Callable, None]):
        super(StructureAE, self).__init__()
        self.dense = nn.Linear(in_dim, embed_dim)
        self.attention_layer = GATConv(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x: Tensor, edge_index: Adj):
        # encoder
        x = self.act(self.dense(x))
        x = F.dropout(x, self.dropout)
        # torch.use_deterministic_algorithms(False)
        h = self.attention_layer(x, edge_index) # deterministic bug
        # torch.use_deterministic_algorithms(True)
        # decoder
        s_ = torch.sigmoid(h @ h.T)
        return s_, h


class AttributeAE(nn.Module):
    """
    Attribute Autoencoder in AnomalyDAE model: the encoder employs 
    two non-linear feature transform to the node attribute x. 
    The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to
    reconstruct the original node attribute.

    Parameters
    ----------
    in_dim (int): input dimension of node data
    embed_dim (int): the latent representation dimension of node
       (after the first linear layer)
    out_dim (int): the output dim after the graph attention layer
    dropout (float): dropout probability for the linear layer
    act (F, optional): Choice of activation function

    Returns
    -------
    x (torch.Tensor): Reconstructed attribute (feature) of nodes.
    """

    def __init__(self, in_dim: int, embed_dim: int, out_dim: int, dropout: float, act: Union[Callable, None]):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # encoder
        x = self.act(self.dense1(x.T))
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout)
        # decoder
        x = h @ x.T
        return x


class AnomalyDAE_Base(nn.Module):
    """
    AnomalyDAE_Base is an anomaly detector consisting of a structure
    autoencoder and an attribute reconstruction autoencoder.

    Parameters
    ----------
    in_node_dim (int): Dimension of input feature
    in_num_dim (int): Dimension of the input number of nodes
    embed_dim (int): Dimension of the embedding after the first reduced linear layer (D1)
    out_dim (int): Dimension of final representation
    dropout (float, optional): Dropout rate of the model, Default: 0
    act (F, optional): Choice of activation function
    """

    def __init__(
        self, 
        in_node_dim: int, 
        in_num_dim: int, 
        embed_dim: int, 
        out_dim: int, 
        dropout: float, 
        act: Union[Callable, None],
    ):
        super(AnomalyDAE_Base, self).__init__()
        self.num_center_nodes = in_num_dim
        self.structure_ae = StructureAE(
            in_node_dim,
            embed_dim,
            out_dim,
            dropout,
            act,
        )
        self.attribute_ae = AttributeAE(
            self.num_center_nodes,
            embed_dim,
            out_dim,
            dropout,
            act,
        )


    def forward(
        self, 
        x: Tensor, 
        edge_index: Adj, 
        batch_size: int,
    ):
        s_, h = self.structure_ae(x, edge_index)
        if batch_size < self.num_center_nodes:
            x = F.pad(x, (0, 0, 0, self.num_center_nodes - batch_size))
        x_ = self.attribute_ae(x[:self.num_center_nodes], h)
        return x_, s_