from .graph_base import torch, nn, Tensor, GCN
from typing import Union, Callable


class DOMINANT_Base(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hid_dim: int, 
        num_layers: int, 
        dropout: float, 
        act: Union[Callable, None],
    ):
        super().__init__()
        # split the number of layers for the encoder and decoders
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers
        self.shared_encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=encoder_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
        )
        self.attr_decoder = GCN(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers,
            out_channels=in_dim,
            dropout=dropout,
            act=act,
        )
        self.struct_decoder = GCN(
            in_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=decoder_layers - 1,
            out_channels=in_dim,
            dropout=dropout,
            act=act,
        )
    #     # Define edge score function parameters
    #     self.p_a = nn.Parameter(torch.DoubleTensor(in_dim), requires_grad=False)
    #     self.p_b = nn.Parameter(torch.DoubleTensor(in_dim), requires_grad=False)
    #     self.reset_parameters()
    
    # def reset_parameters(self):
    #     p_a_ = self.p_a.unsqueeze(0)
    #     nn.init.xavier_uniform_(p_a_.data, gain=1.414)
    #     p_b_ = self.p_b.unsqueeze(0)
    #     nn.init.xavier_uniform_(p_b_.data, gain=1.414)

    def forward(self, x: Tensor, edge_index: Tensor):
        """
        Outputs:
            x_ (estimated node features): |V| X in_channels
            s_ (estimated adjacency matrix): |V| X |V| 
        """
        # encode
        h = self.shared_encoder(x, edge_index)
        # decode feature matrix
        x_ = self.attr_decoder(h, edge_index)
        # decode adjacency matrix
        h_ = self.struct_decoder(h, edge_index)
        s_ = h_ @ h_.T
        # return reconstructed matrices
        return x_, s_