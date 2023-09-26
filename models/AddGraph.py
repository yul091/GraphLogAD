from typing import Union, Callable
from torch.nn.utils.rnn import pad_sequence
from .graph_base import torch, nn, GCN, Data, Batch


class AddGraph_Base(nn.Module):
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
        self.decoder_layers = int(num_layers / 2)
        self.encoder_layers = num_layers - self.decoder_layers
        self.shared_encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=self.encoder_layers,
            out_channels=hid_dim,
            dropout=dropout,
            act=act,
        )
        self.rnn = nn.GRU(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=self.decoder_layers,
            batch_first=True,
            dropout=dropout,
        )


    def forward(self, G: Union[Batch, Data]):
        """
        Inputs:
            x (node features): |V| X in_channels
            edge_index (adjacency matrix): 2 X |E|
        Outputs:
            x_ (estimated node features): |V| X in_channels
        """
        x_list = [graph.x for graph in G.to_data_list()]
        batched_x = pad_sequence(x_list, batch_first=True) # |G| X |V| X in_channels
        h0 = batched_x.mean(dim=1).expand(
            self.decoder_layers, 
            batched_x.shape[0], 
            batched_x.shape[2],
        ) # num_layers X |G| X in_channels
        # GCN encoder
        hidden = self.shared_encoder(G.x, G.edge_index) # |V| X in_channels
        h_list = [hidden[G.batch == i] for i in range(G.num_graphs)]
        batched_h = pad_sequence(h_list, batch_first=True) # |G| X |V| X in_channels
        # RNN decoder
        x_, h = self.rnn(batched_h, h0) # |G| X |V| X in_channels
        x_dense = torch.cat(
            [x_[i, :h_list[i].shape[0]] for i in range(G.num_graphs)], 
            dim=0,    
        ) # |V| X in_channels
        return x_dense