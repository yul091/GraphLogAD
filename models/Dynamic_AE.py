from transformers import (
    AutoConfig,
    AutoModel,
)
from .SetGPT2 import GPT2LMHeadModel
from .SetBart import BartForConditionalGeneration
from .graph_base import torch, nn, F, Optional, GCN, Tensor, Adj, Batch
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding


class DynamicEncoderDecoder(nn.Module):
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        in_channels: Optional[int] = 1041,
        num_nodes: Optional[int] = None,
        out_channels: Optional[int] = 128,
        num_layers: Optional[int] = 4, 
        dropout: Optional[float] = 0.3, 
        act: Optional[F.relu] = F.relu, 
        use_seq_loss: Optional[bool] = False,
    ):
        super().__init__()
        
        # Define transformer config and model
        if model_path is None:
            model_path = 'xlnet-base-cased' # OR: 'facebook/bart-large', 'gpt2'
        
        self.config = AutoConfig.from_pretrained(model_path)
        self.use_seq_loss = use_seq_loss
        self.embed_dim = self.config.hidden_size

        if self.config.model_type == 'bart':
            # Self-defined set Transformer 
            self.seq2seq = BartForConditionalGeneration.from_pretrained(model_path)
            # # Self-defined graph position embedding
            self.wpe = BartLearnedPositionalEmbedding(
                self.config.max_position_embeddings,
                self.embed_dim,
            )
        else:
            # # Self-defined graph position embedding
            self.wpe = nn.Embedding(
                self.config.max_position_embeddings, 
                self.embed_dim,
            )
            if self.config.model_type == "gpt2":
                # GPT-2 decoder
                self.config.output_hidden_states = True # we need both loss and last_hidden_state
                self.seq2seq = GPT2LMHeadModel.from_pretrained(model_path, config=self.config)
            else:
                # XLNet, Bert
                self.seq2seq = AutoModel.from_pretrained(model_path)
        
        # Adapt vocabulary to node space
        if num_nodes is not None:
            self.seq2seq.resize_token_embeddings(num_nodes) 

        # split the number of layers for the encoder and decoders
        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        # Define graph encoder decoder
        self.shared_encoder = GCN(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=self.embed_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            act=act,
        )
        self.attr_decoder = GCN(
            in_channels=self.embed_dim,
            hidden_channels=out_channels,
            out_channels=in_channels,
            num_layers=decoder_layers,
            dropout=dropout,
            act=act,
        )
        self.struct_decoder = GCN(
            in_channels=self.embed_dim,
            hidden_channels=out_channels,
            out_channels=in_channels,
            num_layers=decoder_layers - 1,
            dropout=dropout,
            act=act,
        )

    def forward(self, x: Tensor, edge_index: Adj, batch: Batch, ids: Tensor, num_graphs: int):
        loss = None # placeholder loss
        # Graph encode
        e_ = self.shared_encoder(x, edge_index) # |V| X E
        
        # Temporal-attentive Learning
        if self.config.model_type == 'bart':
            # Set graph position embedding
            graph_embed = self.wpe((1, num_graphs)) # |G| X E
            node_embed = torch.stack([graph_embed[graphid] for graphid in batch], dim=0) # |V| X E
            e_ = e_ + node_embed # |V| X E

            # Seq encode
            outputs = self.seq2seq.model.encoder(
                inputs_embeds=e_.unsqueeze(0), 
            )
            h = outputs.last_hidden_state.squeeze(0) # |V| X E
            # Calculate seq2seq loss
            if self.use_seq_loss:
                # Split the embedding e_ into former and latter parts
                half_idx = (batch >= int((batch[-1]+1)/2)).nonzero()[0].item()
                # Using former sequence to predict latter sequence
                seq2seq_outputs = self.seq2seq(
                    inputs_embeds=e_[:half_idx].unsqueeze(0), 
                    labels=ids[half_idx:].unsqueeze(0),
                    decoder_inputs_embeds=e_[half_idx:].unsqueeze(0),
                ) 
                # ('loss', 'logits', 'encoder_last_hidden_state')
                loss = seq2seq_outputs.loss
        else: # xlnet, gpt-2, bert
            # Set graph position embedding
            position_ids = torch.arange(0, num_graphs, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).view(-1, num_graphs)
            graph_embed = self.wpe(position_ids).squeeze(0) # |G| X E
            node_embed = torch.stack([graph_embed[graphid] for graphid in batch], dim=0) # |V| X E
            e_ = e_ + node_embed # |V| X E
            # Seq transform (decode)
            if self.config.model_type == 'gpt2':
                if self.use_seq_loss:
                    outputs = self.seq2seq(
                        inputs_embeds=e_.unsqueeze(0),
                        labels=ids.unsqueeze(0),
                    )
                    loss = outputs.loss
                    # print('language modeling loss: {}'.format(loss.item()))
                    h = outputs.hidden_states[-1].squeeze(0) # |V| X E
                else:
                    outputs = self.seq2seq.transformer(
                        inputs_embeds=e_.unsqueeze(0),
                    )
                    h = outputs.last_hidden_state.squeeze(0) # |V_cut| X E
            elif self.config.model_type  == 'xlnet':
                outputs = self.seq2seq(
                    inputs_embeds=e_.unsqueeze(0),
                )
                h = outputs.last_hidden_state.squeeze(0) # |V_cut| X E
            # Seq transform (BERT encoder)
            else:
                outputs = self.seq2seq.encoder(e_.unsqueeze(0))
                h = outputs.last_hidden_state.squeeze(0) # |V_cut| X E

        # Decode feature matrix
        x_ = self.attr_decoder(h, edge_index) # |V| X E
        # Decode adjacency matrix
        h_ = self.struct_decoder(h, edge_index) # |V| X E
        s_ = h_ @ h_.T # |V_cut| X |V_cut|
        
        return x, x_, s_, loss