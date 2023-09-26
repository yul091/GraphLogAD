import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from argparse import Namespace
from typing import Dict, List, Union
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import degree
import pytorch_lightning as pl

from utils import (
    cal_auc_score, 
    cal_aupr_score, 
    cal_accuracy, 
    cal_cls_report, 
    # classification_report,
    to_dense_adj,
    dense_to_sparse,
)
from models.graph_base import (
    Data,
    Batch,
    Tensor,
    Adj,
    MLP,
    GCN,
)
from models.DOMINANT import DOMINANT_Base
from models.CONAD import CONAD_Base
from models.Anomaly_DAE import AnomalyDAE_Base
from models.SCAN import SCAN
from models.Dynamic_edge import DynamicEdge
from models.DeepTraLog import DeepTraLog_Base
from models.AddGraph import AddGraph_Base

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
import pdb


class EdgeDetectionModel(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.args = hparams
        self.in_channels = self._get_hparam(hparams, 'feature_dim')
        self.embed_dim = 768

        # Logging
        self.start = datetime.now()

        # Logistics
        self.n_gpus = self._get_hparam(hparams, 'n_gpus', 1)
        self.checkpoint_dir = self._get_hparam(hparams, 'checkpoint_dir', '.')
        self.n_workers = self._get_hparam(hparams, 'n_workers', 1)
        self.event_only = self._get_hparam(hparams, 'event_only', False)

        # Training args
        self.lr = self._get_hparam(hparams, 'lr', 1e-3)
        self.weight_decay = self._get_hparam(hparams, 'weight_decay', 1e-5)
        self.train_batch_size = self._get_hparam(hparams, 'train_batch_size', 64)
        self.max_length = self._get_hparam(hparams, 'max_length', 1024)
        self.multi_granularity = self._get_hparam(hparams, 'multi_granularity', False)
        self.global_weight = self._get_hparam(hparams, 'global_weight', 0.5)

        # Model args
        model_kwargs = self._get_hparam(hparams, 'model_kwargs', dict())
        self.out_channels = model_kwargs.get('output_dim', 768)
        self.layers = model_kwargs.get('layers', 3)
        self.dropout = model_kwargs.get('dropout', 0.3)
        self.model_type = model_kwargs.get('model_type', 'dynamic')
        self.alpha = model_kwargs.get('alpha', 0.5)
        self.act = model_kwargs.get('act', F.relu)
        self.beta = model_kwargs.get('beta', 1.0)
        self.mu = model_kwargs.get('mu', 0.3)
        self.gamma = model_kwargs.get('gamma', 0.5)

        # Models
        model_path = self._get_hparam(hparams, 'pretrained_model_path', 'facebook/bart-base')
        self.num_nodes = self._get_hparam(hparams, 'num_nodes')
        # Models
        if self.model_type == 'ae-dominant':
            self.model = DOMINANT_Base(
                in_dim=self.in_channels, 
                hid_dim=self.out_channels, 
                num_layers=self.layers, 
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'ae-anomalydae':
            self.num_nodes = self._get_hparam(hparams, 'num_nodes')
            self.model = AnomalyDAE_Base(
                in_node_dim=self.in_channels,
                in_num_dim=self.num_nodes,
                embed_dim=self.out_channels,
                out_dim=self.out_channels,
                dropout=self.dropout,
                act=self.act,
            )
            self.theta = model_kwargs.get('theta', 1.01)
            self.eta = model_kwargs.get('eta', 1.01)
        elif self.model_type == 'ae-conad':
            self.model = CONAD_Base(
                in_dim=self.in_channels,
                hid_dim=self.out_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
            self.r = model_kwargs.get('r', 0.2)
            self.m = model_kwargs.get('m', 50)
            self.k = model_kwargs.get('k', 50)
            self.f = model_kwargs.get('f', 10)
            self.eta = model_kwargs.get('eta', 0.5)
            margin = model_kwargs.get('margin', 0.5)
            self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)
        elif self.model_type == 'ae-gcnae':
            self.model = GCN(
                in_channels=self.in_channels,
                hidden_channels=self.out_channels,
                out_channels=self.in_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'ae-mlpae':
            self.model = MLP(
                in_channels=self.in_channels,
                hidden_channels=self.out_channels,
                out_channels=self.in_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'ae-scan':
            self.eps = model_kwargs.get('eps', 0.5)
            self.mu = model_kwargs.get('mu', 2)
            self.contamination = model_kwargs.get('contamination', 0.1)
            self.model = SCAN(
                eps=self.eps, 
                mu=self.mu, 
                contamination=self.contamination,
            )
        elif self.model_type == 'deeptralog':
            self.model = DeepTraLog_Base(
                in_dim=self.in_channels, 
                hid_dim=self.out_channels, 
                num_layers=self.layers, 
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'addgraph':
            self.model = AddGraph_Base(
                in_dim=self.in_channels, 
                hid_dim=self.out_channels, 
                num_layers=self.layers, 
                dropout=self.dropout,
                act=self.act,
            )  
        elif self.model_type == 'dynamic':
            self.model = DynamicEdge(
                model_path=model_path,
                in_channels=self.in_channels,
                num_nodes=self.num_nodes,
                out_channels=self.out_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        else:
            raise NotImplementedError('Model type {} not implemented'.format(self.model_type))
        
        if self.model_type in ['dynamic', 'addgraph']:
            # Define edge score function parameters
            # self.p_a = nn.Parameter(torch.DoubleTensor(self.embed_dim), requires_grad=True)
            # self.p_b = nn.Parameter(torch.DoubleTensor(self.embed_dim), requires_grad=True)
            # self.reset_parameters()
            self.p_a = nn.Linear(self.embed_dim, 1, bias=False)
            self.p_b = nn.Linear(self.embed_dim, 1, bias=False)
        
        # Logging
        print('Created {} module \n{} \nwith {:,} GPUs {:,} workers'.format(
            self.model.__class__.__name__, self.model, self.n_gpus, self.n_workers))
        # Loss
        self.mse_loss = MSELoss(reduction='none')
        # Save hyperparameters
        self.global_outputs = defaultdict(np.array)
        self.global_labels = defaultdict(np.array)
        self.train_dists = []
        self.decision_scores = []
        self.train_avg = torch.normal(mean=0, std=1, size=(self.embed_dim,)) # E
        self.save_hyperparameters()
        
    # def reset_parameters(self):
    #     p_a_ = self.p_a.unsqueeze(0)
    #     nn.init.xavier_uniform_(p_a_.data, gain=1.414)
    #     p_b_ = self.p_b.unsqueeze(0)
    #     nn.init.xavier_uniform_(p_b_.data, gain=1.414)
        
    @property
    def on_cuda(self):
        return next(self.parameters()).is_cuda

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default: bool = None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default

    def _sample_nodes(self, batch: Batch):
        perm = torch.randperm(batch.num_graphs)
        accum_nodes = 0
        data_list = []
        for graph_id in perm:
            data = batch.get_example(graph_id)
            if accum_nodes + data.num_nodes <= self.max_length:
                accum_nodes += data.num_nodes
                data_list.append(data)

        return batch.from_data_list(data_list)
    
    def loss_func(self, x, x_, s, s_):
        if self.model_type in ['ae-dominant', 'ae-conad']:
            # attribute reconstruction loss
            diff_attribute = torch.pow(x - x_, 2)
            attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
            # structure reconstruction loss
            diff_structure = torch.pow(s - s_, 2)
            structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
            score = self.alpha * attribute_errors + (1 - self.alpha) * structure_errors
            return score
        elif self.model_type == 'ae-anomalydae':
            # generate hyperparameter - structure penalty
            reversed_adj = 1 - s
            thetas = torch.where(
                reversed_adj > 0, reversed_adj,
                torch.full(s.shape, self.theta).to(self.device))
            # generate hyperparameter - node penalty
            reversed_attr = 1 - x
            etas = torch.where(
                reversed_attr == 1, reversed_attr,
                torch.full(x.shape, self.eta).to(self.device))
            # attribute reconstruction loss
            diff_attribute = torch.pow(x_ - x, 2) * etas
            attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
            # structure reconstruction loss
            diff_structure = torch.pow(s_ - s, 2) * thetas
            structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
            score = self.alpha * attribute_errors + (1 - self.alpha) * structure_errors
            return score
        else:
            raise TypeError(f"Unsupported model type {self.model_type}")
        
    def _data_augmentation(self, x: Tensor, adj: Adj):
        rate = self.r
        num_added_edge = self.m
        surround = self.k
        scale_factor = self.f
        adj_aug, feat_aug = deepcopy(adj), deepcopy(x)
        num_nodes = adj_aug.shape[0]
        label_aug = torch.zeros(num_nodes, dtype=torch.int32)
        prob = torch.rand(num_nodes)
        label_aug[prob < rate] = 1

        # high-degree
        n_hd = torch.sum(prob < rate / 4)
        edges_mask = torch.rand(n_hd, num_nodes) < num_added_edge / num_nodes
        edges_mask = edges_mask.to(self.device)
        adj_aug[prob <= rate / 4, :] = edges_mask.float()
        adj_aug[:, prob <= rate / 4] = edges_mask.float().T

        # outlying
        ol_mask = torch.logical_and(rate / 4 <= prob, prob < rate / 2)
        # torch.use_deterministic_algorithms(False)
        adj_aug[ol_mask, :] = 0 # deterministic Bug
        adj_aug[:, ol_mask] = 0
        # torch.use_deterministic_algorithms(True)

        # deviated
        dv_mask = torch.logical_and(rate / 2 <= prob, prob < rate * 3 / 4)
        feat_c = feat_aug[torch.randperm(num_nodes)[:surround]]
        ds = torch.cdist(feat_aug[dv_mask], feat_c)
        feat_aug[dv_mask] = feat_c[torch.argmax(ds, 1)]

        # disproportionate
        mul_mask = torch.logical_and(rate * 3 / 4 <= prob, prob < rate * 7 / 8)
        div_mask = rate * 7 / 8 <= prob
        feat_aug[mul_mask] *= scale_factor
        feat_aug[div_mask] /= scale_factor

        edge_index_aug = dense_to_sparse(adj_aug)[0].to(self.device)
        feat_aug = feat_aug.to(self.device)
        label_aug = label_aug.to(self.device)
        return feat_aug, edge_index_aug, label_aug
    
    def select_node(self, non_adj: Union[list, int]):
        return random.choice(non_adj) if isinstance(non_adj, list) else non_adj
    
    def get_nonedge(self, s: Tensor):
        non_adj = []
        new_s = s.clone()
        for i in range(s.size()[0]):
            new_s[i,i] = 1
            non_adj.append((new_s[i] == 0).nonzero().squeeze().tolist())
        return non_adj
    
    def score_func(self, hidden: Tensor, rows: Tensor, cols: Tensor, weights: float):
        # if self.model_type == 'dynamic' or self.model_type == 'addgraph':
        #     s = self.p_a * hidden[i] + self.p_b * hidden[j]
        #     s = F.dropout(s, self.dropout, training=self.training)
        #     score = weight * torch.sigmoid(self.beta * torch.norm(s, 2).pow(2) - self.mu)
        # else:
        #     score = weight * torch.sigmoid(hidden[i] @ hidden[j] - self.mu)
        # return score
        # print("rows: ", rows)
        hidden_i = hidden.index_select(0, rows) # |E| x d
        hidden_j = hidden.index_select(0, cols) # |E| x d
        if self.model_type == 'dynamic' or self.model_type == 'addgraph':
            # s = self.p_a.expand_as(hidden_i) * hidden_i + self.p_b.expand_as(hidden_j) * hidden_j
            # s = F.dropout(s, self.dropout, training=self.training)
            # score = weights * torch.sigmoid(self.beta * torch.norm(s, p=2, dim=1).pow(2) - self.mu) # |E|
            s = (self.p_a(hidden_i) + self.p_b(hidden_j)).squeeze(1) # |E|
            score = weights * torch.sigmoid(self.beta * s - self.mu) # |E|
        else:
            score = weights * torch.sigmoid((hidden_i + hidden_j).mean(dim=1) - self.mu) # |E|
        return score
        

    # @profile
    def margin_loss(self, hidden: Tensor, G: Union[Data, Batch], split: str = 'train'):
        score = []
        all_nodes = 0
        
        if split == 'train' or split == 'val':
            # hidden: |V| X E, G: |V| in |G|
            all_degrees = degree(G.edge_index[0], G.num_nodes)
            loss = 0
            for k in range(G.num_graphs): 
                graph: Data = G[k]
                graph_feature = hidden[all_nodes:all_nodes+graph.num_nodes]
                degrees = all_degrees[all_nodes:all_nodes+graph.num_nodes]
                if degrees.size()[0] == 2: # no negative edge exists!
                    continue 
                s = G.s[all_nodes:all_nodes+graph.num_nodes, all_nodes:all_nodes+graph.num_nodes]
                non_adj = self.get_nonedge(s) # get non adjacent nodes
                all_nodes += graph.num_nodes
                
                rows, cols, new_rows, new_cols, weights = [], [], [], [], []
                for i, j in graph.edge_index.T.tolist():
                    # pos_score = self.score_func(graph_feature, i, j, s[i, j])
                    prob_ij = degrees[i]/(degrees[i] + degrees[j]).item() if degrees[i] + degrees[j] else 0
                    # Negative sampling
                    if (not non_adj[i]) and (not non_adj[j]): # node i and j connect to all other nodes
                        continue # no negative edge exists!
                    elif not non_adj[i]: # node i connect to all other nodes (except itself)
                        i_prime, j_prime = self.select_node(non_adj[j]), j
                    elif not non_adj[j]: # node j connect to all other nodes (except itself)
                        i_prime, j_prime = i, self.select_node(non_adj[i])
                    else:
                        if random.random() <= prob_ij: # replace node i
                            i_prime, j_prime = self.select_node(non_adj[j]), j
                        else: # replace node j
                            i_prime, j_prime = i, self.select_node(non_adj[i])
                    
                    rows.append(i)
                    cols.append(j)
                    new_rows.append(i_prime)
                    new_cols.append(j_prime)
                    weights.append(s[i, j])
                    # neg_score = self.score_func(graph_feature, i_prime, j_prime, s[i, j])
                    # if pos_score <= neg_score:
                    #     edge_loss = F.relu(self.gamma + pos_score - neg_score)
                    #     # print('edge_loss', edge_loss)
                    #     loss += edge_loss
                    #     score.append(pos_score.detach().cpu())
                
                rows = torch.tensor(rows, dtype=torch.long, device=hidden.device)
                cols = torch.tensor(cols, dtype=torch.long, device=hidden.device)
                new_rows = torch.tensor(new_rows, dtype=torch.long, device=hidden.device)
                new_cols = torch.tensor(new_cols, dtype=torch.long, device=hidden.device)
                weights = torch.tensor(weights, dtype=torch.float, device=hidden.device)
                pos_scores = self.score_func(graph_feature, rows, cols, weights) # |E|
                neg_scores = self.score_func(graph_feature, new_rows, new_cols, weights) # |E|
                effective_edges = pos_scores <= neg_scores
                edge_loss = F.relu(self.gamma + pos_scores - neg_scores)[effective_edges].sum()
                loss += edge_loss
                score.extend(pos_scores[effective_edges].detach().cpu().tolist())

            if not score:
                score = torch.tensor([])
                loss = torch.tensor(0.0, requires_grad=True)
                return loss, score
            else:
                score = torch.tensor(score)
                return loss/len(score), score
        else:
            for k in range(G.num_graphs): 
                graph: Data = G[k]
                graph_feature = hidden[all_nodes:all_nodes+graph.num_nodes]
                s = G.s[all_nodes:all_nodes+graph.num_nodes, all_nodes:all_nodes+graph.num_nodes]
                all_nodes += graph.num_nodes
                rows, cols, weights = [], [], []
                for i, j in graph.edge_index.T.tolist():
                    rows.append(i)
                    cols.append(j)
                    weights.append(s[i, j])
                    # edge_score = self.score_func(graph_feature, i, j, s[i, j])
                    # score.append(edge_score.detach().cpu())
                
                rows = torch.tensor(rows, dtype=torch.long, device=hidden.device)
                cols = torch.tensor(cols, dtype=torch.long, device=hidden.device)
                weights = torch.tensor(weights, dtype=torch.float, device=hidden.device)
                pos_scores = self.score_func(graph_feature, rows, cols, weights)
                score.extend(pos_scores.detach().cpu().tolist())
                
            # score = torch.stack(score) 
            score = torch.tensor(score)
            return score.mean(), score
                        

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, # l2 regularization
        )
        return optimizer
    
    def global_objective(self, x_: Tensor, G: Union[Data, Batch]):
        x_graph = global_max_pool(x_, G.batch) # V X E -> B X E
        # Handling average feature vector
        targets = self.train_avg.expand(x_graph.shape[0], -1) # B X E
        if self.on_cuda:
            targets = targets.cuda()
        # Calculate loss and save to dict
        individual_loss = self.mse_loss(x_graph, targets).sum(dim=-1) # B
        avg_loss = individual_loss.mean() # float
        return individual_loss, avg_loss

    # @profile
    def training_step(self, batch: Union[Data, Batch], batch_idx: int, split: str = 'train'): 
        # Sampling subgraph
        if batch.num_nodes > self.max_length:
            G = self._sample_nodes(batch)
        else:
            G = batch
            
        # Generate adjacency matrix
        if not G.edge_index.shape[-1]: # empty edge index
            # print("Empty edge index !!!")
            G.s = torch.zeros((G.num_nodes, G.num_nodes))
            if self.on_cuda:
                G.s = G.s.cuda()
        else:
            G.s = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]

        # Automated balancing by std
        if self.alpha is None:
            self.alpha = torch.std(G.s).detach() / (torch.std(G.x).detach() + torch.std(G.s).detach())

        # Forward pass
        if self.model_type.lower() == 'ae-dominant':
            x_, s_ = self.forward(
                x=G.x,
                edge_index=G.edge_index,
            )
        elif self.model_type.lower() == 'ae-anomalydae':
            x_, s_ = self.forward(
                x=G.x,
                edge_index=G.edge_index,
                batch_size=G.num_nodes,
            )
        elif self.model_type.lower() == 'ae-conad':
            x_aug, edge_index_aug, label_aug = self._data_augmentation(G.x, G.s)
            h_aug = self.model.embed(x_aug, edge_index_aug)
            h = self.model.embed(G.x, G.edge_index)
            margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
            margin_loss = torch.mean(margin_loss)
            x_, s_ = self.model.reconstruct(h, G.edge_index)
        elif self.model_type.lower() == 'ae-gcnae':
            x_ = self.forward(
                x=G.x, 
                edge_index=G.edge_index,
            )
        elif self.model_type.lower() == 'ae-mlpae':
            x_ = self.model(
                x=G.x,
            )
        elif self.model_type in ['deeptralog', 'addgraph']:
            x_ = self.forward(
                G = G,
            )
        elif self.model_type == 'dynamic':
            x_ = self.forward(
                x=G.x, 
                edge_index=G.edge_index, 
                batch=G.batch, 
                num_graphs=G.num_graphs, # for generating position embedding
            ) # |V| X E
        else:
            raise NotImplementedError
        
        # Handling scores and loss
        labels = G.y
        individual_loss, avg_loss = self.global_objective(x_, G)
        
        # Calculate loss and save to dict
        if split == 'train' or split == 'val':
            if self.model_type in ['ae-gcnae', 'ae-mlpae', 'deeptralog']:
                if self.model_type == 'deeptralog':
                    loss = avg_loss
                else:
                    loss = torch.mean(torch.mean(F.mse_loss(x_, G.x, reduction='none'), dim=1))
                _, scores = self.margin_loss(x_, G, split=split) # |E|
            elif self.model_type in 'dynamic':
                loss, scores = self.margin_loss(x_, G, split=split) # |E|
                if self.multi_granularity:
                    loss = loss + self.global_weight * avg_loss # B
            elif self.model_type == 'addgraph':
                loss, scores = self.margin_loss(x_, G, split=split) # |E|
            else: # ae-conad, ae-dominant, ae-anomalydae
                scores = self.loss_func(G.x, x_, G.s, s_) # |V|
                if self.model_type == 'ae-conad':
                    loss = self.eta * torch.mean(scores) + (1 - self.eta) * margin_loss
                else:
                    loss = torch.mean(scores)
                _, scores = self.margin_loss(x_, G, split=split) # |E|
                    
            # Store training score distribution for analysis
            if split == 'train':  
                self.decision_scores.extend(scores.detach().cpu().tolist())
                # Update train L2 distances
                self.train_dists.extend(individual_loss.detach().tolist())
        else:
            loss, scores = self.margin_loss(x_, G, split=split) # |E|
            if self.model_type == 'dynamic' and self.multi_granularity:
                loss = loss + self.global_weight * avg_loss # B

            labels = G.y[:scores.shape[0]] # needed when some of the nodes are cut
        
        # print("G.x {}: {}".format(G.x.shape, G.x))
        # print("G.y {}: {}".format(G.y.shape, G.y))
        # print("G.node_label {}: {}".format(G.node_label.shape, G.node_label))
        # print("G.batch {}: {}".format(G.batch.shape, G.batch))
        logging_dict = {'train_loss': loss.detach().item()}
        graph_labels = []
        for k in range(G.num_graphs): 
            graph: Data = G[k]
            graph_labels.append(int(graph.y.sum().item() > 0))
        graph_labels = torch.tensor(graph_labels)
        # graph_labels = [int(G.node_label[G.batch == i].sum().item() > 0) for i in range(G.num_graphs)]
        
        return {
            'loss': loss,
            'graph_loss': individual_loss,
            'scores': scores,
            'preds': x_, 
            'labels': labels,
            'graph_labels': graph_labels,
            'log': logging_dict, # Tensorboard logging for training
            'progress_bar': logging_dict, # Progress bar logging for TQDM
        }

    def training_epoch_end(self, train_step_outputs: List[dict], split: str = 'train'):
        event_scores = torch.cat([ins['scores'].detach().cpu() for ins in train_step_outputs], dim=0) # N
        scores = event_scores.numpy() # N
        
        if split == 'train':
            preds = [ins['preds'].detach().cpu() for ins in train_step_outputs] 
            # Update train dists and thresholds
            sorted_scores = sorted(scores)
            self.thre_max = max(scores)
            self.thre_mean = np.mean(scores)
            self.thre_top80 = sorted_scores[int(0.8*len(scores))]
            print("Epoch {} max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                self.current_epoch, 
                self.thre_max, 
                self.thre_top80, 
                self.thre_mean,
            ))
            # Update graph train dists and thresholds
            self.train_avg = torch.cat(preds, dim=0).mean(dim=0) 
            sorted_train_dists = sorted(self.train_dists)
            self.thre_graph_max = max(self.train_dists)
            self.thre_graph_mean = np.mean(self.train_dists)
            self.thre_graph_top80 = sorted_train_dists[int(0.8*len(self.train_dists))]
            self.train_dists = []
            print("[Graph-level] train avg (sum) {}, max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                self.train_avg.sum(), 
                self.thre_max, 
                self.thre_top80, 
                self.thre_mean,
            ))
            
        elif split == 'val':
            val_loss = sum(scores) / event_scores.shape[0] if event_scores.shape[0] else 0
            # val_loss = sum(scores)
            print('Epoch {} val_loss: {:.4f}'.format(self.current_epoch, val_loss))
        else:
            event_labels = torch.cat([ins['labels'].detach().cpu() for ins in train_step_outputs], dim=0) # N
            scores = torch.cat([ins['scores'].detach().cpu() for ins in train_step_outputs], dim=0).numpy() # N
            labels = event_labels.numpy() # |E|
            num_anomalies = sum(labels)
            normal_rate = 1 - num_anomalies / len(labels)
            
            ####################################################################################################################
            # # Sample normal samples -> keep 1: 1 ratio
            # anomaly_ids = np.where(labels == 1)[0].tolist()
            # normal_ids = np.where(labels == 0)[0].tolist()
            # if len(normal_ids) > num_anomalies:
            #     subnormal_ids = random.sample(normal_ids, num_anomalies)
            #     new_ids = sorted(anomaly_ids + subnormal_ids)
            #     scores = scores[new_ids]
            #     labels = labels[new_ids]
            #     normal_rate = 0.5
            ####################################################################################################################
            
            if self.decision_scores:
                sorted_scores = sorted(self.decision_scores)
                if not hasattr(self, 'thre_max'):
                    self.thre_max = max(self.decision_scores)
                if not hasattr(self, 'thre_mean'):
                    self.thre_mean = np.mean(self.decision_scores)
                if not hasattr(self, 'thre_top80'):
                    self.thre_top80 = sorted_scores[int(0.8*len(self.decision_scores))]
                if not hasattr(self, 'thre_adapt'):
                    self.thre_adapt = sorted_scores[int(normal_rate*len(self.decision_scores))]
            else:
                print("Using default threshold 0.5 for anomaly detection !!!")
                self.thre_max, self.thre_top80, self.thre_mean, self.thre_adapt = 0.8, 0.8, 0.5, 0.8
            print("Predicting {} test samples, {} ({:.2f}%) anomalies, using max thre {:.4f}, adapt thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                len(labels),
                sum(labels),
                sum(labels)*100/len(labels),
                self.thre_max, 
                self.thre_adapt,
                self.thre_top80, 
                self.thre_mean,
            ))
            
            graph_loss = [ins['graph_loss'].detach().cpu() for ins in train_step_outputs]
            graph_loss = torch.cat(graph_loss, dim=0).numpy() # |G|
            graph_labels = torch.cat([ins['graph_labels'].detach().cpu() for ins in train_step_outputs], dim=0) 
            graph_labels = graph_labels.numpy() # |G|
            num_graph_anomalies = sum(graph_labels)
            normal_graph_rate = 1 - num_graph_anomalies / len(graph_labels)
            
            if self.train_dists:
                sorted_train_dists = sorted(self.train_dists)
                if not hasattr(self, 'thre_graph_max'):
                    self.thre_graph_max = max(self.train_dists)
                if not hasattr(self, 'thre_graph_mean'):
                    self.thre_graph_mean = np.mean(self.train_dists)
                if not hasattr(self, 'thre_graph_top80'):
                    self.thre_graph_top80 = sorted_train_dists[int(0.8*len(self.train_dists))]
                if not hasattr(self, 'thre_graph_adapt'):
                    self.thre_graph_adapt = sorted_train_dists[int(normal_graph_rate*len(self.train_dists))]
            else:
                print("Using default threshold 0.5 for graph anomaly detection !!!")
                self.thre_graph_max, self.thre_graph_top80, self.thre_graph_mean, self.thre_graph_adapt = 0.8, 0.8, 0.5, 0.8
            
            # Calculating AUC
            auc_score = cal_auc_score(labels, scores)
            aupr_score = cal_aupr_score(labels, scores)
            graph_auc_score = cal_auc_score(graph_labels, graph_loss)
            graph_aupr_score = cal_aupr_score(graph_labels, graph_loss)
            
            # Threshold
            thre_dict = {
                'top80%': self.thre_top80, 
                'mean': self.thre_mean, 
                'adapt': self.thre_adapt,
            }
            pred_dict = defaultdict(np.array)
            for name, threshold in thre_dict.items():
                acc_score = cal_accuracy(labels, scores, threshold)
                pred_array, cls_report = cal_cls_report(labels, scores, threshold, output_dict=True)
                pred_results = {
                    'AUC': [auc_score], 
                    'AUPR': [aupr_score], 
                    'ACC({})'.format(name): [acc_score],
                }
                stat_df = pd.DataFrame(pred_results)
                cls_df = pd.DataFrame(cls_report).transpose()
                pred_dict[name] = pred_array
                print(stat_df)
                print(cls_df)
                # Save predicting results (regarding each threshold)
                stat_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-results-{name}.csv'))
                cls_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-cls-report-{name}.csv'))

            pred_dict['GT'] = labels
            pred_df = pd.DataFrame(pred_dict)
            pred_df.to_csv(os.path.join(self.checkpoint_dir, f'predictions.csv'))
            
            # Threshold (Graph)
            thre_dict = {
                'top80%': self.thre_graph_top80, 
                'mean': self.thre_graph_mean, 
                'adapt': self.thre_graph_adapt,
            }
            for name, threshold in thre_dict.items():
                # print("graph labels {}, graph loss {}".format(graph_labels, graph_loss))
                acc_score = cal_accuracy(graph_labels, graph_loss, threshold)
                pred_array, cls_report = cal_cls_report(graph_labels, graph_loss, threshold, output_dict=True)
                pred_results = {
                    'AUC': [graph_auc_score], 
                    'AUPR': [graph_aupr_score], 
                    'ACC({})'.format(name): [acc_score],
                }
                stat_df = pd.DataFrame(pred_results)
                cls_df = pd.DataFrame(cls_report).transpose()
                pred_dict[name] = pred_array
                print(stat_df)
                print(cls_df)
                # Save predicting results (regarding each threshold)
                stat_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-results-{name}(graph).csv'))
                cls_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-cls-report-{name}(graph).csv'))

        
    def validation_step(self, batch: Data, batch_idx: int, *args, **kwargs):
        loss_dict = self.training_step(batch, batch_idx, split='val')
        log_dict = loss_dict['log']
        log_dict['val_loss'] = log_dict.pop('train_loss')
        self.log("val_loss", log_dict['val_loss'], batch_size=loss_dict['scores'].size(0))
        return {
            'loss': loss_dict['loss'],
            'scores': loss_dict['scores'],
            'labels': loss_dict['labels'],
            'log': log_dict,
            'progress_bar': log_dict,
            'graph_loss': loss_dict['graph_loss'],
            'graph_labels': loss_dict['graph_labels'],
        }
    
    def validation_epoch_end(self, validation_step_outputs: List[dict]):
        self.training_epoch_end(validation_step_outputs, 'val')
    
    def test_step(self, batch: Data, batch_idx: int):
        loss_dict = self.training_step(batch, batch_idx, split='test')
        log_dict = loss_dict['log']
        log_dict['test_loss'] = log_dict.pop('train_loss')
        self.log("test_loss", log_dict['test_loss'], batch_size=loss_dict['scores'].size(0))
        return {
            'loss': loss_dict['loss'],
            'scores': loss_dict['scores'],
            'labels': loss_dict['labels'],
            'log': log_dict, # Tensorboard logging
            'progress_bar': log_dict, # Progress bar logging for TQDM
            'graph_loss': loss_dict['graph_loss'],
            'graph_labels': loss_dict['graph_labels'],
        }

    def test_epoch_end(self, test_step_outputs: List[dict]):
        self.training_epoch_end(test_step_outputs, 'test')