import os
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from argparse import Namespace
from collections import defaultdict
import torch
from torch.nn import MSELoss
import torch.nn.functional as F
from typing import Dict, List
import pytorch_lightning as pl

from utils import (
    EMBED_SIZE,
    cal_auc_score, 
    cal_aupr_score, 
    cal_accuracy, 
    cal_cls_report, 
    classification_report,
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
    GCNNodeEmbedding,
    SAGENodeEmbedding,
    GINNodeEmbedding,
    GATNodeEmbedding,
    TransformerNodeEmbedding,
)
from models.DOMINANT import DOMINANT_Base
from models.CONAD import CONAD_Base
from models.Anomaly_DAE import AnomalyDAE_Base
from models.SCAN import SCAN
from models.Dynamic_AE import DynamicEncoderDecoder



class NodeConv(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.args = hparams
        self.in_channels = self._get_hparam(hparams, 'feature_dim')
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
        # Model args
        model_kwargs = self._get_hparam(hparams, 'model_kwargs', dict())
        self.out_channels = model_kwargs.get('output_dim', 128)
        self.layers = model_kwargs.get('layers', 3)
        self.dropout = model_kwargs.get('dropout', 0.1)
        self.model_type = model_kwargs.get('model_type', 'gcn')
        # Obtain event feature column index (for node classification)
        tag2id = self._get_hparam(hparams, 'tag2id', dict())
        self.event_id = EMBED_SIZE + tag2id['event']
        # Models
        if self.model_type.lower() == 'gcn':
            self.model = GCNNodeEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        elif self.model_type.lower() == 'sage':
            self.model = SAGENodeEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        elif self.model_type.lower() == 'gin':
            self.model = GINNodeEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        elif self.model_type.lower() == 'gat':
            self.model = GATNodeEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        else:
            self.model = TransformerNodeEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        # Loss
        self.mse_loss = MSELoss(reduction='none')
        # Logging
        print('Created {} module \n{} \nwith {:,} GPUs {:,} workers'.format(
            self.model.__class__.__name__, self.model, self.n_gpus, self.n_workers))
        # Save hyperparameters
        self.global_outputs = defaultdict(np.array)
        self.global_labels = defaultdict(np.array)
        if self.event_only:
            self.train_dists = defaultdict(list)
        else: 
            self.train_dists = []
        self.save_hyperparameters()
        
    @property
    def on_cuda(self):
        return next(self.parameters()).is_cuda

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, # l2 regularization
        )
        return optimizer

    def training_step(self, G: Data, batch_idx: int, split='train'): 
        preds = self.forward(
            x=G.x, 
            edge_index=G.edge_index, 
            batch=G.batch,
        )
        if self.event_only:
            # Handling labels and preds
            is_event_nodes = G.x[:, self.event_id] == 1
            # Extract event node embeddings: B*|V| X H -> B*|V'| X H
            x = G.x[is_event_nodes]
            x_list = x.tolist()
            preds = preds[is_event_nodes]
            labels = G.y[is_event_nodes]
        else:
            x = G.x
            labels = G.y

        # Handling average feature vector
        if hasattr(self, 'train_avg'):
            if self.event_only:
                # Used when validation/test contains events which are not in the training data
                event_avg_pool = torch.stack(list(self.train_avg.values()), dim=0).detach().mean(dim=0)
                targets = [self.train_avg.get(tuple(x), event_avg_pool) for x in x_list]
                targets = torch.stack(targets, dim=0) # B X F
            else:
                targets = self.train_avg.expand(preds.shape[0], -1) # B X F
        else:
            if self.event_only:
                targets = [torch.normal(mean=0, std=1, size=(2*self.out_channels,)) for _ in x_list]
                targets = torch.stack(targets, dim=0) # B X F
            else:
                targets = torch.normal(mean=0, std=1, size=(2*self.out_channels,)).expand(preds.shape[0], -1) # B X F
                
        if self.on_cuda:
            targets = targets.cuda()
        
        # Calculate loss and save to dict
        individual_loss = self.mse_loss(preds, targets).sum(dim=-1) # B
        avg_loss = individual_loss.mean() # float
        
        if split == 'test':
            loss = individual_loss # B
        else:
            loss = avg_loss # float
            if split == 'train':
                if self.event_only:
                    # Update train L2 distances
                    for i, x_row in enumerate(x_list):
                        self.train_dists[tuple(x_row)].append(individual_loss[i].detach().item())
                else:
                    # Update train L2 distances
                        self.train_dists.extend(individual_loss.detach().tolist())
            
        logging_dict = {'train_loss': avg_loss.detach().item()}
        return {
            'loss': loss,
            'x': x,
            'preds': preds, # For calculating averaged feature vector
            'labels': labels,
            'log': logging_dict, # Tensorboard logging for training
            'progress_bar': logging_dict, # Progress bar logging for TQDM
        }

    def training_epoch_end(self, train_step_outputs: List[dict], split: str = 'train'):
        print("Processing {} data outputs ...".format(split))
        preds = torch.cat([instance['preds'].detach().cpu() for instance in train_step_outputs], dim=0) # N X F
        labels = torch.cat([instance['labels'].detach().cpu() for instance in train_step_outputs], dim=0).numpy() # N
        x_list = torch.cat([instance['x'].detach().cpu() for instance in train_step_outputs], dim=0).tolist()
        self.global_outputs[split] = preds.numpy() # N X F
        self.global_labels[split] = labels # N
        
        if split == 'train':
            if self.event_only:
                # Update average train feature vector
                event2preds = defaultdict(list)
                for x, preds in zip(x_list, preds):
                    event2preds[tuple(x)].append(preds)
                self.train_avg = {k: torch.stack(preds, dim=0).mean(dim=0) for k, preds in event2preds.items()}
                # Update train dists and thresholds
                self.thre_max, self.thre_mean, self.thre_top80 = {}, {}, {}
                for x_tuple, dists in self.train_dists.items():
                    sorted_dists = sorted(dists)
                    self.thre_max[x_tuple] = max(dists)
                    self.thre_mean[x_tuple] = np.mean(dists)
                    self.thre_top80[x_tuple] = sorted_dists[int(0.8*len(dists))]
                self.train_dists = defaultdict(list)
                print("Epoch {} #events {}, max thre (avg event) {:.4f}, 80% thre (avg event) {:.4f}, mean thre (avg event) {:.4f}".format(
                    self.current_epoch, 
                    len(self.thre_top80),
                    np.mean(list(self.thre_max.values())), 
                    np.mean(list(self.thre_top80.values())), 
                    np.mean(list(self.thre_mean.values())),
                ))
            else:
                # Update average train feature vector
                self.train_avg = preds.mean(dim=0) # F
                # Update train dists and thresholds
                sorted_train_dists = sorted(self.train_dists)
                self.thre_max = max(self.train_dists)
                self.thre_mean = np.mean(self.train_dists)
                self.thre_top80 = sorted_train_dists[int(0.8*len(self.train_dists))]
                self.train_dists = []
                print("Epoch {} train avg (sum) {}, max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                    self.current_epoch, self.train_avg.sum(), self.thre_max, self.thre_top80, self.thre_mean,
                ))
        elif split == 'val':
            avg_loss = sum(ins['loss'].detach().item()*ins['preds'].shape[0] for ins in train_step_outputs)/preds.shape[0] \
                if preds.shape[0] else 0
            print('Epoch {} avg val_loss: {}'.format(self.current_epoch, avg_loss))
        else:
            loss = torch.cat([instance['loss'].detach().cpu() for instance in train_step_outputs], dim=0).numpy() # N
            # Calculating AUC
            auc_score = cal_auc_score(labels, loss)
            aupr_score = cal_aupr_score(labels, loss)
            # Threshold
            thre_dict = {
                'top80%': self.thre_top80, 
                'mean': self.thre_mean, 
                # 'max': self.thre_max,
            }
            pred_dict = defaultdict(np.array)
            if self.event_only:
                print("Test ({} samples) using max thre (avg event) {:.4f}, 80% thre (avg event) {:.4f}, mean thre (avg event) {:.4f}".format(
                    labels.shape[0],
                    np.mean(list(self.thre_max.values())), 
                    np.mean(list(self.thre_top80.values())), 
                    np.mean(list(self.thre_mean.values())),
                ))
                for name, threshold_dict in thre_dict.items():
                    # Handling acc calculation for each event
                    thre_event_avg = np.mean(list(threshold_dict.values()))
                    TP = 0
                    pred_array = []
                    for idx, x in enumerate(x_list):
                        threshold = threshold_dict.get(tuple(x), thre_event_avg)
                        TP += int(loss[idx] > threshold) == labels[idx]
                        pred_array.append(int(loss[idx] > threshold))
                    
                    pred_array = np.array(pred_array)
                    acc_score = TP/len(labels) if len(labels) else 0
                    cls_report = classification_report(labels, pred_array, output_dict=True)
                    pred_results = {'AUC': [auc_score], 'AUPR': [aupr_score], 'ACC({})'.format(name): [acc_score]}
                    stat_df = pd.DataFrame(pred_results)
                    cls_df = pd.DataFrame(cls_report).transpose()
                    pred_dict[name] = pred_array
                    print(stat_df)
                    print(cls_df)
                    # Save predicting results (regarding each threshold)
                    stat_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-results-{name}.csv'))
                    cls_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-cls-report-{name}.csv'))
            else:
                print("Test ({} samples) using train avg (sum) {}, max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                    labels.shape[0], self.train_avg.sum(), self.thre_max, self.thre_top80, self.thre_mean,
                ))
                for name, threshold in thre_dict.items():
                    acc_score = cal_accuracy(labels, loss, threshold)
                    pred_array, cls_report = cal_cls_report(labels, loss, threshold, output_dict=True)
                    pred_results = {'AUC': [auc_score], 'AUPR': [aupr_score], 'ACC({})'.format(name): [acc_score]}
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

        
    def validation_step(self, batch: Data, batch_idx: int, *args, **kwargs):
        loss_dict = self.training_step(batch, batch_idx, split='val')
        log_dict = loss_dict['log']
        log_dict['val_loss'] = log_dict.pop('train_loss')
        self.log("val_loss", log_dict['val_loss'], batch_size=loss_dict['preds'].size(0))
        return {
            'loss': loss_dict['loss'],
            'preds': loss_dict['preds'],
            'x': loss_dict['x'],
            'labels': loss_dict['labels'],
            'log': log_dict, # Tensorboard logging
            'progress_bar': log_dict, # Progress bar logging for TQDM
        }

    def validation_epoch_end(self, validation_step_outputs: List[Dict]):
        self.training_epoch_end(validation_step_outputs, 'val')
    
    def test_step(self, batch: Data, batch_idx: int):
        loss_dict = self.training_step(batch, batch_idx, split='test')
        log_dict = loss_dict['log']
        log_dict['test_loss'] = log_dict.pop('train_loss')
        self.log("test_loss", log_dict['test_loss'], batch_size=loss_dict['preds'].size(0))
        return {
            'loss': loss_dict['loss'],
            'preds': loss_dict['preds'],
            'x': loss_dict['x'],
            'labels': loss_dict['labels'],
            'log': log_dict, # Tensorboard logging
            'progress_bar': log_dict, # Progress bar logging for TQDM
        }

    def test_epoch_end(self, test_step_outputs: List[Dict]):
        self.training_epoch_end(test_step_outputs, 'test')
        

        
        
class AENodeConv(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.args = hparams
        self.in_channels = self._get_hparam(hparams, 'feature_dim')
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
        # Model args
        model_kwargs = self._get_hparam(hparams, 'model_kwargs', dict())
        self.out_channels = model_kwargs.get('output_dim', 128)
        self.layers = model_kwargs.get('layers', 4)
        self.dropout = model_kwargs.get('dropout', 0.3)
        self.model_type = model_kwargs.get('model_type', 'ae-dominant')
        self.alpha = model_kwargs.get('alpha', 0.5)
        self.act = model_kwargs.get('act', F.relu)
        # Obtain event feature column index (for node classification)
        tag2id = self._get_hparam(hparams, 'tag2id', dict())
        self.event_id = EMBED_SIZE + tag2id['event']
        # Models
        if self.model_type.lower() == 'ae-dominant':
            self.model = DOMINANT_Base(
                in_dim=self.in_channels, 
                hid_dim=self.out_channels, 
                num_layers=self.layers, 
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type.lower() == 'ae-anomalydae':
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
        elif self.model_type.lower() == 'ae-conad':
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
        elif self.model_type.lower() == 'ae-gcnae':
            self.model = GCN(
                in_channels=self.in_channels,
                hidden_channels=self.out_channels,
                out_channels=self.in_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type.lower() == 'ae-mlpae':
            self.model = MLP(
                in_channels=self.in_channels,
                hidden_channels=self.out_channels,
                out_channels=self.in_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type.lower() == 'ae-scan':
            self.eps = model_kwargs.get('eps', 0.5)
            self.mu = model_kwargs.get('mu', 2)
            self.contamination = model_kwargs.get('contamination', 0.1)
            self.model = SCAN(
                eps=self.eps, 
                mu=self.mu, 
                contamination=self.contamination,
            )
        else:
            model_path = self._get_hparam(hparams, 'pretrained_model_path', 'facebook/bart-base')
            self.num_nodes = self._get_hparam(hparams, 'num_nodes')
            self.lambda_seq = model_kwargs.get('lambda_seq', 1.0)
            self.model = DynamicEncoderDecoder(
                model_path=model_path,
                in_channels=self.in_channels,
                num_nodes=self.num_nodes,
                out_channels=self.out_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
                use_seq_loss=True if self.lambda_seq != 0 else False,
            )
        # Logging
        print('Created {} module \n{} \nwith {:,} GPUs {:,} workers'.format(
            self.model.__class__.__name__, self.model, self.n_gpus, self.n_workers))
        # Save hyperparameters
        self.decision_scores = []
        self.save_hyperparameters()
        
    @property
    def on_cuda(self):
        return next(self.parameters()).is_cuda

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default: bool = None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default


    def _data_augmentation(self, x: Tensor, adj: Adj):
        """
        Data augmentation on the input graph. Four types of
        pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate
            Structure, high-degree
            Structure, outlying
        
        Parameters
        -----------
        x : note attribute matrix
        adj : dense adjacency matrix

        Returns
        -------
        feat_aug, adj_aug, label_aug : augmented
            attribute matrix, adjacency matrix, and
            pseudo anomaly label to train contrastive
            graph representations
        """
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
        if self.model_type.lower() in ['ae-dominant', 'ae-conad', 'ae-dynamic']:
            # attribute reconstruction loss
            diff_attribute = torch.pow(x - x_, 2)
            attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
            # structure reconstruction loss
            diff_structure = torch.pow(s - s_, 2)
            structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
            score = self.alpha * attribute_errors + (1 - self.alpha) * structure_errors
            return score
        elif self.model_type.lower() == 'ae-anomalydae':
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


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, # l2 regularization
        )
        return optimizer

    def training_step(self, batch: Data, batch_idx: int, split='train'): 
        # Sampling subgraph
        if batch.num_nodes > self.max_length:
            # perm = torch.randperm(batch.num_nodes)
            # idx, _ = perm[:self.max_length].sort()
            # torch.use_deterministic_algorithms(False)
            # G = batch.subgraph(idx) # deterministic bug
            # torch.use_deterministic_algorithms(True)
            G = self._sample_nodes(batch)
        else:
            G = batch

        x = G.x
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
            self.alpha = torch.std(G.s).detach() / (torch.std(x).detach() + torch.std(G.s).detach())

        # Forward pass
        if self.model_type.lower() == 'ae-dominant':
            x_, s_ = self.forward(
                x=x,
                edge_index=G.edge_index,
            )
        elif self.model_type.lower() == 'ae-anomalydae':
            x_, s_ = self.forward(
                x=x,
                edge_index=G.edge_index,
                batch_size=G.num_nodes,
            )
        elif self.model_type.lower() == 'ae-conad':
            x_aug, edge_index_aug, label_aug = self._data_augmentation(x, G.s)
            h_aug = self.model.embed(x_aug, edge_index_aug)
            h = self.model.embed(x, G.edge_index)

            margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
            margin_loss = torch.mean(margin_loss)
            x_, s_ = self.model.reconstruct(h, G.edge_index)
        elif self.model_type.lower() == 'ae-gcnae':
            x_ = self.forward(
                x=x, 
                edge_index=G.edge_index,
            )
        elif self.model_type.lower() == 'ae-mlpae':
            x_ = self.model(
                x=x,
            )
        elif self.model_type.lower() == 'ae-scan':
            scores = self.model(G)

        else: # ae-dynamic
            outputs = self.forward(
                x=x, 
                edge_index=G.edge_index, 
                batch=G.batch, 
                ids=G.ids, # for seq2seq teacher-forcing loss
                num_graphs=G.num_graphs, # for generating position embedding
            )
            x, x_, s_, lm_loss = outputs

        # Handling scores and loss
        if self.event_only:
            # Handling labels and preds
            if self.model_type.lower() == 'ae-scan':
                is_event_nodes = x[:, self.event_id] == 1
                scores = scores[is_event_nodes]
                labels = G.y[is_event_nodes]
            else:
                # Extract event nodes: B*|V| X H -> B*|V'| X H
                is_event_nodes = x[:, self.event_id] == 1
                event_x_ = x_[is_event_nodes]
                event_x = x[is_event_nodes]
                labels = G.y[is_event_nodes]
                # Calculate loss and save to dict
                if self.model_type.lower() not in ['ae-gcnae', 'ae-mlpae']:
                    event_s = G.s[is_event_nodes][:, is_event_nodes]
                    event_s_ = s_[is_event_nodes][:, is_event_nodes]
                    scores = self.loss_func(event_x, event_x_, event_s, event_s_) # |V|
                else:
                    scores = torch.mean(F.mse_loss(event_x_, event_x, reduction='none'), dim=1)
        else:
            labels = G.y
            # Calculate loss and save to dict
            if self.model_type.lower() not in ['ae-gcnae', 'ae-mlpae', 'ae-scan']: 
                scores = self.loss_func(x, x_, G.s, s_) # |V|
            elif self.model_type.lower() != 'ae-scan':
                scores = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1)

        # Store training score distribution for analysis
        if split == 'train' or self.model_type.lower() == 'ae-scan':
            self.decision_scores.extend(scores.detach().cpu().tolist())

        if self.model_type.lower() == 'ae-conad':
            loss = self.eta * torch.mean(scores) + (1 - self.eta) * margin_loss
        elif self.model_type.lower() == 'ae-dynamic':
            if self.lambda_seq != 0 and split != 'test':
                loss = torch.mean(scores) + self.lambda_seq * lm_loss
            else:
                loss = torch.mean(scores)

            if split == 'test':
                labels = G.y[:scores.shape[0]] # needed when some of the nodes are cut
        else:
            loss = torch.mean(scores)

        logging_dict = {'train_loss': loss.detach().item()}
        return {
            'loss': loss,
            'scores': scores,
            'labels': labels,
            'log': logging_dict, # Tensorboard logging for training
            'progress_bar': logging_dict, # Progress bar logging for TQDM
        }

    def training_epoch_end(self, train_step_outputs: List[dict], split: str = 'train'):
        event_scores = torch.cat([instance['scores'].detach().cpu() for instance in train_step_outputs], dim=0) # N
        scores = event_scores.numpy() # N
        
        if split == 'train':
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
        elif split == 'val':
            avg_loss = sum(ins['loss'].detach().item()*ins['scores'].shape[0] for ins in train_step_outputs)/event_scores.shape[0] \
                if event_scores.shape[0] else 0
            print('Epoch {} avg val_loss: {}'.format(self.current_epoch, avg_loss))
        else:
            event_labels = torch.cat([instance['labels'].detach().cpu() for instance in train_step_outputs], dim=0) # N
            labels = event_labels.numpy() # N
            if not hasattr(self, 'thre_max'):
                if self.decision_scores:
                    sorted_scores = sorted(self.decision_scores)
                    self.thre_max = max(self.decision_scores)
                    self.thre_mean = np.mean(self.decision_scores)
                    self.thre_top80 = sorted_scores[int(0.8*len(self.decision_scores))]
                else:
                    self.thre_max, self.thre_top80, self.thre_mean = 0.5, 0.5, 0.5
            print("Predicting {} test samples, {} ({:.2f}%) anomalies, using max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                len(labels),
                sum(labels),
                sum(labels)*100/len(labels),
                self.thre_max, 
                self.thre_top80, 
                self.thre_mean,
            ))
            loss = torch.cat([instance['scores'].detach().cpu() for instance in train_step_outputs], dim=0).numpy() # N
            # Calculating AUC
            auc_score = cal_auc_score(labels, loss)
            aupr_score = cal_aupr_score(labels, loss)
            # Threshold
            thre_dict = {
                'top80%': self.thre_top80, 
                'mean': self.thre_mean, 
                # 'max': self.thre_max,
            }
            pred_dict = defaultdict(np.array)
            for name, threshold in thre_dict.items():
                acc_score = cal_accuracy(labels, loss, threshold)
                pred_array, cls_report = cal_cls_report(labels, loss, threshold, output_dict=True)
                pred_results = {'AUC': [auc_score], 'AUPR': [aupr_score], 'ACC({})'.format(name): [acc_score]}
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
        }

    def test_epoch_end(self, test_step_outputs: List[dict]):
        self.training_epoch_end(test_step_outputs, 'test')