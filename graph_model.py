import os
from copy import deepcopy
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import Namespace
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Callable
import torch
from torch.nn import MSELoss
from torch_geometric.utils import degree
import pytorch_lightning as pl

from utils import (
    cal_auc_score, 
    cal_aupr_score, 
    cal_accuracy, 
    cal_cls_report, 
)
from models.graph_base import (
    Data,
    GCNGraphEmbedding,
    SAGEGraphEmbedding,
    GINGraphEmbedding,
    GATGraphEmbedding,
    TransformerGraphEmbedding,
)


class GraphConv(pl.LightningModule):
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
        # Training args
        self.lr = self._get_hparam(hparams, 'lr', 1e-3)
        self.weight_decay = self._get_hparam(hparams, 'weight_decay', 1e-5)
        # Model args
        model_kwargs = self._get_hparam(hparams, 'model_kwargs', dict())
        self.out_channels = model_kwargs.get('output_dim', 128)
        self.layers = model_kwargs.get('layers', 3)
        self.dropout = model_kwargs.get('dropout', 0.1)
        self.model_type = model_kwargs.get('model_type', 'gcn')
        # Models
        if self.model_type.lower() == 'gcn':
            self.model = GCNGraphEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        elif self.model_type.lower() == 'sage':
            self.model = SAGEGraphEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        elif self.model_type.lower() == 'gin':
            self.model = GINGraphEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        elif self.model_type.lower() == 'gat':
            self.model = GATGraphEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        else:
            self.model = TransformerGraphEmbedding(self.dropout, self.in_channels, self.out_channels, self.layers)
        # Loss
        self.mse_loss = MSELoss(reduction='none')
        # Logging
        print('Created {} module \n{} \nwith {:,} GPUs {:,} workers'.format(
            self.model.__class__.__name__, self.model, self.n_gpus, self.n_workers))
        # Save hyperparameters
        self.global_outputs = defaultdict(np.array)
        self.global_labels = defaultdict(np.array)
        self.train_dists = []
        self.train_avg = torch.normal(mean=0, std=1, size=(2*self.out_channels,)) # F
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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, # l2 regularization
        )
        return optimizer

    def training_step(self, G: Data, batch_idx: int, split: str = 'train'): 
        preds = self.forward(
            x=G.x, 
            edge_index=G.edge_index, 
            batch=G.batch,
        )
        # Handling average feature vector
        targets = self.train_avg.expand(preds.shape[0], -1) # B X F
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
                # Update train L2 distances
                self.train_dists.extend(individual_loss.detach().tolist())
            
        logging_dict = {'train_loss': avg_loss.detach().item()}
        return {
            'loss': loss,
            'preds': preds, # For calculating averaged feature vector
            'labels': G.y,
            'log': logging_dict, # Tensorboard logging for training
            'progress_bar': logging_dict, # Progress bar logging for TQDM
        }

    def training_epoch_end(self, train_step_outputs: List[dict], split: str = 'train'):
        preds = [instance['preds'].detach().cpu() for instance in train_step_outputs]
        if split == 'train':
            # Update average train feature vector
            self.train_avg = torch.cat(preds, dim=0).mean(dim=0) 
            # Update train dists and thresholds
            sorted_train_dists = sorted(self.train_dists)
            self.thre_max = max(self.train_dists)
            self.thre_mean = np.mean(self.train_dists)
            self.thre_top80 = sorted_train_dists[int(0.8*len(self.train_dists))]
            self.train_dists = []
            print("Epoch {} train avg (sum) {}, max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                self.current_epoch, self.train_avg.sum(), self.thre_max, self.thre_top80, self.thre_mean,
            ))
        
        preds = torch.cat(preds, dim=0).numpy() # N X F
        labels = [instance['labels'].detach().cpu() for instance in train_step_outputs]
        labels = torch.cat(labels, dim=0).numpy() # N
        self.global_outputs[split] = preds
        self.global_labels[split] = labels

        if split == 'val':
            avg_loss = torch.stack([instance['loss'].detach().cpu() for instance in train_step_outputs]).mean()
            print('Epoch {} avg val_loss: {}'.format(self.current_epoch, avg_loss.detach().item()))
        elif split == 'test':
            print("Test ({} samples) using train avg (sum) {}, max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                len(labels), self.train_avg.sum(), self.thre_max, self.thre_top80, self.thre_mean,
            ))
            loss = [instance['loss'].detach().cpu() for instance in train_step_outputs]
            loss = torch.cat(loss, dim=0).numpy() # N
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
        
    def validation_step(self, G: Data, batch_idx: int, *args, **kwargs):
        loss_dict = self.training_step(G, batch_idx, split='val')
        log_dict = loss_dict['log']
        log_dict['val_loss'] = log_dict.pop('train_loss')
        self.log("val_loss", log_dict['val_loss'], batch_size=loss_dict['preds'].size(0))
        return {
            'loss': loss_dict['loss'],
            'preds': loss_dict['preds'],
            'labels': loss_dict['labels'],
            'log': log_dict, # Tensorboard logging
            'progress_bar': log_dict, # Progress bar logging for TQDM
        }

    def validation_epoch_end(self, validation_step_outputs: List[dict]):
        self.training_epoch_end(validation_step_outputs, 'val')
    
    def test_step(self, G: Data, batch_idx: int):
        loss_dict = self.training_step(G, batch_idx, split='test')
        log_dict = loss_dict['log']
        log_dict['test_loss'] = log_dict.pop('train_loss')
        self.log("test_loss", log_dict['test_loss'], batch_size=loss_dict['preds'].size(0))
        return {
            'loss': loss_dict['loss'],
            'preds': loss_dict['preds'],
            'labels': loss_dict['labels'],
            'log': log_dict, # Tensorboard logging
            'progress_bar': log_dict, # Progress bar logging for TQDM
        }

    def test_epoch_end(self, test_step_outputs: List[dict]):
        self.training_epoch_end(test_step_outputs, 'test')
        
    