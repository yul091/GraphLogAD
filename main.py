import logging
# Suppress tokenization warnings
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
import sys
sys.dont_write_bytecode = True
import torch
import glob
import time
import numpy as np
import pandas as pd
from math import ceil, floor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from graph_model import GraphConv
from node_model import NodeConv, AENodeConv
from edge_model import EdgeDetectionModel
from utils import LABEL2TEMPLATE, EMBED_SIZE
from graph_dataset import HDFSDataset, BGLDataset, BGLNodeDataset, SockShopNodeDataset
from torch_geometric.loader import DataLoader


if __name__ == '__main__':
    import os
    from datetime import datetime
    import argparse
    from argparse import Namespace
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Parser args
    parser = argparse.ArgumentParser(
        description='Training GNN anomaly detector'
    )

    # Logistics
    parser.add_argument(
        '--root', '-r', type=str, 
        default='dataset/HDFS', 
        help='Path to the root dir where the processed torch_geometric.data.Dataset is generated')
    parser.add_argument(
        '--checkpoint_dir', '-c', type=str, 
        default='results/hdfs/few-shot/jumpingknowledge', 
        help='Path to the checkpoint folder')
    parser.add_argument(
        '--fast_exp', action='store_true', 
        default=False, 
        help='Run in fast mode')
    parser.add_argument(
        '--graph_output_dir', type=str, 
        default='GNN-results-visualization', 
        help='Path to the directory for saving visualization results')
    parser.add_argument(
        '--classification', type=str, 
        default='graph',
        choices=['graph', 'node', 'edge'], 
        help='Graph classification or node classification for anomaly detection')
    parser.add_argument(
        '--multi_granularity', action='store_true',
        help='Whether to use multi-granularity model')
    parser.add_argument(
        '--global_weight', type=float,
        default=0.5,
        help='Weight for global graph embedding')
    parser.add_argument(
        '--pretrained_model_path', type=str, 
        default='facebook/bart-base',
        choices=[
            'facebook/bart-base',
            'xlnet-base-cased',
            'gpt2',
            'bert-base-uncased',
            'bert-base-cased',
        ],
        help='pre-trained language model name or path')
    parser.add_argument(
        '--max_length', type=int, 
        default=1024,
        help='max sequence length for transformers sequence to sequence model')
    parser.add_argument(
        '--visualization', action='store_true', 
        default=False,
        help='Whether to visualize predicted embedding distributions')

    # Training args
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--test', '-T', action='store_true', default=False)
    parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs to run')
    parser.add_argument('--no_early_stopping', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 regularization')
    parser.add_argument('--from_scratch', action='store_true', default=False)
    parser.add_argument('--event_only', action='store_true', default=False)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)

    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=768)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda_seq', type=float, default=0.1)
    parser.add_argument('--model_type', type=str, default='gcn', 
                        choices=[
                            'gcn', 
                            'sage', 
                            'gin', 
                            'gat', 
                            'transformer', 
                            'ae-dominant', 
                            'ae-anomalydae', 
                            'ae-conad', 
                            'ae-gcnae', 
                            'ae-mlpae', 
                            'ae-scan', 
                            'ae-dynamic',
                            'deeptralog',
                            'addgraph',
                            'dynamic',
                        ], 
                        help='graph convolutional model')
    
    args = parser.parse_args()


    # Arguments
    checkpoint_dir = args.checkpoint_dir
    root = args.root # root dir
    is_fast_exp = args.fast_exp
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    seed = args.seed
    n_workers = args.num_workers
    is_test_run = args.test
    max_epochs = args.max_epochs
    no_early_stopping = args.no_early_stopping
    lr = args.lr
    weight_decay = args.weight_decay
    train_from_scratch = args.from_scratch
    event_only = args.event_only
    classification = args.classification
    pretrained_model_path = args.pretrained_model_path
    max_length = args.max_length
    graph_output_dir = args.graph_output_dir
    do_train = args.do_train
    do_eval = args.do_eval
    visualization = args.visualization
    multi_granularity = args.multi_granularity
    global_weight = args.global_weight

    print('!!!!!!!!!!! Graph Neural Network MODEL !!!!!!!!!!!')
    
    # Reproductibility
    seed_everything(seed=seed, workers=True)
    # set_seed(seed=seed)
    if is_test_run:
        print('TEST RUN - setting  `fast_dev_dun`')

    # Make the model output / checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define tag_to_id dict
    tag2id = {ent:i for i, ent in enumerate(LABEL2TEMPLATE.keys())}
    tag2id['event'] = len(tag2id)
    tag2id['component'] = len(tag2id)
    # tag2id['device'] = len(tag2id) # for AIT dataset

    # Define model args
    # in_channels = EMBED_SIZE + len(tag2id)
    # in_channels = EMBED_SIZE
    in_channels = 768 # for sentence transformers 
    
    model_kwargs = {
        'layers': args.layers,
        'dropout': args.dropout,
        'output_dim': args.output_dim,
        'model_type': args.model_type,
        'alpha': args.alpha,
        'lambda_seq': args.lambda_seq,
    }

    # Construct the system
    hparams = Namespace(
        df=pd.DataFrame([]),
        n_workers=n_workers,
        n_gpus=1,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        model_kwargs=model_kwargs,
        lr=lr,
        weight_decay=weight_decay,
        feature_dim=in_channels,
        tag2id=tag2id,
        checkpoint_dir=checkpoint_dir,
        event_only=event_only,
        pretrained_model_path=pretrained_model_path,
        max_length=max_length,
        global_weight=global_weight,
        multi_granularity=multi_granularity,
    )

    # Get train and test datasets
    print("Getting datasets ...")
    if 'HDFS' in root:
        graph_data = HDFSDataset(root, hparams=hparams)
        
    elif 'BGL' in root or 'AIT' in root:
        if classification == 'graph':
            graph_data = BGLDataset(root, hparams=hparams)
        else:
            graph_data = BGLNodeDataset(root, hparams=hparams)
            
    elif 'sockshop' in root:
        graph_data = SockShopNodeDataset(root, hparams=hparams)
        print(graph_data.graph_stats)
    else:
        raise ValueError()

    if classification == 'graph':
        all_labels = graph_data.graph_stats['label']
    else: # edge/node
        all_labels = np.array([0 if sum(x) == 0 else 1 for x in graph_data.graph_stats['label']])
        all_logs = sum(len(x) for x in graph_data.graph_stats['label'])
        all_anomalies = sum(sum(x) for x in graph_data.graph_stats['label'])
        print("Total relations: {}, total anomalous relations: {}".format(all_logs, all_anomalies))
        
    anomaly_size = sum(all_labels)
    print("Total graphs: {}, anomaly graphs: {}".format(len(all_labels), anomaly_size))
    normal_size = len(all_labels) - anomaly_size

    n_train = floor(normal_size*0.8) 
    val_size = ceil(n_train*0.2)
    train_size = floor(n_train*0.8)
    test_size = len(graph_data) - train_size - val_size
    test_anomaly_rate = anomaly_size/test_size

    train_graph_data = graph_data[:train_size]
    val_graph_data = graph_data[train_size:train_size + val_size]
    test_graph_data = graph_data[train_size + val_size:]
    print("Train data size: {}, validation data size: {}, test data size: {}, test anomaly ratio: {:.4f}".format(
        len(train_graph_data), len(val_graph_data), len(test_graph_data), test_anomaly_rate))

    # Instantiate train & test dataloaders
    if args.model_type == 'ae-dynamic':
        print("NO SHUFFLING FOR TRAIN DATA LOADER !!!")
        train_loader = DataLoader(
            train_graph_data, 
            batch_size=train_batch_size, 
            shuffle=False, # DO NOT SHUFFLE BECAUSE SEQUENCE ORDER IS IMPORTANT!
            num_workers=n_workers,
        )
    else:
        train_loader = DataLoader(
            train_graph_data, 
            batch_size=train_batch_size, 
            shuffle=True, 
            num_workers=n_workers,
        )
        
    val_loader = DataLoader(
        val_graph_data, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=n_workers,
    )
    test_loader = DataLoader(
        test_graph_data, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=n_workers,
    )

    if args.model_type in ['ae-dynamic', 'ae-anomalydae']:
        # Add number of nodes into hyperparameters
        hparams = Namespace(
            df=pd.DataFrame([]),
            n_workers=n_workers,
            n_gpus=1,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            model_kwargs=model_kwargs,
            lr=lr,
            weight_decay=weight_decay,
            feature_dim=in_channels,
            tag2id=tag2id,
            checkpoint_dir=checkpoint_dir,
            event_only=event_only,
            pretrained_model_path=pretrained_model_path,
            max_length=max_length,
            num_nodes=graph_data.num_nodes,
            multi_granularity=multi_granularity,
            global_weight=global_weight,
        )

    # ##############################################################################
    # #                                Grad Search                                 #
    # ##############################################################################

    # # Wrap the trainer call in a function for multiple training runs
    # def train_gnn(config, num_epochs=10, num_gpus=1):
    #     if 'HDFS' in root:
    #         model = GraphConv(config)
    #     elif 'BGL' in root:
    #         if classification =='graph':
    #             model = GraphConv(config)
    #         else:
    #             if args.model_type.startswith('ae'):
    #                 model = AENodeConv(config)
    #             else:
    #                 model = NodeConv(config)

    #     metrics = {"loss": "ptl/val_loss"}

    #     # Most basic trainer, uses good defaults
    #     checkpoint_callback = ModelCheckpoint(
    #         monitor='val_loss',
    #         dirpath=checkpoint_dir,
    #         save_top_k=20,
    #     )

    #     # progress bar refresh rate is overridden by this
    #     refresh_callback = TQDMProgressBar(refresh_rate=1)

    #     trainer = Trainer(
    #         max_epochs=num_epochs,
    #         gpus=num_gpus,
    #         accumulate_grad_batches=1,
    #         default_root_dir=checkpoint_dir,
    #         gradient_clip_val=1.0,
    #         logger=False,
    #         callbacks=[TuneReportCallback(metrics, on="validation_end"), checkpoint_callback, refresh_callback],
    #         enable_progress_bar=True,
    #     )
    #     trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # # Define search space
    # config = {
    #     # 'lr': tune.loguniform(1e-4, 1e-2),
    #     # 'layers': tune.choice([2, 3, 4, 5, 6]),
    #     # 'dropout': tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    #     # 'output_dim': tune.choice([64, 128, 256, 512, 1024]),
    #     # 'alpha': tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
    #     'theta': tune.choice([10.0, 40.0, 90.0]),
    #     # 'eta': tune.choice([3.0, 5.0, 8.0]),
    # }

    # trainable = tune.with_parameters(
    #     train_gnn,
    #     num_epochs=10, 
    #     num_gpus=1,
    # )

    # analysis = tune.run(
    #     trainable,
    #     resources_per_trial={'gpu': 1},
    #     metric='loss',
    #     mode='min',
    #     config=config,
    #     name='tune_bgl_node',
    # )

    # print(analysis.best_config)
    # print(analysis.best_result)

    ##############################################################################
    #                                  TRAINING                                  #
    ##############################################################################
    
    if classification =='graph':
        model = GraphConv(hparams)
    elif classification == 'node':
        if args.model_type.startswith('ae'):
            model = AENodeConv(hparams)
        else:
            model = NodeConv(hparams)
    else: # edge
        model = EdgeDetectionModel(hparams)

    print('View tensorboard logs by running\ntensorboard --logdir {} and going to http://localhost:6006 on your browser'.format(checkpoint_dir))

    # Load latest checkpoint if exists and specified!
    latest_checkpoint = None
    if train_from_scratch:
        print('!! TRAINING FROM SCRATCH !!')
    elif os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print('!! RESUMING FROM CHECKPOINT: {} !!'.format(latest_checkpoint))
            # Load checkpoint
            model = model.load_from_checkpoint(latest_checkpoint)
    else:
        print('!! TRAINING FROM SCRATCH !!')

    # Most basic trainer, uses good defaults
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        save_top_k=20,
    )

    # Earlystop default is overridden by this
    early_stop_callback = False if no_early_stopping else \
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=4,
            verbose=False,
            mode='min',
        )

    # progress bar refresh rate is overridden by this
    refresh_callback = TQDMProgressBar(refresh_rate=1)

    # Training checkpoints
    trainer = Trainer(
        gpus=1,
        accumulate_grad_batches=1,
        default_root_dir=checkpoint_dir,
        gradient_clip_val=1.0,
        logger=False,
        fast_dev_run=is_test_run,
        min_epochs=min(1, max_epochs),
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback, refresh_callback],
        enable_progress_bar=True,
        # deterministic=True, # reproductibility
    )
    start = datetime.now()
    print('{} - Created {} object'.format(
        datetime.now() - start, trainer.__class__.__name__
    ))
    
    # # Search hyperparameters
    # lr_finder = trainer.tuner.lr_find(
    #     model, 
    #     train_dataloaders=train_loader, 
    #     val_dataloaders=val_loader,
    #     max_lr=1e-2,
    #     min_lr=1e-7,
    # )
    # new_lr = lr_finder.suggestion() # pick point based on plot, or get suggestion
    # print('best learning rate: {}'.format(new_lr))
    # model.hparams.lr = new_lr # update hparams of the model

    # Fitting the model
    if args.model_type != 'ae-scan' and do_train:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print('{} - Fitted model {}'.format(
            datetime.now() - start, os.path.basename(checkpoint_dir)
        ))

    # Test the model
    if do_eval:
        start = time.time()
        trainer.test(model, dataloaders=test_loader) # same as trainer.test(ckpt_path="best")
        end = time.time()
        total_test_time = end - start
        # Calculate the average time per instance
        num_test_graphs = len(test_graph_data)
        num_test_logs = sum(len(x) for x in test_graph_data.graph_stats['label'])
        avg_time_test_graph = total_test_time / num_test_graphs if num_test_graphs else 0
        avg_time_test_log = total_test_time / num_test_logs if num_test_logs else 0
        efficiency_file = os.path.join(model.checkpoint_dir, f'efficiency_analysis.txt')
        with open(efficiency_file, 'w') as f:
            f.write('Total testing time: {:.2f} s, total # graphs: {}, total # logs: {}\n'.format(
                total_test_time, num_test_graphs, num_test_logs))
            f.write('Average time per testing graph: {:.6f} s\n'.format(avg_time_test_graph))
            f.write('Average time per testing log: {:.6f} s\n'.format(avg_time_test_log))

    # Visualization (Train + Val + Test)
    if visualization:
        print("Visualizing embedding distributions ...")
        if not os.path.exists(graph_output_dir):
            os.makedirs(graph_output_dir)

        all_preds = torch.cat([torch.FloatTensor(model.global_outputs[x]) for x in ['train', 'val', 'test']], dim=0) # |V| * H
        all_preds = torch.cat([all_preds, model.train_avg.unsqueeze(0)], dim=0).numpy()
        all_labels = np.array(['train']*len(model.global_labels['train']) + ['val']*len(model.global_labels['val']) + model.global_labels['test'].tolist() + ['train_avg']) # |V|

        embeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_preds) # |V| X 2
        cdict = {'0': 'blue', '1': 'red', 'train': 'green', 'val': 'black', 'train_avg': 'yellow'}
        markers = {'0': '.', '1': '+', 'train': 'x', 'val': '^', 'train_avg':'1'}
        fig, ax = plt.subplots(figsize=(10,10))
        for g in np.unique(all_labels):
            ix = np.where(all_labels == g)
            ax.scatter(embeds[:,0][ix], embeds[:,1][ix], c = cdict[g], marker=markers[g], label = g, s = 100)
        ax.legend()
        plt.savefig(os.path.join(graph_output_dir, f'visualization-all-{start}.png'))


        # Visualization (Train/Validation)
        all_preds = torch.cat([torch.FloatTensor(model.global_outputs[x]) for x in ['train', 'val']], dim=0) # |V| * H
        all_preds = torch.cat([all_preds, model.train_avg.unsqueeze(0)], dim=0).numpy()
        all_labels = np.array(['train']*len(model.global_labels['train']) + ['val']*len(model.global_labels['val']) + ['train_avg']) # |V|

        embeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_preds) # |V| X 2
        cdict = {'train': 'green', 'val': 'black', 'train_avg': 'yellow'}
        markers = {'train': 'x', 'val': '.', 'train_avg':'1'}
        fig, ax = plt.subplots(figsize=(10,10))
        for g in np.unique(all_labels):
            ix = np.where(all_labels == g)
            ax.scatter(embeds[:,0][ix], embeds[:,1][ix], c = cdict[g], marker=markers[g], label = g, s = 100)
        ax.legend()
        plt.savefig(os.path.join(graph_output_dir, f'visualization-train-val-{start}.png'))


        # Visualization (Test)
        all_preds = torch.cat([torch.tensor(model.global_outputs['test']), model.train_avg.unsqueeze(0)], dim=0).numpy() # |V| X H
        all_labels = np.array(model.global_labels['test'].tolist() + ['train_avg']) # |V|


        embeds = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_preds) # |V| X 2
        cdict = {'0': 'blue', '1': 'brown', 'train_avg': 'yellow'}
        markers = {'0': '+', '1': '.', 'train_avg':'1'}
        fig, ax = plt.subplots(figsize=(10,10))
        for g in np.unique(all_labels):
            ix = np.where(all_labels == g)
            ax.scatter(embeds[:,0][ix], embeds[:,1][ix], c = cdict[g], marker=markers[g], label = g, s = 100)
        ax.legend()
        plt.savefig(os.path.join(graph_output_dir, f'visualization-test-{start}.png'))

        print("Visualization finished!!!")