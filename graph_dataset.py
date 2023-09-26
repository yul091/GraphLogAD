import os
import re
import json
import math
import pickle
import pandas as pd
from tqdm import tqdm
import os.path as osp
from argparse import Namespace
from pyvis.network import Network
from collections import defaultdict
from typing import List
from utils import LABEL2TEMPLATE, SentenceEncoder
import torch
from torch_geometric.data import Data, Dataset
from transformers import (
    # AutoTokenizer,
    # AutoModel,
    logging,
)
from datasets import load_dataset
# Suppress pre-trained model warnings
logging.set_verbosity_warning()
logging.set_verbosity_error()


class HDFSDataset(Dataset):
    """Define HDFS GraphDataset (torch_geometric.data.Dataset)"""
    invalid_entities = set([('ip', '127.0.0.1'), ('pid', '0')]) # filtering invalid entities
    shape_dict = {'event': 'circle', 'component': 'triangle', 'others': 'dot'}

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, hparams=None):
        self.args = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.df = self._get_hparam(hparams, 'df', pd.DataFrame([]))
        self.tag2id = self._get_hparam(hparams, 'tag2id')
        # self.plm = AutoModel.from_pretrained('bert-large-uncased').to(self.device)
        # self.ptk = AutoTokenizer.from_pretrained('bert-large-uncased')
        self.sent_encoder = SentenceEncoder(device=self.device)
        self.using_event_template = self._get_hparam(hparams, 'using_event_template', True)
        self.batch_size = self._get_hparam(hparams, 'batch_size', 16)
        # if self.using_event_template:
        #     print("Uing event template as event embedding inputs !!!")
        # else:
        #     print("Uing event ID (e.g., a4bd6#2o) as event embedding inputs !!!")
        super().__init__(root, transform, pre_transform, pre_filter)
        # Get statistics of the datasets
        # try:
        #     self.graph_stats = load_dataset('json', data_files=self.raw_paths[0])['train']
        # except:
        graph_stats = []
        f = open(self.raw_paths[0], 'r')
        for line in f:
            # print("line ({}): {}".format(type(line), line))
            graph_stats.append(json.loads(line))
        self.graph_stats = pd.DataFrame(graph_stats)
            
        if not hasattr(self, 'node_set'):
            f = open(os.path.join(self.root, 'others', 'node.pickle'), 'rb')
            self.node_set = pickle.load(f)
        
        self.num_nodes = len(self.node_set)

    @property
    def raw_file_names(self):
        return ['graph.json']

    @property
    def processed_file_names(self):
        f = open(self.raw_paths[0], 'r') # ../graph.json
        return [f'graph_{idx}.pt' for idx in range(len(f.readlines()))] 

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default


    def _visualize(self, idx, directed=False, out_dir=None, name=None):
        graph_dict=self.graph_stats[idx] # dict: graphID, nodes, label
        ent2tag = {node[1]:node[0] for node in graph_dict['nodes']}
        ent2id = {ent:i for i, ent in enumerate(ent2tag.keys())}
        label = 'anomaly' if graph_dict['label'] == 1 else 'normal'

        # Define graph
        net = Network('1000px', '2000px', directed=directed)
        for node_pair in graph_dict['nodes']:
            # Add first node
            shape1 = self.shape_dict.get(node_pair[0], self.shape_dict['others'])
            net.add_node(ent2id[node_pair[1]], label=node_pair[1], shape=shape1)

            # Add second node
            shape2 = self.shape_dict.get(node_pair[2], self.shape_dict['others'])
            net.add_node(ent2id[node_pair[3]], label=node_pair[3], shape=shape2)

            # Add edges
            net.add_edge(ent2id[node_pair[1]], ent2id[node_pair[3]])

        if out_dir is not None:
            graph_dir = out_dir
        else:
            graph_dir = os.path.join(self.root, 'graph')
        if not os.path.isdir(graph_dir):
            os.makedirs(graph_dir)

        # Save graph_idx.html
        if name is None:
            net.show(os.path.join(graph_dir, f'graph_{idx}({label}).html'))
        else:
            net.show(os.path.join(graph_dir, name))


    def download(self):
        # Download to `self.raw_dir`.
        graph_list = []
        self.node_set = set()
        self.event2template = defaultdict(str)
        
        for idx, row in tqdm(self.df.iterrows()):
            graph_dict = {'graphID': idx, 'nodes': [], 'label': row.Label}
            graph_node_set = set()
            for j in range(len(row.EventId)):
                # Connect eventID to Component
                component = re.split('([\[\]])', row.Component[j])[0] # remove [XXXX] part
                graph_dict['nodes'] += [('event', row.EventId[j], 'component', component), ('component', component, 'event', row.EventId[j])]
                # Newly added: Time, Pid
                # graph_dict['nodes'] += [('event', row.EventId[j], 'time', str(row.Datetime[j])), ('time', str(row.Datetime[j]), 'event', row.EventId[j])]
                graph_dict['nodes'] += [('event', row.EventId[j], 'pid', str(row.Pid[j])), ('pid', str(row.Pid[j]), 'event', row.EventId[j])]

                # Connect eventID to each entity
                for entity in row.Preds[j]:
                    if tuple(entity) not in self.invalid_entities: # valid entity pair
                        graph_dict['nodes'] += [('event', row.EventId[j], entity[0], entity[1]), (entity[0], entity[1], 'event', row.EventId[j])]
                        graph_node_set.add(entity[1]) # add entity node

                graph_node_set.add(row.EventId[j]) # add event node
                graph_node_set.add(component) # add component node
                graph_node_set.add(str(row.Pid[j])) # add pid node

                # Store event2template
                if row.EventId[j] not in self.event2template:
                    self.event2template[row.EventId[j]] = row.EventTemplate[j]

            # Integrate all graph nodes
            self.node_set.update(graph_node_set) # merge all nodes into node set
            graph_list.append(graph_dict)

        with open(os.path.join(self.raw_dir, 'graph.json'), 'w', encoding='utf-8') as file:
            for dic in graph_list:
                json.dump(dic, file) 
                file.write("\n")

        # Save node set for reuse
        node_dir = os.path.join(self.root, 'others')
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)
        
        f = open(os.path.join(node_dir, 'node.pickle'), 'wb')
        pickle.dump(self.node_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        


    def _get_graph_data(self, graph_dict: dict):
        # Get {ent: tag} dict
        ent_tag_dict = {x[1]:x[0] for x in graph_dict['nodes']} # {entity: tag}

        # Get edge index: 2 X |E|
        ent2id = {ent:i for i,ent in enumerate(ent_tag_dict.keys())} # node index
        edge_index = [[ent2id[x[1]], ent2id[x[3]]] for x in graph_dict['nodes']] 
        edge_index = torch.tensor(edge_index, dtype=torch.long).T 

        # Get feature vectors (sentence embedding): |V| X F
        ids = []
        graph_prompts = []
        for ent, tag in ent_tag_dict.items():
            # tag_embed = F.one_hot(torch.tensor(self.tag2id[tag]), num_classes=len(self.tag2id)).float() # num_tags 
            ids.append(self.node2id[ent]) # assign each node a unique id

            #########################################################
            # Generate template
            if tag == 'event':
                if self.using_event_template:
                    ent = self.event2template[ent]
                    prompt = ent
                else:
                    prompt = ent + ' is an event ID .'

            elif tag == 'component':
                prompt = ent + ' is a log message component .'
            
            else:
                prompt = ent + LABEL2TEMPLATE[tag][0]
            #########################################################
            graph_prompts.append(prompt)

        feature = self._tokenize_and_embed(graph_prompts)
        ids = torch.LongTensor(ids)
        label = torch.LongTensor([graph_dict['label']]) 
        return Data(x=feature, edge_index=edge_index, y=label, ids=ids)


    def _tokenize_and_embed(self, graph_prompts: List[str]):
        # Batch handling
        if len(graph_prompts) > self.batch_size:
            feature = []
            num_batch = math.ceil(len(graph_prompts)/self.batch_size)
            for i in range(num_batch):
                batch_prompts = graph_prompts[i*self.batch_size: min(len(graph_prompts), (i+1)*self.batch_size)]
                # # Tokenize
                # tokenized_inputs = self.ptk(
                #     batch_prompts, 
                #     max_length=1024,
                #     padding=False,
                #     truncation=True,
                # )
                # # Pad dynamically 
                # # ['input_ids', 'token_type_ids', 'attention_mask']
                # batch = self.ptk.pad(
                #     tokenized_inputs,
                #     padding=True,
                #     max_length=1024,
                #     pad_to_multiple_of=8,
                #     return_tensors="pt",
                # ).to(self.device) 
                # # Encode
                # # ['last_hidden_state', 'pooler_output']
                # encode_outputs = self.plm(**batch) 
                # feature.append(encode_outputs.pooler_output.detach().cpu()) # B X H
                encode_outputs = self.sent_encoder.encode(batch_prompts)
                feature.append(encode_outputs.detach().cpu()) # B X H

            feature = torch.cat(feature, dim=0)
        else:
            # # Tokenize
            # tokenized_inputs = self.ptk(
            #     graph_prompts, 
            #     max_length=1024,
            #     padding=False,
            #     truncation=True,
            # )
            # # Pad dynamically 
            # # ['input_ids', 'token_type_ids', 'attention_mask']
            # batch = self.ptk.pad(
            #     tokenized_inputs,
            #     padding=True,
            #     max_length=1024,
            #     pad_to_multiple_of=8,
            #     return_tensors="pt",
            # ).to(self.device)
            # # Encode
            # # ['last_hidden_state', 'pooler_output']
            # encode_outputs = self.plm(**batch)
            # feature = encode_outputs.pooler_output.detach().cpu() # B X H
            encode_outputs = self.sent_encoder.encode(graph_prompts)
            feature = encode_outputs.detach().cpu() # B X H
        
        return feature


    def process(self):
        idx = 0
        if not hasattr(self, 'node_set'):
            f = open(os.path.join(self.root, 'others', 'node.pickle'), 'rb')
            self.node_set = pickle.load(f)

        node_list = list(self.node_set)
        self.node2id = {x: i for i, x in enumerate(node_list)}

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            graph_list = load_dataset('json', data_files=raw_path)['train']

            for graph_dict in tqdm(graph_list):
                data = self._get_graph_data(graph_dict)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'graph_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return data




class BGLDataset(HDFSDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, hparams=None):
        super().__init__(root, transform, pre_transform, pre_filter, hparams)
        
    def download(self):
        # Download to `self.raw_dir`.
        graph_list = []
        self.node_set = set()
        self.event2template = defaultdict(str)

        for idx, row in tqdm(self.df.iterrows()):
            graph_dict = {'graphID': idx, 'nodes': [], 'label': row.Label}
            graph_node_set = set()
            for j in range(len(row.EventId)):
                # Connect eventID to Component
                component = re.split('([\[\]])', row.Component[j])[0] # remove [XXXX] part
                graph_dict['nodes'] += [('event', row.EventId[j], 'component', component), ('component', component, 'event', row.EventId[j])]
                # # Newly added: time
                # graph_dict['nodes'] += [('event', row.EventId[j], 'time', str(row.Datetime[j])), ('time', str(row.Datetime[j]), 'event', row.EventId[j])]
                # Connect eventID to each entity
                for entity in row.Preds[j]:
                    if tuple(entity) not in self.invalid_entities: # valid entity pair
                        graph_dict['nodes'] += [('event', row.EventId[j], entity[0], entity[1]), (entity[0], entity[1], 'event', row.EventId[j])]
                        graph_node_set.add(entity[1]) # add entity node

                graph_node_set.add(row.EventId[j]) # add event node
                graph_node_set.add(component) # add component node

                # Store event2template
                if row.EventId[j] not in self.event2template:
                    self.event2template[row.EventId[j]] = row.EventTemplate[j]
            
            # Integrate all graphs
            self.node_set.update(graph_node_set) # merge all nodes into node set
            graph_list.append(graph_dict)

        with open(os.path.join(self.raw_dir, 'graph.json'), 'w', encoding='utf-8') as file:
            for dic in graph_list:
                json.dump(dic, file) 
                file.write("\n")

        # Save node set for reuse
        node_dir = os.path.join(self.root, 'others')
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)
        
        f = open(os.path.join(node_dir, 'node.pickle'), 'wb')
        pickle.dump(self.node_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    def process(self):
        idx = 0
        if not hasattr(self, 'node_set'):
            f = open(os.path.join(self.root, 'others', 'node.pickle'), 'rb')
            self.node_set = pickle.load(f)

        node_list = list(self.node_set)
        self.node2id = {x: i for i, x in enumerate(node_list)}

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            graph_list = load_dataset('json', data_files=raw_path)['train']

            for graph_dict in tqdm(graph_list):
                data = self._get_graph_data(graph_dict)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'graph_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return data




class BGLNodeDataset(HDFSDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, hparams=None):
        super().__init__(root, transform, pre_transform, pre_filter, hparams)
        
    def _visualize(self, idx, directed=False, out_dir=None,name=None):
        graph_dict=self.graph_stats[idx] # dict: graphID, nodes, label
        ent2tag = {node[1]:node[0] for node in graph_dict['nodes']}
        ent2id = {ent:i for i, ent in enumerate(ent2tag.keys())}
        
        # Define graph
        net = Network('1000px', '2000px', directed=directed)
        for node_pair, y in zip(graph_dict['nodes'], graph_dict['label']):

            # Add first node
            shape1 = self.shape_dict.get(node_pair[0], self.shape_dict['others'])
            if node_pair[0] == 'event' and y == 1:
                net.add_node(ent2id[node_pair[1]], label=node_pair[1], color='red', shape=shape1)
            else:
                net.add_node(ent2id[node_pair[1]], label=node_pair[1], shape=shape1)

            # Add second node
            shape2 = self.shape_dict.get(node_pair[2], self.shape_dict['others'])
            if node_pair[2] == 'event' and y == 1:
                net.add_node(ent2id[node_pair[3]], label=node_pair[3], color='red', shape=shape2)
            else:
                net.add_node(ent2id[node_pair[3]], label=node_pair[3], shape=shape2)

            # Add edges
            net.add_edge(ent2id[node_pair[1]], ent2id[node_pair[3]])

        if out_dir is not None:
            graph_dir = out_dir
        else:
            graph_dir = os.path.join(self.root, 'graph')
        if not os.path.isdir(graph_dir):
            os.makedirs(graph_dir)

        # Save graph_idx.html
        if name is None:
            label = 'anomaly' if sum(graph_dict['label']) > 0 else 'normal'
            net.show(os.path.join(graph_dir, f'graph_{idx}({label}).html'))
        else:
            net.show(os.path.join(graph_dir, name))


    def download(self):
        # Download to `self.raw_dir`.
        graph_list = []
        self.node_set = set()
        self.event2template = defaultdict(str)

        for idx, row in tqdm(self.df.iterrows()):
            graph_dict = {'graphID': idx, 'nodes': [], 'label': []}
            graph_node_set = set()
            for j in range(len(row.EventId)):
                node_list = []
                # Connect eventID to Component
                component = re.split('([\[\]])', row.Component[j])[0] # remove [XXXX] part
                node_list += [('event', row.EventId[j], 'component', component), ('component', component, 'event', row.EventId[j])]
                
                # Connect eventID to each entity
                for entity in row.Preds[j]:
                    if tuple(entity) not in self.invalid_entities: # valid entity pair
                        node_list += [('event', row.EventId[j], entity[0], entity[1]), (entity[0], entity[1], 'event', row.EventId[j])]
                        graph_node_set.add(entity[1]) # add entity node

                graph_node_set.add(row.EventId[j]) # add event node
                graph_node_set.add(component) # add component node

                graph_dict['nodes'] += node_list
                graph_dict['label'] += [row.EventLabels[j]]*len(node_list) # broadcast to all node pairs for each log

                # Store event2template
                if row.EventId[j] not in self.event2template:
                    self.event2template[row.EventId[j]] = row.EventTemplate[j]
            
            # Integrate all graphs
            self.node_set.update(graph_node_set) # merge all nodes into node set
            graph_list.append(graph_dict)

        with open(os.path.join(self.raw_dir, 'graph.json'), 'w', encoding='utf-8') as file:
            for dic in graph_list:
                json.dump(dic, file) 
                file.write("\n")

        # Save node set for reuse
        node_dir = os.path.join(self.root, 'others')
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)
        
        f = open(os.path.join(node_dir, 'node.pickle'), 'wb')
        pickle.dump(self.node_set, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        f = open(os.path.join(node_dir, 'template.pickle'), 'wb')
        pickle.dump(self.event2template, f, protocol=pickle.HIGHEST_PROTOCOL)


    def _get_graph_data(self, graph_dict: dict):
        # Get {ent: tag} dict
        ent_tag_dict = {x[1]:x[0] for x in graph_dict['nodes']} # {entity: tag}

        # Get edge index: 2 X |E|
        ent2id = {ent:i for i,ent in enumerate(ent_tag_dict.keys())} # node index
        edge_index = [[ent2id[x[1]], ent2id[x[3]]] for x in graph_dict['nodes']] 
        edge_index = torch.tensor(edge_index, dtype=torch.long).T 

        # Get feature vectors (sentence embedding): |V| X F
        graph_prompts = []
        ids = []
        for ent, tag in ent_tag_dict.items():
            # tag_embed = F.one_hot(torch.tensor(self.tag2id[tag]), num_classes=len(self.tag2id)).float() # num_tags 
            ids.append(self.node2id[ent]) # assign each node a unique id

            #########################################################
            # Generate template
            if tag == 'event':
                if self.using_event_template:
                    ent = self.event2template[ent]
                    prompt = ent
                else:
                    prompt = ent + ' is an event ID .'

            elif tag == 'component':
                prompt = ent + ' is a log message component .'
            
            else:
                prompt = ent + LABEL2TEMPLATE[tag][0]
            #########################################################
            graph_prompts.append(prompt)

        feature = self._tokenize_and_embed(graph_prompts)

        # Handle label (event special)
        ent2label = defaultdict(int)
        for node, label in zip(graph_dict['nodes'], graph_dict['label']):
            ent2label[node[1]] += label
            ent2label[node[1]] = min(1, ent2label[node[1]])
        
        label = torch.LongTensor(list(ent2label.values()))
        ids = torch.LongTensor(ids)
        edge_label = torch.LongTensor(graph_dict['label']) 
        return Data(x=feature, edge_index=edge_index, y=edge_label, ids=ids, node_label=label)

    def process(self):
        idx = 0
        if not hasattr(self, 'node_set'):
            f = open(os.path.join(self.root, 'others', 'node.pickle'), 'rb')
            self.node_set = pickle.load(f)

        if not hasattr(self, 'event2template'):
            f = open(os.path.join(self.root, 'others', 'template.pickle'), 'rb')
            self.event2template = pickle.load(f)

        node_list = list(self.node_set)
        self.node2id = {x: i for i, x in enumerate(node_list)}
        
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            graph_list = load_dataset('json', data_files=raw_path)['train']

            for graph_dict in tqdm(graph_list):
                data = self._get_graph_data(graph_dict)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'graph_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return data
    
    
    
class SockShopNodeDataset(HDFSDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, hparams=None):
        super().__init__(root, transform, pre_transform, pre_filter, hparams)
        
    def _visualize(self, idx, directed=False, out_dir=None,name=None):
        graph_dict=self.graph_stats[idx] # dict: graphID, nodes, label
        ent2tag = {node[1]:node[0] for node in graph_dict['nodes']}
        ent2id = {ent:i for i, ent in enumerate(ent2tag.keys())}
        
        # Define graph
        net = Network('1000px', '2000px', directed=directed)
        for node_pair, y in zip(graph_dict['nodes'], graph_dict['label']):

            # Add first node
            shape1 = self.shape_dict.get(node_pair[0], self.shape_dict['others'])
            if node_pair[0] == 'event' and y == 1:
                net.add_node(ent2id[node_pair[1]], label=node_pair[1], color='red', shape=shape1)
            else:
                net.add_node(ent2id[node_pair[1]], label=node_pair[1], shape=shape1)

            # Add second node
            shape2 = self.shape_dict.get(node_pair[2], self.shape_dict['others'])
            if node_pair[2] == 'event' and y == 1:
                net.add_node(ent2id[node_pair[3]], label=node_pair[3], color='red', shape=shape2)
            else:
                net.add_node(ent2id[node_pair[3]], label=node_pair[3], shape=shape2)

            # Add edges
            net.add_edge(ent2id[node_pair[1]], ent2id[node_pair[3]])

        if out_dir is not None:
            graph_dir = out_dir
        else:
            graph_dir = os.path.join(self.root, 'graph')
        if not os.path.isdir(graph_dir):
            os.makedirs(graph_dir)

        # Save graph_idx.html
        if name is None:
            label = 'anomaly' if sum(graph_dict['label']) > 0 else 'normal'
            net.show(os.path.join(graph_dir, f'graph_{idx}({label}).html'))
        else:
            net.show(os.path.join(graph_dir, name))


    def download(self):
        # Download to `self.raw_dir`.
        graph_list = []
        self.node_set = set()

        for idx, row in tqdm(self.df.iterrows()):
            graph_dict = {'graphID': idx, 'nodes': [], 'label': []}
            graph_node_set = set()
            for j in range(len(row.EventLabels)):
                node_list = []
                # Connect source to Timestamp
                node_list += [('timestamp', str(row.Timestamp[j]), 'source', row.source[j]), ('source', row.source[j], 'timestamp', str(row.Timestamp[j]))]
                self.node_set.add(row.source[j]) # add source node
                self.node_set.add(row.Timestamp[j]) # add timestamp node
                
                # Connect customer entity to other entities
                user = None
                all_others = []
                for (key, value) in row.Preds[j]:
                    if key == 'customer':
                        user = value
                    else:
                        all_others.append((key, value))
                        
                if user:
                    graph_node_set.add(user) # add customer node
                    for (key, value) in all_others:
                        node_list += [('customer', user, key, value), (key, value, 'customer', user)]
                        graph_node_set.add(value) # add entity node

                graph_dict['nodes'] += node_list
                graph_dict['label'] += [row.EventLabels[j]]*len(node_list) # broadcast to all node pairs for each log
            
            # Integrate all graphs
            self.node_set.update(graph_node_set) # merge all nodes into node set
            graph_list.append(graph_dict)

        with open(os.path.join(self.raw_dir, 'graph.json'), 'w') as file:
            for dic in graph_list:
                json.dump(dic, file) 
                file.write("\n")

        # Save node set for reuse
        node_dir = os.path.join(self.root, 'others')
        if not os.path.exists(node_dir):
            os.makedirs(node_dir)
        
        f = open(os.path.join(node_dir, 'node.pickle'), 'wb')
        pickle.dump(self.node_set, f, protocol=pickle.HIGHEST_PROTOCOL)


    def _get_graph_data(self, graph_dict: dict):
        # Get {ent: tag} dict
        ent_tag_dict = {x[1]:x[0] for x in graph_dict['nodes']} # {entity: tag}

        # Get edge index: 2 X |E|
        ent2id = {ent:i for i,ent in enumerate(ent_tag_dict.keys())} # node index
        edge_index = [[ent2id[x[1]], ent2id[x[3]]] for x in graph_dict['nodes']] 
        edge_index = torch.tensor(edge_index, dtype=torch.long).T 

        # Get feature vectors (sentence embedding): |V| X F
        graph_prompts = []
        ids = []
        for ent, tag in ent_tag_dict.items():
            # tag_embed = F.one_hot(torch.tensor(self.tag2id[tag]), num_classes=len(self.tag2id)).float() # num_tags 
            ids.append(self.node2id[ent]) # assign each node a unique id
            #########################################################
            # Generate template
            if tag in LABEL2TEMPLATE:
                prompt = str(ent) + LABEL2TEMPLATE[tag][0]
            else:
                prompt = str(ent) + ' is a {} entity .'.format(tag)
            #########################################################
            graph_prompts.append(prompt)

        feature = self._tokenize_and_embed(graph_prompts)

        # Handle label (event special)
        ent2label = defaultdict(int)
        for node, label in zip(graph_dict['nodes'], graph_dict['label']):
            ent2label[node[1]] += label
            ent2label[node[1]] = min(1, ent2label[node[1]])
        
        label = torch.LongTensor(list(ent2label.values()))
        ids = torch.LongTensor(ids)
        edge_label = torch.LongTensor(graph_dict['label']) 
        return Data(x=feature, edge_index=edge_index, y=edge_label, ids=ids, node_label=label)

    def process(self):
        idx = 0
        if not hasattr(self, 'node_set'):
            f = open(os.path.join(self.root, 'others', 'node.pickle'), 'rb')
            self.node_set = pickle.load(f)

        node_list = list(self.node_set)
        self.node2id = {x: i for i, x in enumerate(node_list)}
        
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            f = open(raw_path, 'r')
            for line in f:
                # print("line ({}): {}".format(type(line), line))
                graph_dict = json.loads(line)
                data = self._get_graph_data(graph_dict)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'graph_{idx}.pt'))
                idx += 1
            

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
        return data