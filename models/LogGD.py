import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Union, Callable, Tuple, List
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score,
)


class LogGDDataset(Dataset):
    def __init__(
        self, 
        data_file: str,
        ids: List[int] = None,
    ):
        self.df = pd.read_csv(data_file)
        if ids is not None:
            self.df = self.df.iloc[ids]
        self.sent_encoder = SentenceTransformer('bert-base-nli-mean-tokens')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]
        feature = self.sent_encoder.encode([instance['EventTemplate']]).squeeze() # (768)
        label = 0 if instance['Label'] == '-' else 1
        event_id = instance['EventId']  # Return the event_id as well to generate edge_index later
        return feature, label, event_id


class My_collate:
    def __init__(self, batch_size):
        self.last_occurrence = {}  # Keep track of the last occurrence of each event
        self.edge_weights = {}  # Keep track of the edge weights
        self.batch_size = batch_size

    def __call__(self, batch: List[Tuple[Tensor, Tensor, str]]):
        # Pad the batch with dummy nodes if necessary
        while len(batch) < self.batch_size:
            dummy_feature = torch.zeros_like(torch.tensor(batch[0][0]))
            dummy_label = torch.tensor(-1)  # Use -1 or any other value that is not a valid label
            dummy_event_id = ''  # Use an empty string or any other value that is not a valid event id
            batch.append((dummy_feature, dummy_label, dummy_event_id))
        
        features = torch.stack([torch.tensor(item[0]) for item in batch])
        labels = torch.stack([torch.tensor(item[1]) for item in batch])
        # Generate graph_label tensor
        graph_label = 0 if labels.sum() == 0 else 1
        graph_label = torch.tensor(graph_label, dtype=torch.long)
        event_ids = [item[2] for item in batch]

        # Generate edge_index tensor and calculate edge weights
        edge_index, edge_weights = [], []
        for i, event_id in enumerate(event_ids):
            if event_id in self.last_occurrence:
                edge = (self.last_occurrence[event_id], i)
                edge_index.append(edge)
                if edge in self.edge_weights:
                    self.edge_weights[edge] += 1
                else:
                    self.edge_weights[edge] = 1
                edge_weights.append(self.edge_weights[edge])
            self.last_occurrence[event_id] = i
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        return features, labels, edge_index, edge_weights, graph_label
  
    
class LogGD(nn.Module):
    def __init__(self, in_features, out_features, num_classes):
        super(LogGD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_query = nn.Linear(in_features, out_features)
        self.W_key = nn.Linear(in_features, out_features)
        self.W_value = nn.Linear(in_features, out_features)

        self.z_in = nn.Embedding(in_features, out_features)
        self.z_out = nn.Embedding(in_features, out_features)

        self.D = nn.Parameter(torch.randn(out_features))

        # Add a feed-forward network for graph classification
        self.fc1 = nn.Linear(2*out_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, num_classes)
        self.layer_norm = nn.LayerNorm(2*out_features)  # Corrected here
        self.activation = nn.GELU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor):
        num_nodes = x.size(0)
        if edge_index.size(0) == 0:  # Check if edge_index is empty
            return torch.zeros((num_nodes, self.out_features))  # Return a default output

        # Equation 6
        in_degrees = edge_index[0].bincount(minlength=num_nodes)
        out_degrees = edge_index[1].bincount(minlength=num_nodes)
        x0 = x + self.z_in(in_degrees) + self.z_out(out_degrees)

        # Equation 7
        Q = self.W_query(x0[edge_index[0]])  # Calculate Q only for the nodes that have edges
        K = self.W_key(x0[edge_index[1]])  # Calculate K only for the nodes that have edges
        V = self.W_value(x0[edge_index[1]])  # Calculate V only for the nodes that have edges
        w = edge_weights.unsqueeze(-1)  # Add an extra dimension to w to match the dimensions of Q, K, and V
        Q0 = w * Q
        K0 = K
        V0 = w * V

        # Equation 8 and 9
        a = Q0 @ K0.t() / self.out_features**0.5
        b_spatial = Q0 @ self.D + K0 @ self.D
        a = a + b_spatial

        # Equation 10
        a_hat = F.softmax(a, dim=-1)
        z = a_hat @ (V0 + self.D)

        # Graph classification
        z_sum = z.sum(dim=0)
        z_max, _ = z.max(dim=0)
        hg = torch.cat([z_sum, z_max], dim=0)  # Equation 11
        
        hg = self.layer_norm(hg)
        hg = self.activation(self.fc1(hg))
        hg = self.activation(self.fc2(hg))
        out = self.fc3(hg)  # Equation 12
        return out
    
    
class LogGDTrainer:
    def __init__(
        self, 
        model: LogGD, 
        device: torch.device,
        checkpoint_dir: str = '.',
    ):
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.model = model.to(device)
        self.checkpoint_dir = checkpoint_dir
        self.best_f1 = 0

    def train(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        for features, labels, edge_index, edge_weights, graph_label in tqdm(dataloader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_weights = edge_weights.to(self.device)
            graph_label = graph_label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features, edge_index, edge_weights)
            loss = self.criterion(outputs, graph_label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, f1):
        if f1 > self.best_f1:
            self.best_f1 = f1
            ckpt_path = os.path.join(self.checkpoint_dir, f'{f1}.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_f1': self.best_f1,
            }, ckpt_path)
    


class LogGDPredictor:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)

    def predict(self, dataloader):
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for features, labels, edge_index, edge_weights, graph_label in tqdm(dataloader):
                features = features.to(self.device)
                edge_index = edge_index.to(self.device)
                edge_weights = edge_weights.to(self.device)

                outputs = self.model(features, edge_index, edge_weights)
                print(outputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(graph_label.unsqueeze(0))
        all_outputs = torch.stack(all_outputs)
        all_labels = torch.cat(all_labels)
        # print(all_outputs)
        # print(all_labels)
        # Convert the outputs to probabilities using softmax
        probabilities = torch.softmax(all_outputs, dim=-1)[:, 1]  # We take the second column because it's the probability of the positive class
        # print(probabilities)
        # Convert the probabilities and labels to numpy arrays
        probabilities = probabilities.numpy()
        all_labels = all_labels.numpy()

        # Calculate the metrics
        precision = precision_score(all_labels, probabilities.round())
        recall = recall_score(all_labels, probabilities.round())
        f1 = f1_score(all_labels, probabilities.round())
        
        if len(np.unique(all_labels)) == 1:
            print("Only one class present in y_true.")
            if np.abs(all_labels[0] - probabilities.round().mean()) < 1e-1:
                auc = 1.0
                aupr = 1.0
            else: 
                auc = 0.0
                aupr = 0.0
        else:
            auc = roc_auc_score(all_labels, probabilities)
            aupr = average_precision_score(all_labels, probabilities)   

        return precision, recall, f1, auc, aupr


    

if __name__ == '__main__':
    import time
    from sklearn.model_selection import train_test_split
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogGD(768, 768, 2)
    num_epochs = 10
    train_batch_size = 512
    eval_batch_size = 1024
    ckpt_dir = "results/BGL/LogGD"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    df = pd.read_csv('dataset/BGL/BGL.log_structured.csv').sample(frac=0.01, random_state=42)
    train_ids, test_ids = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    train_dataset = LogGDDataset('dataset/BGL/BGL.log_structured.csv', train_ids)
    val_dataset = LogGDDataset('dataset/BGL/BGL.log_structured.csv', val_ids)
    test_dataset = LogGDDataset('dataset/BGL/BGL.log_structured.csv', test_ids)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=My_collate(train_batch_size))
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=My_collate(eval_batch_size))
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=My_collate(eval_batch_size))
    
    trainer = LogGDTrainer(model, device, ckpt_dir)
    predictor = LogGDPredictor(model, device)
    
    # for epoch in range(num_epochs):
    #     loss = trainer.train(train_loader)
    #     print(f"Epoch {epoch+1}, Loss: {loss}")
    #     precision, recall, f1, auc, aupr = predictor.predict(val_loader)
    #     trainer.save_checkpoint(f1)

    start = time.time()
    precision, recall, f1, auc, aupr = predictor.predict(test_loader)
    end = time.time()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-1 Score: {f1}")
    print(f"AUC: {auc}")
    print(f"AUPR: {aupr}")
    
    print(f"Total test time: {end - start} s")
    avg_log_test_time = (end - start) / len(test_ids) if len(test_ids) != 0 else 0
    print(f"Average test time per log: {avg_log_test_time} s")
