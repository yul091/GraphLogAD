import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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
import time


class LogFlashDataset(Dataset):
    def __init__(
        self, 
        data_file: str,
        ids: List[int] = None,
    ):
        self.df = pd.read_csv(data_file)
        if ids is not None:
            self.df = self.df.iloc[ids]
        self.event2id = {event: i for i, event in enumerate(self.df['EventId'].unique())}
        self.sent_encoder = SentenceTransformer('bert-base-nli-mean-tokens')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]
        feature = self.sent_encoder.encode([instance['EventTemplate']]).squeeze() # (768)
        label = 0 if instance['Label'] == '-' else 1
        id = self.event2id[instance['EventId']]  # Return the node_id as well to generate edge_index later
        return feature, label, id


class My_collate:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def __call__(self, batch: List[Tuple[Tensor, Tensor, str]]):
        # Pad the batch with dummy nodes if necessary
        while len(batch) < self.batch_size:
            dummy_feature = torch.zeros_like(torch.tensor(batch[0][0]))
            dummy_label = torch.tensor(-1)  # Use -1 or any other value that is not a valid label
            dummy_id = 0  # Use an empty string or any other value that is not a valid event id
            batch.append((dummy_feature, dummy_id, dummy_label))
        
        features = torch.stack([torch.tensor(item[0]) for item in batch])
        labels = torch.stack([torch.tensor(item[1]) for item in batch])
        # Generate graph_label tensor
        graph_label = 0 if labels.sum() == 0 else 1
        graph_label = torch.tensor(graph_label, dtype=torch.long)
        ids = labels = torch.stack([torch.tensor(item[2]) for item in batch])
        return features, ids, graph_label


class LogFlash(torch.nn.Module):
    def __init__(
        self, 
        num_templates: int, 
        time_window_size: int, 
        decay_rate: float, 
        update_step_size: float, 
        feature_dim: int,
        num_classes: int,
    ):
        super(LogFlash, self).__init__()
        self.num_templates = num_templates
        self.time_window_size = time_window_size
        self.decay_rate = decay_rate
        self.update_step_size = update_step_size
        self.transition_rate_matrix = torch.zeros((num_templates, num_templates))
        self.time_weight_matrix = torch.zeros((num_templates, num_templates))
        self.last_occurrence = torch.zeros(num_templates, dtype=torch.long)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
            torch.nn.Sigmoid()
        )

    def forward(self, template_stream, feature_stream):
        anomaly_scores = []
        for t in range(len(template_stream)):
            # current_template = template_stream[t]
            current_feature = feature_stream[t]
            # sub_stream = template_stream[max(0, t - self.time_window_size):t]
            # for i in range(len(sub_stream)):
            #     if sub_stream[i] != current_template:
            #         self.update_transition_rate(sub_stream[i], current_template)
            #         self.update_time_weight(sub_stream[i], current_template, t - self.last_occurrence[sub_stream[i]])
            #     self.last_occurrence[sub_stream[i]] = t
            # for i in range(self.num_templates):
            #     if i not in sub_stream and self.transition_rate_matrix[i, current_template] > 0:
            #         self.transition_rate_matrix[i, current_template] *= self.decay_rate
            anomaly_score = self.network(current_feature)
            anomaly_scores.append(anomaly_score)
            
        # graph_level_anomaly_score = torch.max(torch.stack(anomaly_scores))
        graph_level_anomaly_score = torch.max(torch.stack(anomaly_scores), dim=0)[0]
        # print(graph_level_anomaly_score)
        return graph_level_anomaly_score
    
    def update_transition_rate(self, i, j):
        self.transition_rate_matrix[i, j] += 1

    def update_time_weight(self, i, j, transition_time):
        if transition_time < self.time_weight_matrix[i, j] - 1:
            self.time_weight_matrix[i, j] *= self.decay_rate
        elif transition_time > self.time_weight_matrix[i, j]:
            self.time_weight_matrix[i, j] = transition_time


class LogFlashTrainer:
    def __init__(
        self, 
        model: LogFlash, 
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
        for features, ids, graph_label in tqdm(dataloader):
            features = features.to(self.device)
            ids = ids.to(self.device)
            graph_label = graph_label.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(ids, features)
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
    
    

class LogFlashPredictor:
    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)

    def predict(self, dataloader):
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for features, ids, graph_label in tqdm(dataloader):
                features = features.to(self.device)
                ids = ids.to(self.device)

                outputs = self.model(ids, features)
                all_outputs.append(outputs.cpu())
                all_labels.append(graph_label.unsqueeze(0))
        all_outputs = torch.stack(all_outputs)
        all_labels = torch.cat(all_labels)

        # Convert the outputs to probabilities using softmax
        probabilities = torch.softmax(all_outputs, dim=-1)[:, 1]  # We take the second column because it's the probability of the positive class

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
    from sklearn.model_selection import train_test_split
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    ckpt_dir = "results/BGL/LogFlash"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    train_batch_size = 512
    eval_batch_size = 1024
    decay_rate = 0.9
    update_step_size = 0.1
    
    df = pd.read_csv('dataset/BGL/BGL.log_structured.csv').sample(frac=0.01, random_state=42)
    num_templates = len(df['EventId'].unique())
    model = LogFlash(
        num_templates, 
        train_batch_size, 
        decay_rate, 
        update_step_size, 
        feature_dim=768,
        num_classes=2,
    )
    
    train_ids, test_ids = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    
    train_dataset = LogFlashDataset('dataset/BGL/BGL.log_structured.csv', train_ids)
    val_dataset = LogFlashDataset('dataset/BGL/BGL.log_structured.csv', val_ids)
    test_dataset = LogFlashDataset('dataset/BGL/BGL.log_structured.csv', test_ids)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=My_collate(train_batch_size))
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=My_collate(eval_batch_size))
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=My_collate(eval_batch_size))
    
    trainer = LogFlashTrainer(model, device, ckpt_dir)
    predictor = LogFlashPredictor(model, device)
    
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