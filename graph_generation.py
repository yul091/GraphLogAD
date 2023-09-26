import sys
sys.dont_write_bytecode = True
import re
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict, defaultdict
from sklearn.model_selection import train_test_split
from utils import (
    TOKENIZE_PATTERN, 
    REGEX_PATTERN, 
    LABEL2TEMPLATE,
    SOCK_SHOP_ENT, 
)
from NER import prediction, get_entities_bio
from argparse import Namespace
from graph_dataset import HDFSDataset, BGLDataset, BGLNodeDataset, SockShopNodeDataset
from datasets import load_from_disk, load_dataset, Dataset


def handle_string(string: str):
    # Extract the JSON part of the string using regular expressions
    json_string = re.search('{.*}', str(string))
    if json_string:
        json_string = json_string.group()

    # Load the JSON string as a Python dictionary
    try:
        data = json.loads(json_string)
    except:
        data = {}

    def extract_pairs(data: dict, pairs: list):
        """Recursively extract key-value pairs from a nested dictionary"""
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    extract_pairs(item, pairs)
            elif isinstance(value, dict):
                if key in SOCK_SHOP_ENT:
                    pairs.append((key, json.dumps(value)))
                extract_pairs(value, pairs)
            else:
                if key in SOCK_SHOP_ENT and value:
                    pairs.append((key, value))
        return pairs

    # Extract all key-value pairs
    pairs = extract_pairs(data, [])
    
    user = None
    all_others = []
    for key, value in pairs:
        if key == 'customer':
            user = value
        elif key in SOCK_SHOP_ENT:
            all_others.append(value)
            
    if user: 
        if len(all_others) >= 10:
            label = 1
        else:
            label = 0
    else:
        label = 0
    
    return pairs, label


def add_sock_shop_preds_to_df(struct_df: pd.DataFrame):
    preds, labels = [], []
    print("Start matching!!! Total number of logs: {}".format(struct_df.shape[0]))
    
    for idx, instance in tqdm(struct_df.iterrows()):
        log = instance['Content']
        pairs, label = handle_string(log)
        preds.append(pairs)
        labels.append(label)
        
    struct_df['Preds'] = preds
    struct_df['Label'] = labels
    print("Finished matching!!! Total number of anomalies: {}".format(sum(labels)))


def add_preds_to_df(struct_df, inference_type='seq2seq', language_model=None, tokenizer=None, strategy=0):
    preds = []
    if inference_type == 'seq2seq':
        # Obtain the extracted entities (&tags) for each event
        pred_pattern = {}
        event_groups = struct_df.groupby(['EventId']).groups
        print("Start inference!!! Total number of events: {}".format(len(event_groups)))
        for eventID, insIDs in tqdm(event_groups.items()):
            instance = struct_df.iloc[int(insIDs[0])] # pick the first instance of each group
            # print('EventID:{}, Log:{}'.format(eventID, instance['Content']))
            pred = prediction(
                instance['Content'], 
                language_model, 
                tokenizer, 
                strategy=strategy,
            )  # token classification
            # print('\tPred:', pred)
            entities = list(get_entities_bio(pred)) # merge tokens within the same entity
            entities.sort(key=lambda x: x[1])
            # print('Extracted entities:', entities)
            # input_tokens = list(filter(None, re.split(TOKENIZE_PATTERN, instance['Content'])))  
            # ent_list = [(tag, ' '.join(input_tokens[start:end+1])) for (tag, start, end) in entities]
            # print('\tExtracted entities:', ent_list)
            pred_pattern[eventID] = entities

        print("Summerized prediction patterns ({})".format(len(pred_pattern)))
        print("Start matching!!! Total number of logs: {}".format(struct_df.shape[0]))

        for idx, instance in tqdm(struct_df.iterrows()):
            ent_ids = pred_pattern[instance['EventId']] # predicted entities for each event
            log = instance['Content']
            input_tokens = list(filter(None, re.split(TOKENIZE_PATTERN, log))) 
            # Map to other logs
            ent_list = [(tag, ' '.join(input_tokens[start:end+1])) for (tag, start, end) in ent_ids] 
            preds.append(ent_list)

    elif inference_type == 'regex':
        print("Start matching!!! Total number of logs: {}".format(struct_df.shape[0]))
        for idx, instance in tqdm(struct_df.iterrows()):
            pred = set()
            log = instance['Content']
            for tag, pat in REGEX_PATTERN.items():
                ans = re.findall(pat, log+' ')
                if ans:
                    for phrase in ans:
                        if isinstance(phrase, str):
                            pred.add((tag, phrase))
                        elif isinstance(phrase, tuple):
                            phrase = max(list(phrase), key=len)
                            pred.add((tag, phrase))
                        else:
                            raise TypeError()
                            
            preds.append(pred)
    else:
        raise ValueError(f"Not Supported inference_type {inference_type}!")

    struct_df['Preds'] = preds


def splitbyinterval(df, interval='2min'):
    new_df = df.copy(deep=True)
    try:
        new_df['Datetime'] = new_df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    except:
        new_df['Datetime'] = pd.to_datetime(new_df['Timestamp'])
    period = new_df.groupby(pd.Grouper(key='Datetime', freq=interval)).ngroup()
    new_df['Period'] = np.char.add('period_', (pd.factorize(period)[0]).astype(str))
    return new_df



def get_train_test_data(data_df):
    # Split train and test data
    print("Splitting graph datasets!!!")
    print(data_df[['Preds', 'EventLabels', 'Label']])
    num_total = data_df.shape[0]
    normal_samples = data_df[data_df.Label == 0]
    anomaly_samples = data_df[data_df.Label == 1]
    num_normal = normal_samples.shape[0]
    num_anomaly = anomaly_samples.shape[0]
    anomaly_rate = num_anomaly/num_total if num_total else 0
    print('normal graphs: {}, anomaly graphs: {}'.format(num_normal, num_anomaly))

    train_df, test_normal_df = train_test_split(normal_samples, test_size=0.2, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed)
    test_df = pd.concat([anomaly_samples, test_normal_df], ignore_index=True)
    test_anomaly_rate = num_anomaly/test_df.shape[0] if test_df.shape[0] else 0

    print("Total number of graphs: {}, normal graphs: {}, anomaly graphs: {}, anomaly ratio: {:.4f}".format(
        num_total, num_normal, num_anomaly, anomaly_rate))
    print("Train data size: {}, validation data size: {}, test data size: {}, test anomaly ratio: {:.4f}".format(
        train_df.shape[0], val_df.shape[0], test_df.shape[0], test_anomaly_rate))

    # Define args for geometric dataset
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return df




if __name__ == '__main__':
    import argparse
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
    )

    # Parser args
    parser = argparse.ArgumentParser(
        description='Generating graphs'
    )

    # Logistics
    parser.add_argument(
        '--common_dir', type=str, 
        default='dataset', 
        help='Path to the common data dir where the processed dataframe is stored')
    parser.add_argument(
        '--root', '-r', type=str, 
        default='dataset/HDFS', 
        help='Path to the root dir where the processed torch_geometric.data.Dataset is generated')
    parser.add_argument(
        '--log_file', '-l', type=str, 
        default='dataset/HDFS/HDFS.log_structured.csv', 
        help='Path to the log structured template file')
    parser.add_argument(
        '--label_file', '-y', type=str, 
        default='dataset/HDFS/anomaly_label.csv', 
        help='Path to the log label file')
    parser.add_argument('--strategy', type=int, 
        default=0, 
        help='Prompt template type for seq2seq NER prediction')
    parser.add_argument('--inference_type', type=str, 
        choices=['seq2seq', 'regex'],
        default='seq2seq', 
        help='Prompt template type for seq2seq NER prediction')
    parser.add_argument(
        '--label_type', type=str,
        default='graph',
        choices=['graph', 'node'],
        help='Node embedding or graph embedding for BGL dataset')
    parser.add_argument(
        '--pretrained_model_name_or_path', '-p', type=str, 
        default='facebook/bart-large', 
        help='Pre-trained seq2seq model')
    parser.add_argument(
        '--interval', type=str,
        default='2min',
        help='Time interval for splitting BGL dataset')
    parser.add_argument(
        '--event_template', action='store_true', 
        default=False,
        help='Whether to use event template as attribute for event nodes')
    parser.add_argument(
        '--use_cache', action='store_true', 
        default=False,
        help='Whether to use saved dataframe for generation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Arguments
    common_dir = args.common_dir
    root = args.root # root dir
    log_file = args.log_file # the structured log file
    label_file = args.label_file # the anomaly label file
    seed = args.seed
    strategy = args.strategy
    inference_type = args.inference_type
    label_type = args.label_type
    interval = args.interval
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    using_event_template = args.event_template
    use_cache = args.use_cache

    if not os.path.isdir(os.path.join(root, 'raw')):
        # Define bart pre-trained model and tokenizer
        if inference_type == 'seq2seq':
            print("Using seq2seq NER model!!!")
            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)
        else:
            print("Using regular expression for NER matching!!!")
            tokenizer = None
            model = None

        # Configure saved json name and path
        if 'HDFS' in log_file:
            if inference_type == 'seq2seq':
                pred_df_name = f'HDFS_pred_seq2seq_{strategy}.json'
                grouped_df_name = f'HDFS_grouped_seq2seq_{strategy}.json'
            else:
                pred_df_name = f'HDFS_pred_regex.json'
                grouped_df_name = f'HDFS_grouped_regex.json'
        elif 'BGL' in log_file:
            if inference_type == 'seq2seq':
                pred_df_name = f'BGL_pred_seq2seq_{strategy}.json'
                grouped_df_name = f'BGL_grouped_seq2seq_{strategy}_{interval}.json'
            else:
                pred_df_name = f'BGL_pred_regex.json'
                grouped_df_name = f'BGL_grouped_regex_{interval}.json'
        elif 'AIT' in log_file:
            if inference_type == 'seq2seq':
                pred_df_name = f'AIT_pred_seq2seq_{strategy}.json'
                grouped_df_name = f'AIT_grouped_seq2seq_{strategy}_{interval}.json'
            else:
                pred_df_name = f'AIT_pred_regex.json'
                grouped_df_name = f'AIT_grouped_regex_{interval}.json'
        elif 'sockshop' in log_file:
            pred_df_name = f'sockshop_pred_regex.json'
            grouped_df_name = f'sockshop_grouped_regex_{interval}.json'
        else:
            raise ValueError("logfile dataset type not supported! Must be HDFS or BGL!")

        if not os.path.isdir(common_dir):
            os.makedirs(common_dir)
        pred_df_path = os.path.join(common_dir, pred_df_name)
        common_df_path = os.path.join(common_dir, grouped_df_name)

        # if os.path.exists(common_df_path) and use_cache:
        #     # Load grouped json dataset
        #     grouped_data = load_dataset('json', data_files={'train': common_df_path}, split='train')
        #     data_df = grouped_data.to_pandas()
        # else:
        # # Generate predictions and save
        # if os.path.exists(pred_df_path) and use_cache:
        #     struct_data = load_dataset('json', data_files={'train': pred_df_path}, split='train')
        #     struct_df = struct_data.to_pandas()
        # else:
        print("Generating predictions !!!")
        if 'BGL' in log_file or 'HDFS' in log_file:
            struct_df = pd.read_csv(log_file, na_filter=False, memory_map=True)
            add_preds_to_df(struct_df, inference_type, model, tokenizer, strategy)
        elif 'sockshop' in log_file:
            struct_df = pd.read_csv(log_file, na_filter=False, memory_map=True)
            struct_df.drop(columns=['Unnamed: 0'], inplace=True)
            struct_df.rename(columns={'log': 'Content', '@timestamp': 'Timestamp'}, inplace=True)
            add_sock_shop_preds_to_df(struct_df)
        else:
            # AIT dataset
            struct_df = load_from_disk(log_file).to_pandas()
            struct_df.Label = struct_df.Label.apply(lambda x: '0' if set(x) == set('0') else '1')
            add_preds_to_df(struct_df, inference_type, model, tokenizer, strategy)

        # # Save pred_df for reusage
        # struct_data = Dataset.from_pandas(struct_df)
        # struct_data.to_json(pred_df_path)

        # Grouped by Time interval (or BlockId for HDFS)
        if 'HDFS' in log_file:
            # Get blockId and corresponding logs
            print("Preparing HDFS dataset ...")
            struct_df['Datetime'] = struct_df['Time'].apply(lambda x: datetime.fromtimestamp(x))
            print("Getting BlockIDs and Logs!!! Total number of logs: {}".format(struct_df.shape[0]))
            data_dict = OrderedDict()
            for idx, row in tqdm(struct_df.iterrows()):
                blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
                blkId_set = set(blkId_list)
                for blk_Id in blkId_set:
                    if not blk_Id in data_dict:
                        data_dict[blk_Id] = defaultdict(list)
                        data_dict[blk_Id]['BlockId'] = blk_Id
                    for col in struct_df.columns:
                        data_dict[blk_Id][col].append(row[col])

            data_df = pd.DataFrame(data_dict.values())
            # Add labels to each block 
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

        elif 'BGL' in log_file:
            # Split by time interval
            print("Preparing for BGL dataset ...")
            print("Split by interval {}!!! Total number of logs: {}".format(interval, struct_df.shape[0]))
            grouped_df = splitbyinterval(struct_df, interval)
            data_dict = OrderedDict()
            for idx, row in tqdm(grouped_df.iterrows()):
                group_id = row['Period']
                if group_id not in data_dict:
                    data_dict[group_id] = defaultdict(list)

                for col in grouped_df.columns:
                    data_dict[group_id][col].append(row[col])
                data_dict[group_id]

            data_df = pd.DataFrame(data_dict.values())
            # Add labels to each group
            data_df['EventLabels'] = data_df['Label'].apply(lambda x: [0 if item=='-' else 1 for item in x])
            data_df['Label'] = data_df['Label'].apply(lambda x: 0 if set(x) == set('-') else 1)
            
        elif 'sockshop' in log_file:
            print("Preparing for sockshop dataset ...")
            print("Split by interval {}!!! Total number of logs: {}".format(interval, struct_df.shape[0]))
            grouped_df = splitbyinterval(struct_df, interval)
            data_dict = OrderedDict()
            for idx, row in tqdm(grouped_df.iterrows()):
                group_id = row['Period']
                if group_id not in data_dict:
                    data_dict[group_id] = defaultdict(list)

                for col in grouped_df.columns:
                    data_dict[group_id][col].append(row[col])
                data_dict[group_id]

            data_df = pd.DataFrame(data_dict.values())
            # Add labels to each group
            data_df['EventLabels'] = data_df['Label'].copy()
            data_df['Label'] = data_df['Label'].apply(lambda x: 0 if set(x) == set([0]) else 1)

        else:
            # Split by time interval
            print("Preparing for AIT dataset ...")
            print("Split by interval {}!!! Total number of logs: {}".format(interval, struct_df.shape[0]))
            grouped_df = splitbyinterval(struct_df, interval)
            data_dict = OrderedDict()
            for idx, row in tqdm(grouped_df.iterrows()):
                group_id = row['Period']
                if group_id not in data_dict:
                    data_dict[group_id] = defaultdict(list)

                for col in grouped_df.columns:
                    data_dict[group_id][col].append(row[col])
                data_dict[group_id]

            data_df = pd.DataFrame(data_dict.values())
            # Add labels to each group
            data_df['EventLabels'] = data_df['Label'].apply(lambda x: [0 if item=='0' else 1 for item in x])
            data_df['Label'] = data_df['Label'].apply(lambda x: 0 if set(x) == set('0') else 1)

            # # Save to common path for reuse
            # grouped_data = Dataset.from_pandas(data_df)
            # grouped_data.to_json(common_df_path)

        df = get_train_test_data(data_df)
    else:
        df = pd.DataFrame([])
    
    # Ontology
    tag2id = {ent:i for i, ent in enumerate(LABEL2TEMPLATE.keys())}
    tag2id['event'] = len(tag2id)
    tag2id['component'] = len(tag2id)
    # tag2id['device'] = len(tag2id) # for AIT dataset

    # Define hyperparameters
    hparams = Namespace(
        df=df,
        tag2id=tag2id,
        using_event_template=using_event_template,
    )

    # Instantiate torch_geometric.data.Dataset
    if 'HDFS' in log_file:
        print("Generating HDFS torch_geometric.data.Dataset!!!")
        graph_data = HDFSDataset(root, hparams=hparams)
    elif 'BGL' in log_file:
        if label_type == 'graph':
            print("Generating BGL torch_geometric.data.Dataset (graph labeling)!!!")
            graph_data = BGLDataset(root, hparams=hparams)
        else:
            print("Generating BGL torch_geometric.data.Dataset (node labeling)!!!")
            graph_data = BGLNodeDataset(root, hparams=hparams)
    elif 'sockshop' in log_file:
        print("Generating Sock Shop torch_geometric.data.Dataset (node labeling)!!!")
        graph_data = SockShopNodeDataset(root, hparams=hparams)
    else:
        # AIT dataset
        if label_type == 'graph':
            print("Generating AIT torch_geometric.data.Dataset (graph labeling)!!!")
            graph_data = BGLDataset(root, hparams=hparams)
        else:
            print("Generating AIT torch_geometric.data.Dataset (node labeling)!!!")
            graph_data = BGLNodeDataset(root, hparams=hparams)
