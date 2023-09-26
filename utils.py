import random
import re
import ast
import json
import numpy as np
import torch
from torch_scatter import scatter
from sklearn.metrics import roc_curve, auc, average_precision_score, classification_report
from sentence_transformers import SentenceTransformer, util


# Strategy (0: '<entity> is a [MASK] entity .', 1: '<entity>=[MASK] .')
LABEL2TEMPLATE = {
    # 'appbuilder': [' is a builder entity .', '=builder .'], 
    # 'appflevel': [' is an f-level entity .', '=f-level .'], 
    'application': [' is an application entity .', '=application .'], 
    # 'appsigs': [' is an appsigs entity .', '=sigs .'], 
    'version': [' is a version entity .', '=version .'], # not in sock shop
    'domain': [' is a domain entity .', '=domain .'], # not in sock shop
    'duration': [' is a duration entity .', '=duration .'], 
    # 'file': [' is a file entity .', '=file .'], 
    'path': [' is a path entity .', '=path .'], 
    'ip': [' is an ip entity .', '=ip .'], 
    # 'method': [' is a method entity .', '=method .'], 
    # 'mode': [' is a mode entity .', '=mode .'], 
    'pid': [' is a pid entity .', '=pid .'], 
    'port': [' is a port entity .', '=port .'], 
    'session': [' is a session entity .', '=session .'], 
    'time': [' is a time entity .', '=time .'], 
    # 'tty': [' is a tty entity .', '=tty .'], 
    'uid': [' is a uid entity .', '=uid .'], 
    'url': [' is a url entity .', '=url .'], 
    # 'usergroup': [' is a usergroup entity .', '=usergroup .'], 
    'user': [' is a user entity .', '=user .'],
    'server': [' is a server entity .', '=server .'],
    'email': [' is an email entity .', '=email .'],
}
NONE2TEMPLATE = [' is not a named entity .', '=none .']
LOG_COLUMN_NAME = 'logex:example'
ENTITY_COLUMN_NAME = 'logex:hasParameterList'
TAG_COLUMN_NAME = 'logex:hasNERtag'
INPUT_COLUMN_NAME ='log'
TARGET_COLUMN_NAME ='prompt'
LABEL_COLUMN_NAME = 'ner_tags'

TOKENIZE_PATTERN = ' |(=)|(:) |([()])|(,) |([\[\]])|([{}])|([<>])|(\.) |(\.$)'

AIT_DATA_ROOT = 'dataset/AIT-LDS-v1_1'
AIT_NAME_DICT = {
    'mailcup': 'mail.cup.com', 
    'mailinsect': 'mail.insect.com', 
    'mailspiral': 'mail.spiral.com', 
    'mailonion': 'mail.onion.com'
}
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 1024 # huggingface pre-trained model hidden size (default)

REGEX_PATTERN = {}
REGEX_PATTERN['ip'] = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[,: )]'
REGEX_PATTERN['port'] = r'[pP]ort[=: |:|=|: |\s/]*(\d{1,5})'
REGEX_PATTERN['version'] = r'[vV]ersion[=: ]*(.*?)(?= )'
REGEX_PATTERN['session'] = r'[sS]ession[=:< ]*(.*?)(?=> )'
REGEX_PATTERN['pid'] = r' [pP]id[:|-|=|\s/]*(\d+)'
REGEX_PATTERN['uid'] = r' [uU]id[:|-|=|\s/]*(\d+)'
REGEX_PATTERN['user'] = r'r?[uU]ser[:|-|=|\s/]*<(\w+)>|r?[uU]ser[:|-|=|\s/]*(\w+)'
REGEX_PATTERN['path'] = r'((((?<!\w)[A-Z,a-z]:)|(\.{1,2}\\))([^\b%\/\|:\n\"]*))|("\2([^%\/\|:\n\"]*)")|((?<!\w)(\.{1,2})?(?<!\/)(\/((\\\b)|[^ \b%\|:\n\"\\\/])+)+\/?)'
REGEX_PATTERN['domain'] = r'\S+\.\w+\S+'
REGEX_PATTERN['email'] = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
REGEX_PATTERN['url'] = r'(https?://\S+)'
REGEX_PATTERN['time'] = r'(([0-1]?\d|2[0-3]):([0-5]?\d):([0-5]?\d))'


SOCK_SHOP_ENT = {
    'customer': r'r?"customer"[:|-|=|\s/]*{\w+}',
    'address': r'r?"address"[:|-|=|\s/]*<(\w+)>',
    'cards': r'r?"card"[:|-|=|\s/]*<(\w+)>',
    'items': r'r?"items"[:|-|=|\s/]*<(\w+)>',
    'quantity': r'r?"quantity"[:|-|=|\s/]*<(\w+)>',
    'unitPrice': r'r?"unitPrice"[:|-|=|\s/]*<(\w+)>',
    'itemId': r'r?"itemId"[:|-|=|\s/]*<(\w+)>',  
    'date': r'r?"date"[:|-|=|\s/]*<(\w+)>',
    "href": r'r?"href"[:|-|=|\s/]*<(\w+)>',
}

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


##################################################################################################################
#                                            Data handling functions                                             #
##################################################################################################################

def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    """Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.
    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    return adj



def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0), edge_attr
    

##################################################################################################################
#                                           Data Generation Functions                                            #
##################################################################################################################

def generate_ngrams(words, n):
    output = []  
    for i in range(len(words)-n+1):
        output.append(' '.join(words[i:i+n]))
    return output


def gen_templates(text, entities, tags, strategy=0, n=8, negrate=1.5, seed=0):
    templates = []
    ent_num = len(entities)
    # words = text.split()
    words = list(filter(None, re.split(TOKENIZE_PATTERN, text)))

    # Enumerate all positive templates
    for ent, tag in zip(entities, tags):
        prompt = ent + LABEL2TEMPLATE[tag][strategy]
        templates.append(prompt)

    # Enumerate all negative templates (using 1-gram ~ n-grams)
    ngrams = []
    for i in range(1, n+1):
        ngrams.extend(generate_ngrams(words, i))
    ngrams = list(set(ngrams) - set(entities)) # exclude entities

    # Sample negative templates
    if not ent_num:
        # If a sentence do not contain entities, still sample some negative phrases
        neg_num = max(1, int(0.1*len(words)))
    else:
        neg_num = min(int(ent_num * negrate), len(ngrams)-1)

    random.seed(seed)
    neg_sampled_grams = random.sample(ngrams, neg_num)
    for phrase in neg_sampled_grams:
        prompt = phrase + NONE2TEMPLATE[strategy]
        templates.append(prompt)

    return templates


def process_corpus(data, savePath):
    with open(savePath, 'w') as f:
        for i in range(len(data)):
            exp = re.sub(';', '', data.iloc[i])  # remove ';' at the end
            kv = exp.strip().split(None, 1) # strip left and right, split by the first whitespace
            if len(kv) == 1: # the beginning of a dict
                if i != 0:
                    # print(j, value)
                    json.dump(value, f) # save a dict object to file
                    f.write('\n')

                value = {}
                value['eventID'] = kv[0]
            else:
                if kv[0] in ['logex:hasAnnotation', 'logex:keyword']: # convert to list
                    vlist = kv[1].split(',')
                    value[kv[0]] = [v.replace('"', "").replace("\\","").strip() for v in vlist]
                elif kv[0] in ['logex:hasParameterList', 'logex:hasNERtag']: # extract elements in a list
                    value[kv[0]] = ast.literal_eval(kv[1]) # extract list from a string
                else:
                    vstr = kv[1].replace('"', "").replace("\\","") # get rid of backslash and quote
                    value[kv[0]] = vstr

    f.close()


def gen_train_prompt(data, trainPath, strategy=0, n=8, negrate=1.5, seed=0):
    with open(trainPath, 'w') as f:
        for i, instance in enumerate(data):
            text = instance[LOG_COLUMN_NAME]
            entities = instance[ENTITY_COLUMN_NAME]
            tags = instance[TAG_COLUMN_NAME]
            # Generate (text, template) pairs
            templates = gen_templates(
                text, 
                entities, 
                tags, 
                strategy=strategy, 
                n=n, 
                negrate=negrate,
                seed=seed,
            )
           
            for temp in templates:
                value = {'log': text, 'prompt': temp}
                f.write(json.dumps(value)) # save a dict object to file
                f.write('\n')

    f.close()


def gen_test_labels(data, testPath):
    with open(testPath, 'w') as f:
        for i, instance in enumerate(data):
            text = instance[LOG_COLUMN_NAME]
            entities = instance[ENTITY_COLUMN_NAME]
            tags = instance[TAG_COLUMN_NAME]
            
            # Tokenize and align labels
            words = list(filter(None, re.split(TOKENIZE_PATTERN, text)))
            labels = ['O' for _ in range(len(words))]
            temp_start = 0
            
            for ent, tag in zip(entities, tags):
                subwords = list(filter(None, re.split(TOKENIZE_PATTERN, ent)))
                n = len(subwords)
                phrase = ' '.join(subwords)

                for j in range(len(words)-n+1):
                    if phrase == ' '.join(words[j:j+n]) and j >= temp_start: # match
                        # print(j, phrase, labels, len(labels))
                        end_idx = min(len(words), j+n)
                        labels[j] = 'B-'+tag
                        for idx in range(j+1, end_idx):
                            labels[idx] = 'I-'+tag
                        temp_start = end_idx # update temp_start 
                        break      
            
            value = {'log': text, 'tokens': words, 'ner_tags': labels, 'entities': entities, 'tags': tags}
            f.write(json.dumps(value)) # save a dict object to file
            f.write('\n')

    f.close()


def save_preds(test_data, pred_entities, true_entities, savePath, results, eval_dict, labeling_technique='prompt'):
    with open(savePath, 'w') as f:
        f.write(json.dumps(results)) # first line: evaluation results
        f.write('\n')
        for ent, res in eval_dict.items():
            f.write(ent+': '+json.dumps(res))
            f.write('\n')

        for i, instance in enumerate(test_data):
            log = instance['log']
            tokens = instance['tokens']
            input_tokens = list(filter(None, re.split(TOKENIZE_PATTERN, log)))
            preds = list(pred_entities[i])
            preds.sort(key=lambda x: x[1])
            labels = list(true_entities[i])
            labels.sort(key=lambda x: x[1])
            if labeling_technique == 'prompt':
                preds = [(tag, ' '.join(input_tokens[start:end+1])) for (tag, start, end) in preds]
                labels = [(tag, ' '.join(input_tokens[start:end+1])) for [tag, start, end] in labels]

            value = {'log': log, 'tokens': tokens, 'preds': preds, 'labels': labels}

            f.write(json.dumps(value)) # save a dict object to file
            f.write('\n')

    f.close()



##################################################################################################################
#                                          Evaluation Metrics                                                    #
##################################################################################################################

def cal_auc_score(labels, preds):
    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr) 
    return roc_auc

def cal_aupr_score(labels, preds):
    aupr = average_precision_score(labels, preds)
    return aupr

def cal_accuracy(labels, preds, threshold=1):
    # sigmoid = 1/(1 + np.exp(-preds)) # normalized to [0,1]
    scores = np.int32(preds > threshold)
    acc = sum(scores == labels)/len(labels) if len(labels) else 0
    return acc

def cal_cls_report(labels, preds, threshold=1, output_dict=True):
    # sigmoid = 1/(1 + np.exp(-preds)) # normalized to [0,1]
    scores = np.int32(preds > threshold)
    report = classification_report(labels, scores, output_dict=output_dict)
    return scores, report

def f1_score(true_entities, pred_entities):
    """Compute the F1 score."""
    nb_correct = sum(len(true_ent & pred_ent) for true_ent, pred_ent in zip(true_entities, pred_entities))
    nb_pred = sum(len(pred_ent) for pred_ent in pred_entities)
    nb_true = sum(len(true_ent) for true_ent in true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def precision_score(true_entities, pred_entities):
    """Compute the precision."""
    nb_correct = sum(len(true_ent & pred_ent) for true_ent, pred_ent in zip(true_entities, pred_entities))
    nb_pred = sum(len(pred_ent) for pred_ent in pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(true_entities, pred_entities):
    """Compute the recall."""
    nb_correct = sum(len(true_ent & pred_ent) for true_ent, pred_ent in zip(true_entities, pred_entities))
    nb_true = sum(len(true_ent) for true_ent in true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def evalutation_report(true_entities, pred_entities, entity_count):
    """Compute the recall."""
    eval_dict = {tag: {'T': 0, 'P': 0, 'TP': 0, 'pre': 0, 'rec': 0, 'f1': 0} for tag in entity_count}
    for true_ent, pred_ent in zip(true_entities, pred_entities):
        if true_ent:
            for tup in list(true_ent):
                if tup[0] in eval_dict:
                    eval_dict[tup[0]]['T'] += 1
        if pred_ent:
            for tup in list(pred_ent):
                if tup[0] in eval_dict:
                    eval_dict[tup[0]]['P'] += 1
        
        intersect = true_ent & pred_ent
        if intersect:
            for tup in list(intersect):
                if tup[0] in eval_dict:
                    eval_dict[tup[0]]['TP'] += 1

    for tag, value in eval_dict.items():
        value['pre'] = value['TP']/value['P'] if value['P'] else 0
        value['rec'] = value['TP']/value['T'] if value['T'] else 0
        value['f1'] = 2*value['pre']*value['rec']/(value['pre']+value['rec']) if value['pre']+value['rec'] else 0

    return eval_dict


##################################################################################################################
#                                            Sentence Encoder                                                    #
##################################################################################################################
class SentenceEncoder:
    def __init__(self, device='cuda'):
        '''
        different version of Universal Sentence Encoder
        https://pypi.org/project/sentence-transformers/
        '''
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1', device)

    def encode(self, sentences):
        '''
        can modify this code to allow batch sentences input
        :param sentence: a String
        :return:
        '''
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.model.encode(sentences, convert_to_tensor=True)

    def get_sim(self, sentence1: str, sentence2: str):
        '''
        can modify this code to allow batch sentences input
        :param sentence1: a String
        :param sentence2: a String
        :return:
        '''
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()

    # find adversarial sample in advs which matches ori best
    def find_best_sim(self, ori, advs, find_min=False):
        ori_embedding = self.model.encode(ori, convert_to_tensor=True)
        adv_embeddings = self.model.encode(advs, convert_to_tensor=True)
        best_adv = None
        best_index = None
        best_sim = 10 if find_min else -10
        for i, adv_embedding in enumerate(adv_embeddings):
            sim = util.pytorch_cos_sim(ori_embedding, adv_embedding).item()
            if find_min:
                if sim < best_sim:
                    best_sim = sim
                    best_adv = advs[i]
                    best_index = i

            else:
                if sim > best_sim:
                    best_sim = sim
                    best_adv = advs[i]
                    best_index = i

        return best_adv, best_index, best_sim
