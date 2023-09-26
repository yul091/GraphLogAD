import sys
sys.dont_write_bytecode = True
import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, load_metric
from transformers import (
    BartTokenizer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, 
    BartForConditionalGeneration,
)
from utils import *


#################################################################################################################################################
###################################################### NER seq2seq inference ####################################################################
#################################################################################################################################################

def template_entity(words, input_TXT, start, tokenizer, model, device=DEVICE, strategy=0):
    '''
    tokenizer: huggingface transformer pre-trained tokenizer.
    model: huggingface transformer pre-trained language model.
    words (list): list of all enumerated word phrases starting from 'start' index. 
    '''
    # input text -> template
    num_words = len(words)
    num_labels = len(LABEL2TEMPLATE) + 1
    input_TXT = [input_TXT]*(num_labels*num_words)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [v[strategy] for v in LABEL2TEMPLATE.values()] + [NONE2TEMPLATE[strategy]]
    entity_dict = {i:k for i,k in enumerate(LABEL2TEMPLATE.keys())}
    entity_dict[len(LABEL2TEMPLATE)] = 'O'
    
    temp_list = []
    for i in range(num_words):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids'] # num_words*num_labels X T
    output_ids[:, 0] = tokenizer.eos_token_id # num_words*num_labels X T
    output_length_list = [0]*num_labels*num_words 

    for i in range(len(temp_list)//num_labels): # word phrase + is (+ not)
        base_length = ((tokenizer(temp_list[i * num_labels], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*num_labels:i*num_labels+num_labels] = [base_length]*num_labels
        output_length_list[i*num_labels+num_labels-1] += 1 # negative ones

    score = [1 for _ in range(num_labels*num_words)] # placeholder for template scores
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0] # 2 means "entity ."
        for i in range(output_ids.shape[1] - 3): # 2 + 1
            logits = output[:, i, :] # num_words*num_labels X V
            logits = logits.softmax(dim=1) # num_words*num_labels X V
            logits = logits.to('cpu').numpy()
            for j in range(num_labels*num_words):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    largest_idx = score.index(max(score))
    # score_temp = [(s, t) for s, t in zip(score, temp_list)]
    # for i, (s, t) in enumerate(score_temp):
    #     if i == largest_idx:
    #         print('(best) score: {}, temp: {}, entity: {}, label: {}'.format(s, t, words[i%num_words], entity_dict[i%num_labels]))
    #     else:
    #         print('score: {}, temp: {}, entity: {}, label: {}'.format(s, t, words[i%num_words], entity_dict[i%num_labels]))
    end = start+(largest_idx//num_labels)
    return [start, end, entity_dict[(largest_idx%num_labels)], max(score)] # [start_index, end_index, label, score]



def prediction(input_TXT, model, tokenizer, strategy=0, n=8, device=DEVICE):
    input_TXT_list = list(filter(None, re.split(TOKENIZE_PATTERN, input_TXT)))
    num_tok = len(input_TXT_list) # number of tokens

    entity_list = []
    for i in range(num_tok): # i: start index
        words = []
        # Enumerate all word phrases starting from i
        for j in range(1, min(n+1, num_tok - i + 1)): # j: offset index (w.r.t. i)
            word = (' ').join(input_TXT_list[i:i+j]) # words[i:i+j]
            words.append(word) 

        entity = template_entity(words, input_TXT, i, tokenizer, model, device, strategy) # [start_index, end_index, label, score]
        if entity[1] >= num_tok:
            entity[1] = num_tok-1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * num_tok

    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
        
    return label_list


def get_entities_bio(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return set([tuple(chunk) for chunk in chunks])


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels



if __name__ == '__main__':

    # Parser args
    parser = argparse.ArgumentParser(
        description='Generate few-shot datasets and extract named entities'
    )

    # Logistics
    parser.add_argument(
        '--template', '-t', type=str, 
        default='dataset/template.txt', 
        help='path to the raw template file')
    parser.add_argument(
        '--gen_data', '-d', type=str, 
        default='dataset/dataset.json', 
        help='path to the generated dataset json file')
    parser.add_argument(
        '--data_name', '-n', type=str, 
        default='bgl', 
        choices=['AIT', 'BGL', 'HDFS'], 
        help='Dataset name')
    parser.add_argument(
        '--output_dir', '-o', type=str,
        default='dataset/NER',
        help='path to the generated files directory'
    )
    parser.add_argument(
        '--ckpt_dir', type=str,
        default='results/BART_seq2seq/10-shot-0',
        help='checkpoint directory'
    )
    parser.add_argument(
        '--strategy', '-s', type=int, 
        default=0,
        choices=[0, 1], 
        help='strategy to generate prompt template')
    parser.add_argument(
        '--n_grams', type=int, default=8,
        help='how many grams for generating entity phrases'
    )
    parser.add_argument(
        '--neg_rate', type=float, default=1.5,
        help='negative rate for sampling negative entity phrases'
    )
    parser.add_argument(
        '--n_shots', type=int, default=10,
        help='how many shots for each class to generate few-shot datasets'
    )
    parser.add_argument(
        '--labeling_technique', type=str,
        default='prompt',
        choices=['prompt', 'regex'],
        help='use prompt seq2seq or regular expression to recognize named entities'
    )

    # Training args
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--max_source_length', type=int, default=1024)
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--model_name_or_path', type=str, default='facebook/bart-large')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=50)
    parser.add_argument(
        '--pad_to_max_length', action='store_true', default=False,
        help="whether to pad all samples to model maximum sentence length")
    parser.add_argument(
        '--ignore_pad_token_for_loss', type=bool,
        default=True,
        help="whether to ignore the tokens corresponding to padded labels in the loss computation or not")
    parser.add_argument(
        '--preprocessing_num_workers', type=int,
        default=None,
        help="the number of processes to use for the preprocessing")
    parser.add_argument(
        '--overwrite_cache', action='store_true', default=False,
        help="overwrite the cached training and evaluation sets")
    args = parser.parse_args()

    # Arguments
    template = args.template
    gen_data = args.gen_data
    seed = args.seed
    data_name = args.data_name
    output_dir = args.output_dir
    ckpt_dir = args.ckpt_dir
    n_shots = args.n_shots
    n_grams = args.n_grams
    neg_rate = args.neg_rate
    strategy = args.strategy
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    pad_to_max_length = args.pad_to_max_length
    max_train_samples = args.max_train_samples
    model_name_or_path = args.model_name_or_path
    num_train_epochs = args.num_train_epochs
    do_train = args.do_train
    do_eval = args.do_eval
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    checkpoint = args.checkpoint
    ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
    preprocessing_num_workers = args.preprocessing_num_workers
    overwrite_cache = args.overwrite_cache
    labeling_technique = args.labeling_technique

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if data_name == 'ait':
        # Preprocess to generate dataset (from raw log templates) for AIT dataset
        raw_data = pd.read_csv(template, sep='\n', header=None)[0]
        process_corpus(raw_data, gen_data) # run to generate few-shot.json
    data = load_dataset('json', data_files=gen_data)['train'] # load preprocessed data

    # Get tag-entity statistics
    entity_set = defaultdict(set)
    entity_count = defaultdict(list)
    for i, instance in enumerate(data):
        for ent, tag in zip(instance[ENTITY_COLUMN_NAME], instance[TAG_COLUMN_NAME]):
                entity_set[tag].add(ent)
                entity_count[tag].append(i)

    entity_occ = sum(len(ids) for ids in entity_count.values())
    print("Total #entities: {}, average #entities per log: {:.3f}".format(entity_occ, entity_occ/len(data)))
    print("Entity distribution ({}): {}".format(len(entity_count), {k:len(v) for k,v in entity_count.items()}))
    print('log: "%s",'%data[LOG_COLUMN_NAME][0], 
          'entities: %s,'%data[ENTITY_COLUMN_NAME][0], 
          'tags: %s.'%data[TAG_COLUMN_NAME][0])
    # for tag, entities in entity_set.items():
    #     print("\t{}: {}".format(tag, entities))

    # Split train(10-shot & 5-shot)/val/test data for NER training
    n_shot_ids = []
    ten_shot_ids = []
    random.seed(seed)
    for tag in entity_count:
        tag_ids = random.choices(entity_count[tag], k=10) 
        ten_shot_ids.extend(tag_ids) # 10-shot
        n_shot_ids.extend(tag_ids[:n_shots])

    n_shot_data = data.select(n_shot_ids).shuffle(seed=seed) # n-shot
    remain_ids = list(set(range(len(data))) - set(ten_shot_ids)) 
    val_ids = random.sample(remain_ids, int(len(remain_ids)*0.5))
    val_data = data.select(val_ids)
    test_ids = list(set(remain_ids) - set(val_ids))
    test_data = data.select(test_ids)

    print('(before template generation) train', n_shot_data)
    print('(before template generation) validation', val_data)
    print('(before template generation) test', test_data)

    # Generate train/val/test dataset
    trainPath = os.path.join(output_dir, f"train-{n_shots}-shot-{strategy}.json")
    valPath = os.path.join(output_dir, f"val-{strategy}.json")
    testPath = os.path.join(output_dir, "test.json")
    gen_train_prompt(n_shot_data, trainPath, strategy, n=n_grams, negrate=neg_rate, seed=seed)
    gen_train_prompt(val_data, valPath, strategy, n=n_grams, negrate=neg_rate, seed=seed)

    # Load preprocessed data
    datasets = load_dataset('json', data_files={'train': trainPath, 'validation': valPath})
    train_set = datasets['train']
    val_set = datasets['validation']

    # Generate test labels
    gen_test_labels(test_data, testPath) 
    test_set = load_dataset('json', data_files=testPath)['train']

    if labeling_technique == 'prompt':
        # Process few-shot data (tokenization)
        padding = "max_length" if pad_to_max_length else False
        column_names = train_set.column_names # string row columns

        # Instantiate tokenizer (bart-large)
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path)

        def preprocess_function(examples):
            model_inputs = tokenizer(
                examples[INPUT_COLUMN_NAME], 
                max_length=max_source_length, 
                padding=padding, 
                truncation=True,
            )
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples[TARGET_COLUMN_NAME], 
                    max_length=max_target_length, 
                    padding=padding, 
                    truncation=True,
                )

            # If padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs


        tokenized_train_data = train_set.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names, # remove columns that contain strings
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )
        tokenized_val_data = val_set.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names, # remove columns that contain strings
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
        )
        print('train', tokenized_train_data)
        print('validation', tokenized_val_data)
        print('test', test_set)

        # Metric
        metric = load_metric("sacrebleu")

        def compute_metrics(eval_preds):
            """`
            preds (np.ndarray[float], np.ndarray[float]): B X T X V, B X T' X V'
            labels (np.ndarray[int]): B X T 
            """
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            
            preds = preds.argmax(axis=-1) # B X T X V (float) -> B X T (int)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            if ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result


        training_args = Seq2SeqTrainingArguments(
            output_dir=ckpt_dir,
            do_train=do_train,
            do_eval=do_eval,
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False, # smaller eval loss is better
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
        )

        # Data collator
        label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        # Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_data if training_args.do_train else None,
            eval_dataset=tokenized_val_data if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if do_train:
            # Start training
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            trainer.save_model()  # Saves the tokenizer too for easy upload

            max_train_samples = (
                max_train_samples if max_train_samples is not None else len(tokenized_train_data)
            )
            metrics["train_samples"] = min(max_train_samples, len(tokenized_train_data))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if do_eval:
            # Evaluate 
            model.eval()
            model.config.use_cache = False
            preds_list, trues_list = [], []

            for instance in tqdm(test_set):
                log = instance['log']
                pred = prediction(log, model, tokenizer, strategy)
                preds_list.append(pred)
                trues_list.append(instance[LABEL_COLUMN_NAME])
                print('Pred:', pred)
                print('Gold:', instance[LABEL_COLUMN_NAME])

            # Handling BIO for extracting entity pairs
            true_entities = [get_entities_bio(true_list) for true_list in trues_list]
            pred_entities = [get_entities_bio(pred_list) for pred_list in preds_list]
            # print(pred_entities, true_entities)
            results = {
                "precision": precision_score(true_entities, pred_entities),
                "recall": recall_score(true_entities, pred_entities),
                "f1": f1_score(true_entities, pred_entities),
            }

            eval_dict = evalutation_report(true_entities, pred_entities, entity_count)
            print(eval_dict)

            # Save predictions
            savePath = os.path.join(output_dir, f"preds-prompt-{n_shots}-shot-{strategy}.json")
            save_preds(test_set, pred_entities, true_entities, savePath, results, eval_dict, labeling_technique) 

    else:
        # initializing the list object
        preds_list, trues_list = [], []
        for i, instance in enumerate(test_set):
            preds, truths = set(), set()
            text = test_set[i]['log']

            for tag, pat in REGEX_PATTERN.items():
                ans = re.findall(pat, text+' ')
                if ans:
                    for phrase in ans:
                        if isinstance(phrase, str):
                            preds.add((tag, phrase))
                        elif isinstance(phrase, tuple):
                            # phrase = max(list(phrase), key=len)
                            phrase = min(list(phrase), key=len)
                            preds.add((tag, phrase))
                        else:
                            raise TypeError()

            for ent, tag in zip(instance['entities'], instance['tags']):
                truths.add((tag, ent))

            trues_list.append(truths)
            preds_list.append(preds)

        true_entities = trues_list
        pred_entities = preds_list
        results = {
            "precision": precision_score(true_entities, pred_entities),
            "recall": recall_score(true_entities, pred_entities),
            "f1": f1_score(true_entities, pred_entities),
        }

        eval_dict = evalutation_report(true_entities, pred_entities, entity_count)
        print(eval_dict)

        # Save predictions
        savePath = os.path.join(output_dir, f"preds-regex.json")
        save_preds(test_set, pred_entities, true_entities, savePath, results, eval_dict, labeling_technique) 

