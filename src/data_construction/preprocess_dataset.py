import os
import sys
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

corpus_name = sys.argv[1]
split = sys.argv[2]
def add_neighbors(example):
    pid = example['pid']
    matrix = np.load('../../data_of_ReGPT/marco/neighbors.P2P.DPR34-1.top50.npy')
    embs = np.load('../../data_of_ReGPT/marco/phrases_embeddings.npy')
    example['neighbors'] = matrix[pid][::10]
    example['neighbor_embeddings'] = embs[example['neighbors']]
    return example

def filter_empty(example):
    return len(example['text']) > 0

def add_text_length(example):
    example["text_length"] = len(example["text"].split())
    return example

def filter_short_and_long_text(example):
    return example['text_length'] >= 64 and example['text_length'] <= 128

def preprocess_dataset(corpus_name, data_path, split='train'):
    dataset = load_dataset('csv', data_files={split: data_path}, delimiter='\t',column_names=['pid', 'text'])

    dataset = dataset.filter(filter_empty)

    dataset = dataset.map(add_text_length)

    dataset = dataset.filter(filter_short_and_long_text)

    dataset = dataset.sort("text_length", reverse=True)

    dataset = dataset[split]

    dataset.save_to_disk(f'../data_of_ReGPT/{corpus_name}/sorted_datasets_{split}')

def preprocess_qa_dataset(corpus_name, data_path, split='train'):
    dataset = load_from_disk(data_path)
    tokenizer = AutoTokenizer.from_pretrained("output/SFT-best")
    def filter_empty(example):
        return len(example['answers']) > 0 and len(example['answers'][0])>0

    def add_text_length(example):
        example["input_ids_length"] = len(tokenizer(example["query"]+"\n\nThe answer is:", example["answers"][0])['input_ids'])
        return example
    
    # dataset = dataset.filter(filter_empty)

    dataset = dataset.map(add_text_length)

    dataset = dataset.sort("input_ids_length", reverse=True)

    dataset.save_to_disk(f'../data_of_ReGPT/{corpus_name}/sorted_datasets_{split}')

# preprocess_dataset(corpus_name, f'../data_of_ReGPT/{corpus_name}/base_data_128.txt', split='train')
# if os.path.exists(f'../data_of_ReGPT/{corpus_name}/test.txt'):
    # preprocess_dataset(corpus_name, f'../data_of_ReGPT/{corpus_name}/test.txt', split='test')

preprocess_qa_dataset(corpus_name, f'../{corpus_name}/sorted_datasets_{split}', split=split)

def preprocess_hotpotqa(split_name='validation'):
    data = load_from_disk('hotpotqa')
    split = data[split_name]
    split = split.remove_columns(['id','type','level','supporting_facts','context'])
    split = split.map(add_text_length)
    split = split.sort("input_ids_length", reverse=True)
    split.save_to_disk(f'data_of_ReGPT/hotpotqa/{split_name}')

def preprocess_2WikiMultihopQA(split_name='dev'):
    data = load_dataset('xanhho/2WikiMultihopQA')
    split = data[split_name]
    print(split)
    split = split.remove_columns(['_id', 'type', 'context', 'supporting_facts', 'evidences'])
    split = split.map(add_text_length)
    split = split.sort("input_ids_length", reverse=True)
    split.save_to_disk(f'data_of_ReGPT/2WikiMultihopQA/{split_name}')


def format_consistency(example):
    example["answer"] = example['answers'][0]
    example['question'] = example['query']
    example.pop('answers')
    example.pop('query')
    return example
def preprocess_nq(split_name='dev'):
    data = load_from_disk('wikipedia-nq')
    split = data[split_name]
    split = split.remove_columns(['query_id','positive_passages','negative_passages'])
    split = split.map(add_text_length)
    split = split.sort("input_ids_length", reverse=True)
    split = split.map(format_consistency)
    split.save_to_disk(f'data_of_ReGPT/nq/{split_name}')

from datasets import concatenate_datasets, load_dataset
train1 = load_from_disk('../data_of_ReGPT/hotpotqa/sorted_datasets_train/')
train2 = load_from_disk('../data_of_ReGPT/2WikiMultihopQA/sorted_datasets_train/')
merged = concatenate_datasets([train1, train2])
merged = merged.sort("input_ids_length", reverse=True)
merged.save_to_disk('../data_of_ReGPT/hotpotqaAnd2WikiMultihopQA/sorted_datasets_train')
test1 = load_from_disk('../data_of_ReGPT/hotpotqa/sorted_datasets_validation/')
test2 = load_from_disk('../data_of_ReGPT/2WikiMultihopQA/sorted_datasets_dev/')
test1 = test1.select(range(1000))
test2 = test2.select(range(1000))
test1.save_to_disk('../data_of_ReGPT/hotpotqa/sorted_datasets_validation_sub')
test2.save_to_disk('../data_of_ReGPT/2WikiMultihopQA/sorted_datasets_dev_sub')
merged = concatenate_datasets([test1, test2])
merged = merged.sort("input_ids_length", reverse=True)
merged.save_to_disk('../data_of_ReGPT/hotpotqaAnd2WikiMultihopQA/sorted_datasets_test')
