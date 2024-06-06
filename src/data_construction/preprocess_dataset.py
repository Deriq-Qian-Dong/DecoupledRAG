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
    dataset = load_from_disk(data_path)[split]
    tokenizer = AutoTokenizer.from_pretrained("output/SFT-best")
    def filter_empty(example):
        return len(example['answers']) > 0 and len(example['answers'][0])>0

    def add_text_length(example):
        example["input_ids_length"] = len(tokenizer(example["query"]+"\n\nThe answer is:", example["answers"][0])['input_ids'])
        return example
    
    dataset = dataset.filter(filter_empty)

    dataset = dataset.map(add_text_length)

    dataset = dataset.sort("text_length", reverse=True)

    dataset.save_to_disk(f'../data_of_ReGPT/{corpus_name}/sorted_datasets_{split}')

# preprocess_dataset(corpus_name, f'../data_of_ReGPT/{corpus_name}/base_data_128.txt', split='train')
# if os.path.exists(f'../data_of_ReGPT/{corpus_name}/test.txt'):
    # preprocess_dataset(corpus_name, f'../data_of_ReGPT/{corpus_name}/test.txt', split='test')

preprocess_qa_dataset(corpus_name, f'../{corpus_name}/sorted_datasets_{split}', split=split)
