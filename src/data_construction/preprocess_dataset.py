import os
import sys
from datasets import load_dataset
corpus_name = sys.argv[1]

def filter_empty(example):
    return len(example['text']) > 0

def add_text_length(example):
    example["text_length"] = len(example["text"].split())
    return example

def filter_short_and_long_text(example):
    return example['text_length'] >= 64 and example['text_length'] <= 128

def preprocess_dataset(corpus_name, data_path, split='train'):
    dataset = load_dataset('csv', data_files={split: data_path}, delimiter='\t',column_names=['text', 'id'])

    dataset = dataset.filter(filter_empty)

    dataset = dataset.map(add_text_length)

    dataset = dataset.filter(filter_short_and_long_text)

    dataset = dataset.sort("text_length", reverse=True)

    dataset = dataset[split]

    dataset.save_to_disk(f'../data_of_ReGPT/{corpus_name}/sorted_datasets_{split}')

preprocess_dataset(corpus_name, f'../data_of_ReGPT/{corpus_name}/base_data_128.txt', split='train')
if os.path.exists(f'../data_of_ReGPT/{corpus_name}/test.txt'):
    preprocess_dataset(corpus_name, f'../data_of_ReGPT/{corpus_name}/test.txt', split='test')
