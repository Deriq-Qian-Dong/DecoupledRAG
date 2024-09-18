import os
import sys
import json
import torch
import numpy as np
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from tqdm import tqdm

def get_config(path="config/llama_config.yaml"):
    with open(path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

# 定义函数来计算 hidden states
def get_hidden_states(text_list, model, tokenizer):
    hidden_states = []
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64)
    # move inputs to cuda
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        knowledge_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
    for idx in range(len(knowledge_outputs)):
        hidden_states.append(knowledge_outputs[idx].detach().cpu().numpy())
    hidden_states = np.concatenate(hidden_states, axis=0)
    return hidden_states


config = get_config()

tokenizer = AutoTokenizer.from_pretrained(config['training']['model_name_or_path'], trust_remote_code=True)
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.eod_id
try:
    tokenizer.pad_token_id = tokenizer.eos_token_id
except:
    pass
tokenizer.truncation_side='left'
tokenizer.padding_side='left'
model = AutoModel.from_pretrained(config['training']['model_name_or_path'])
# cast model to fp16
model = model.half()
model = model.to("cuda")
model.eval()
for dataset_name in config['dataset']['test']:
    dataset = load_from_disk(config['dataset']['test'][dataset_name]['data_name_or_path'])
    num_samples = len(dataset)
    # 1000 samples per shard
    num_shards = max(num_samples//1000, 1)
    dataset = dataset.shard(num_shards=num_shards, index=0)
    # flantten the datasets
    dataset = dataset.flatten_indices()
    corpus = load_from_disk(config['dataset']['test'][dataset_name]['corpus'])
    # 对数据集进行处理，计算每个样本中 neighbors 列的 hidden states
    def add_hidden_states_to_dataset(example):
        neighbors = corpus[example['neighbors'][:20]]['text']  # 获取 neighbors 列
        example['neighbors_hidden_states'] = get_hidden_states(neighbors, model, tokenizer)  # 计算 hidden states 并存入新列
        return example
    dataset = dataset.map(add_hidden_states_to_dataset)
    dataset.save_to_disk(config['dataset']['test'][dataset_name]['data_name_or_path'].replace("QA_datasets_wTop50", "QA_datasets_wTop50wHiddenStates"))  # 保存处理后的数据集
