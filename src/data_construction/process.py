import numpy as np
embds = np.load('../../data_of_ReGPT/marco/gpt2_embeddings.npy')
matrix = np.load('../../data_of_ReGPT/marco/neighbors.P2P.DPR34-1.top50.npy')
def add_neighbors(example):
    pid = example['pid']
    example['neighbors'] = matrix[pid][::10]
    example['neighbor_gpt2_embeddings'] = embds[example['neighbors']]
    return example
from datasets import load_from_disk
train_dataset = load_from_disk('../../data_of_ReGPT/marco/sorted_datasets_train_gpt2/')
test_dataset = load_from_disk('../../data_of_ReGPT/marco/sorted_datasets_test_gpt2//')
train_dataset = train_dataset.map(add_neighbors)
test_dataset = test_dataset.map(add_neighbors)
train_dataset.save_to_disk('train')
test_dataset.save_to_disk('test')
