import os
from transformers import AutoTokenizer
from dataset_factory import CorpusPretrainDataset
from collections import Counter
from nltk.util import ngrams
from tqdm import tqdm
import numpy as np

os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

def main():
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    args = {"train_or_test": "train", "data_name_or_path": "../WikiText-103/sorted_datasets/", "max_seq_len": 128, 'split': 'train'}
    dataset = CorpusPretrainDataset(tokenizer, args)
    sorted_n_grams = []
    for n in range(2, 8):
        n_grams = []
        for i in tqdm(range(len(dataset))):
            tokens = dataset[i].split()
            sentence_n_grams = ngrams(tokens, n, pad_left=False, pad_right=False)
            n_grams.extend(sentence_n_grams)
        
        n_grams_counter = Counter(n_grams)
        sorted_n_grams.extend(list(n_grams_counter.items()))
        print(f"n={n} done")
    
    sorted_n_grams.sort(key=lambda x: x[1], reverse=True)
    with open('n_grams.txt', 'w') as f:
        for n_gram, count in sorted_n_grams:
            n_gram = ' '.join(n_gram)
            f.write(f"{n_gram}\t{count}\n")
    vocab_size = 1000000
    tokenizer_vocab_size = tokenizer.vocab_size
    tmp = sorted_n_grams[:vocab_size-tokenizer_vocab_size]
    tmp = ["▁"+"▁".join(t[0]) for t in tmp]
    num_added_tokens = tokenizer.add_tokens(tmp)
    tokenizer.save_pretrained('Llama2-phrase-tokenizer-WikiText-103')

def load_n_grams():
    n_grams = []
    with open('n_grams.txt', 'r') as f:
        for line in f:
            n_gram, count = line.strip().split('\t')
            n_grams.append(n_gram)
    return n_grams

if __name__ == "__main__":
    main()