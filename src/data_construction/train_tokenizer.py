from transformers import AutoTokenizer
corpus_name = "WikiText-103"
model_name_or_path = 'gpt2'
old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': f'{corpus_name}/base_data_128.txt'}, delimiter='\t',column_names=['text', 'id'])
def data_loader(dataset, batch_size=1000):
    batch = []
    for example in dataset['train']:
        if type(example['text']) is str:
            batch.append(example['text'])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

training_corpus = data_loader(dataset)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 500000, show_progress=True)
tokenizer.save_pretrained(f"{model_name_or_path}-phrase-tokenizer-trained-on-{corpus_name}/")

def get_phrase():
    phrase = []
    for token_id in range(tokenizer.vocab_size):
        phrase.append(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([token_id])))
    return phrase

phrases = get_phrase()
import numpy as np
import os
os.makedirs(f"../phrases_{corpus_name}_{model_name_or_path}", exist_ok=True)
np.save(open(f"phrases_{corpus_name}_{model_name_or_path}/phrases.npy",'wb'), phrases)

