import os
import sys
os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

encoder_model_name_or_path = "../data/RetroMAE_MSMARCO_distill/"
# encoder_model_name_or_path = sys.argv[2]

base_dir = f'../data_of_ReGPT/marco'
os.makedirs(base_dir, exist_ok=True)

phrase_embeddings = []
batch_size = 2048

phrases = []
with open("../data_of_ReGPT/marco/collection.tsv") as f:
    lines = f.readlines()
for line in lines:
    phrases.append(line.split("\t")[1].strip())
print('load model from', encoder_model_name_or_path)
model = AutoModel.from_pretrained(encoder_model_name_or_path)
model.cuda()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name_or_path)

for i in tqdm(range(0, len(phrases), batch_size)):
    texts = phrases[i:i+batch_size]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:,0,:]
    phrase_embeddings.append(embeddings.cpu().detach().numpy())
    
phrase_embeddings = np.concatenate(phrase_embeddings)
np.save(open(f"{base_dir}/phrases_embeddings.npy",'wb'), phrase_embeddings)

# corpus = torch.from_numpy(phrase_embeddings)
# norms = torch.norm(corpus, p=2, dim=1)
# normalized_vectors = corpus / norms.view(-1, 1)
# normalized_vectors = normalized_vectors.numpy()
# np.save(open(f"{base_dir}/phrases_embeddings_normalized.npy",'wb'), normalized_vectors)
