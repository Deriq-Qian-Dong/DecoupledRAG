import os
import sys
os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk

encoder_model_name_or_path = "../data/RetroMAE_MSMARCO_distill/"
# encoder_model_name_or_path = sys.argv[2]

base_dir = f'../data_of_ReGPT/En-Wiki'
os.makedirs(base_dir, exist_ok=True)

phrase_embeddings = []
batch_size = 20480

phrases = []
data = load_from_disk("../data_of_ReGPT/En-Wiki/sorted_datasets_train")
torch.distributed.init_process_group(backend="nccl", init_method='env://')
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
data = data.shard(num_shards=world_size, index=local_rank, contiguous=True)
print('local_rank', local_rank, 'world_size', world_size, 'data size', len(data))
for item in tqdm(data):
    phrases.append(item['text'])
# with open("../data_of_ReGPT/marco/collection.tsv") as f:
    # lines = f.readlines()
# for line in lines:
    # phrases.append(line.split("\t")[1].strip())
print('load model from', encoder_model_name_or_path)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
model = AutoModel.from_pretrained(encoder_model_name_or_path)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name_or_path)

for i in tqdm(range(0, len(phrases), batch_size)):
    texts = phrases[i:i+batch_size]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:,0,:]
    phrase_embeddings.append(embeddings.cpu().detach().numpy())
    
phrase_embeddings = np.concatenate(phrase_embeddings)
np.save(open(f"{base_dir}/phrases_embeddings_{local_rank}.npy",'wb'), phrase_embeddings)
# wait for all processes to finish
torch.distributed.barrier()
if local_rank == 0:
    phrase_embeddings = []
    for i in range(world_size):
        phrase_embeddings.append(np.load(open(f"{base_dir}/phrases_embeddings_{i}.npy",'rb')))
    phrase_embeddings = np.concatenate(phrase_embeddings)
    np.save(open(f"{base_dir}/phrases_embeddings.npy",'wb'), phrase_embeddings)
    print('save phrase embeddings to', f"{base_dir}/phrases_embeddings.npy")

# corpus = torch.from_numpy(phrase_embeddings)
# norms = torch.norm(corpus, p=2, dim=1)
# normalized_vectors = corpus / norms.view(-1, 1)
# normalized_vectors = normalized_vectors.numpy()
# np.save(open(f"{base_dir}/phrases_embeddings_normalized.npy",'wb'), normalized_vectors)
