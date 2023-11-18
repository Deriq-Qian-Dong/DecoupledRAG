import os

os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

corpus_name = "WikiText-103"
model_name_or_path = "facebook/contriever"

os.makedirs(f"../phrases_{corpus_name}_{model_name_or_path}", exist_ok=True)

phrase_embeddings = []
batch_size = 128

phrases = np.load(open(f"../phrases_{corpus_name}_{model_name_or_path}/phrases.npy",'rb'))
phrases = phrases.tolist()
model = AutoModel.from_pretrained(model_name_or_path)
model.cuda()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
for i in tqdm(range(0, len(phrases), batch_size)):
    texts = phrases[i:i+batch_size]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    phrase_embeddings.append(embeddings.cpu().detach().numpy())
    
phrase_embeddings = np.concatenate(phrase_embeddings)
np.save(open(f"../phrases_{corpus_name}_{model_name_or_path}/phrases_embeddings.npy",'wb'), phrase_embeddings)

corpus = torch.from_numpy(phrase_embeddings)
norms = torch.norm(corpus, p=2, dim=1)
normalized_vectors = corpus / norms.view(-1, 1)
normalized_vectors = normalized_vectors.numpy()
np.save(open(f"../phrases_{corpus_name}_{model_name_or_path}/phrases_embeddings_normalized.npy",'wb'), normalized_vectors)
