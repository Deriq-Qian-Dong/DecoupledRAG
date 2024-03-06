import os

os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig

corpus_name = "original"
model_name_or_path = "repllama"

os.makedirs(f"../phrases_{corpus_name}_{model_name_or_path}", exist_ok=True)

phrase_embeddings = []
batch_size = 512

phrases = np.load(open(f"../phrases_{corpus_name}_{model_name_or_path}/phrases.npy",'rb'))
phrases = phrases.tolist()
def get_model(peft_model_name, base_model_name_or_path):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model
model =get_model('../repllama', '../LLAMA-2-7B')
model.cuda()
tokenizer = AutoTokenizer.from_pretrained('../LLAMA-2-7B')
for i in tqdm(range(0, len(phrases), batch_size)):
    texts = phrases[i:i+batch_size]
    texts = [f'passage: {text}</s>' for text in texts]
    batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch_tokens.to("cuda")
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]
    embeddings = last_hidden_state[:,-1,:]
    phrase_embeddings.append(embeddings.cpu().detach().numpy())
    
phrase_embeddings = np.concatenate(phrase_embeddings)
np.save(open(f"../phrases_{corpus_name}_{model_name_or_path}/phrases_embeddings.npy",'wb'), phrase_embeddings)

corpus = torch.from_numpy(phrase_embeddings)
norms = torch.norm(corpus, p=2, dim=1)
normalized_vectors = corpus / norms.view(-1, 1)
normalized_vectors = normalized_vectors.numpy()
np.save(open(f"../phrases_{corpus_name}_{model_name_or_path}/phrases_embeddings_normalized.npy",'wb'), normalized_vectors)
