import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
phrase_embeddings = []
batch_size = 8
corpus_name = "WikiText-103"
phrases = np.load(open(f"phrases_{corpus_name}.npy",'rb'))
phrases = phrases.tolist()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
for i in tqdm(range(0,len(phrases), batch_size)):
    texts = phrases[i:i+batch_size]
    batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch_tokens.to("cuda")
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).hidden_states[-1]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
    embeddings = sum_embeddings / sum_mask
    phrase_embeddings.append(embeddings.cpu().detach().numpy())
    
phrase_embeddings = np.concatenate(phrase_embeddings)
np.save(open(f"phrases_embeddings_{corpus_name}.npy",'wb'),phrase_embeddings)

corpus = torch.from_numpy(phrase_embeddings)
norms = torch.norm(corpus, p=2, dim=1)
normalized_vectors = corpus / norms.view(-1, 1)
normalized_vectors = normalized_vectors.numpy()
np.save(open(f"phrases_embeddings_{corpus_name}_normalized.npy",'wb'), normalized_vectors)


