import sys
import torch
import numpy as np
from tqdm import tqdm

corpus_name = 'original'
vocab_size = 32000
negative_depth = 101

base_dir = f'../data_of_ReGPT/phrases_{corpus_name}_repllama'
embedding_path = f"{base_dir}/phrases_embeddings_normalized.npy"
print('load phrase embeddings from', embedding_path)
phrase_embeddings = np.load(open(embedding_path,'rb'))
corpus = torch.from_numpy(phrase_embeddings)
corpus = corpus.cuda()

def topk_query_passage(query_vector, passage_vector, k):
    """
    对query vector和passage vector进行内积计算，并返回top k的索引

    Args:
        query_vector (torch.Tensor): query向量，形状为 (batch_size, query_dim)
        passage_vector (torch.Tensor): passage向量，形状为 (batch_size, passage_dim)
        k (int): 返回的top k值

    Returns:
        torch.Tensor: top k值的索引，形状为 (batch_size, k)
    """
    # 计算query向量和passage向量的内积
    scores = torch.matmul(query_vector, passage_vector.t())  # 形状为 (batch_size, batch_size)



    # 对每个batch进行排序，取top k值
    res_dist, res_p_id = torch.topk(scores, k=k, dim=1)  # 形状为 (batch_size, k)

    return res_dist.cpu().numpy(), res_p_id.cpu().numpy()

def read_embed(file_name, bs=100):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        with tqdm(total=len(emb_np)//bs+1) as pbar:
            while(i < len(emb_np)):
                vec_list = emb_np[i:i+bs]
                i += bs
                pbar.update(1)
                yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in tqdm(inp):
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list

def search(index, emb_file, qid_list, outfile, top_k):
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in read_embed(emb_file):
            q_emb_matrix = np.array(batch_vec)
            q_emb_matrix = torch.from_numpy(q_emb_matrix)
            q_emb_matrix = q_emb_matrix.cuda()
            res_dist, res_p_id = topk_query_passage(q_emb_matrix, index, top_k)
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, j+1, score))
                q_idx += 1

search(corpus, embedding_path, list(range(vocab_size)), f"{base_dir}/negatives.tsv", negative_depth)
