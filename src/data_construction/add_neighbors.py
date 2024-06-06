from datasets import load_from_disk
import numpy as np
query = load_from_disk('../data_of_ReGPT/msmarco_qa/sorted_datasets_train/')
data = load_from_disk('../data_of_ReGPT/marco/collection/')
f = open('./output/res.top10.step0.InContextExamples').readlines()
qid2pids = {}
for line in f:
    qid,pid,_,_ = line.split('\t')
    qid,pid = int(qid), int(pid)
    tmp = qid2pids.get(qid, [])
    tmp.append(pid)
    qid2pids[qid] = tmp


embeddings = np.load('../data_of_ReGPT/marco/phrases_embeddings.npy')


def add_neighbors(example):
    qid = example['query_id']
    example['neighbors'] = qid2pids[qid]
    example['neighbor_embeddings'] = embeddings[example['neighbors']]
    return example


query = query.map(add_neighbors)