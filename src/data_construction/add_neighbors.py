from datasets import load_from_disk
import numpy as np

def add_neighbors_to_dataset(dataset_path, neighbors_file_path):
    # 加载数据集
    query = load_from_disk(dataset_path)
    
    # 读取 neighbors 文件并构建 qid2pids 字典
    with open(neighbors_file_path, 'r') as f:
        qid2pids = {}
        for line in f:
            qid, pid, _, _ = line.split('\t')
            qid, pid = int(qid), int(pid)
            tmp = qid2pids.get(qid, [])
            tmp.append(pid)
            qid2pids[qid] = tmp

    # 定义添加邻居的函数
    def add_neighbors(example):
        qid = example['query_id']
        example['neighbors'] = qid2pids.get(qid, [])
        # 如果你需要添加邻居的嵌入，可以解除注释以下行并设置 embeddings
        # example['neighbor_embeddings'] = embeddings[example['neighbors']]
        return example

    # 应用 add_neighbors 函数到数据集中
    query = query.map(add_neighbors)
    
    return query

if __name__ == '__main__':
    dataset_paths = ['../data_of_ReGPT/QA_datasets_woEmb/2WikiMultihopQA/sorted_datasets_train/', '../data_of_ReGPT/QA_datasets_woEmb/2WikiMultihopQA/sorted_datasets_dev/', '../data_of_ReGPT/QA_datasets_woEmb/nq/sorted_datasets_train/', '../data_of_ReGPT/QA_datasets_woEmb/nq/sorted_datasets_test/']
    corpus_names = ['2WikiMultihopQA-train', '2WikiMultihopQA-dev', 'nq-train', 'nq-test']
    for dataset_path, corpus_name in dataset_paths, corpus_names:
        # 调用函数
        neighbors_file_path = f'./output/res.top50.step0.{corpus_name}'
        query_with_neighbors = add_neighbors_to_dataset(dataset_path, neighbors_file_path)
        query_with_neighbors.save_to_disk(dataset_path.replace('QA_datasets_woEmb', 'QA_datasets_wTop50Emb'))
