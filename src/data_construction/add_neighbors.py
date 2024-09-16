from datasets import load_from_disk
import numpy as np

def add_neighbors_to_dataset(dataset_path, neighbors_file_path):
    # 加载数据集
    query = load_from_disk(dataset_path)
    # add query_id to examples
    for i, example in enumerate(query):
        example['query_id'] = i
    
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
    # dev_querys: 
    #     - ../data_of_ReGPT/new_qa_datasets/MuSiQue/train
    #     - ../data_of_ReGPT/new_qa_datasets/MuSiQue/validation
    #     - ../data_of_ReGPT/new_qa_datasets/complex_web_questions/train
    #     - ../data_of_ReGPT/new_qa_datasets/complex_web_questions/validation
    #     - ../data_of_ReGPT/new_qa_datasets/metaqa/train
    #     - ../data_of_ReGPT/new_qa_datasets/metaqa/test
    # dataset_paths = ['../data_of_ReGPT/QA_datasets_wTop10/nq/sorted_datasets_train/', '../data_of_ReGPT/QA_datasets_wTop10/nq/sorted_datasets_test/',
    #                  '../data_of_ReGPT/QA_datasets_wTop10/eli5/sorted_datasets_train/', '../data_of_ReGPT/QA_datasets_wTop10/eli5/sorted_datasets_test/',
    #                  '../data_of_ReGPT/QA_datasets_wTop10/msmarco_qa/sorted_datasets_train/', '../data_of_ReGPT/QA_datasets_wTop10/msmarco_qa/sorted_datasets_test/']
    dataset_paths = ['../data_of_ReGPT/new_qa_datasets/MuSiQue/train', '../data_of_ReGPT/new_qa_datasets/MuSiQue/validation',
                        '../data_of_ReGPT/new_qa_datasets/complex_web_questions/train', '../data_of_ReGPT/new_qa_datasets/complex_web_questions/validation',
                        '../data_of_ReGPT/new_qa_datasets/metaqa/train', '../data_of_ReGPT/new_qa_datasets/metaqa/test']
    # corpus_names = ['nq-train', 'nq-test', 'eli5-train', 'eli5-test', 'msmarco_qa-train', 'msmarco_qa-test']
    corpus_names = ['MuSiQue-train', 'MuSiQue-validation', 'complex_web_questions-train', 'complex_web_questions-validation', 'metaqa-train', 'metaqa-test']
    for dataset_path, corpus_name in zip(dataset_paths, corpus_names):
        print(f'Processing {corpus_name}...')
        # 调用函数
        neighbors_file_path = f'./output/res.top50.step0.{corpus_name}'
        query_with_neighbors = add_neighbors_to_dataset(dataset_path, neighbors_file_path)
        query_with_neighbors.save_to_disk(dataset_path.replace('QA_datasets_wTop10', 'QA_datasets_wTop50'))

