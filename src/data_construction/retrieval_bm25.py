from pyserini.search.lucene import LuceneSearcher
from datasets import load_from_disk
from tqdm import tqdm
searcher = LuceneSearcher('../data_of_ReGPT/Wiki-corpus/bm25_index/')
searcher.set_bm25(0.82, 0.68)

data = load_from_disk('../data_of_ReGPT/QA_datasets/nq/sorted_datasets_test')
with open('./output/res.top5.nq.test', 'w') as f:
    for example in tqdm(data):
        query = example['question']
        hits = searcher.search(query, 5)
        for hit in hits:
            f.write(f"{example['question']}\t{hit.docid}\t{hit.score}\t{hit.text}\n")

    

