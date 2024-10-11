from torch.utils.data import DataLoader, Dataset, BatchSampler, DistributedSampler, Sampler
from datasets import load_dataset, load_from_disk
class QADataset4Chat(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.number_of_docs = args['number_of_docs']
        print('number_of_docs:', self.number_of_docs, 'path:', args['data_name_or_path'])
        self.tokenizer = tokenizer
        self.setup_datasets()
        self.corpus = load_from_disk(args['corpus'])

    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        if 'query' in sample:
            query = sample['query']
        else:
            query = sample['question']
        if 'answers' in sample:
            answer = sample['answers'][0]
        else:
            answer = sample['answer']
        # hits = self.searcher.search(query, 5)
        retrieved_docs = self.corpus[sample['neighbors']]['text']
        # references = "references:\n"
        # for doc in retrieved_docs:
            # references += doc+'\n'
        query = query
        chat = [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': answer}]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return chat
    
    def __len__(self):
        return len(self.datasets)

