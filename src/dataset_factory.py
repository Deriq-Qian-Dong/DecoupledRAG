import os
import re
import torch
import random
import numpy as np
from utils import print_rank_0
from torch.utils.data import DataLoader, Dataset, BatchSampler, DistributedSampler, Sampler
from transformers import DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset, load_from_disk
import math
from registry import register_class
from prompt_templates import QA_PROMPT
try:
    from pyserini.search.lucene import LuceneSearcher
except:
    LuceneSearcher = None
class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, max_tokens, num_replicas, rank):
        dataset.datasets = dataset.datasets.shard(num_shards=num_replicas, index=rank)
        self.max_tokens = max_tokens
        dataset.update_total_tokens()
        total_tokens = dataset.total_tokens
        self.dataset = dataset
        self.num_samples = math.ceil(total_tokens / self.max_tokens)

    def __iter__(self):
        batch = []
        current_batch_tokens = 0
        max_seq_len_in_batch = 0
        batches_yielded = 0
        
        for idx in range(len(self.dataset)):
            seq_len = min(self.dataset[idx][-1], self.dataset.args['max_seq_len'])
            max_seq_len_in_batch = max(max_seq_len_in_batch, seq_len)
            if current_batch_tokens + max_seq_len_in_batch > self.max_tokens:
                if batch:
                    yield batch
                    batches_yielded += 1
                    if batches_yielded >= self.num_samples:
                        break
                batch = []
                max_seq_len_in_batch = 0
                current_batch_tokens = 0
            batch.append(idx)
            current_batch_tokens += max_seq_len_in_batch

        if batch and (batches_yielded < self.num_samples):
            yield batch

    def __len__(self):
        return self.num_samples
    
@register_class
class RAGPretrainDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.epoch = 0
        self.corpus = load_from_disk(self.args['corpus'])
        self.setup_datasets()

    def add_input_ids(self, example):
        psg = example['text']
        example['input_ids'] = self.tokenizer(psg).input_ids
        example['input_ids_length'] = len(example['input_ids'])
        return example

    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        # self.datasets = self.datasets.map(self.add_input_ids)
        # self.datasets = self.datasets.flatten_indices()
        # self.datasets = self.datasets.sort('input_ids_length', reverse=True)
        # try:
        #     self.datasets = self.datasets.select(range(2205416-10000,2205416))
        # except:
        #     pass
        self.num_samples = len(self.datasets)
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
    
    def update_total_tokens(self):
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
        self.num_samples = len(self.datasets)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        print_rank_0(f'[!] set epoch to {epoch}')

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = sample['text']
        neighbor_embeddings = sample['neighbor_embeddings']
        # recover from the neighbors
        # neighbors = sample['neighbors'][1:]
        # copy itself
        neighbors = sample['neighbors']
        chat = [{'role': 'user', 'content': "Recover the knowledge:"}, {'role': 'assistant', 'content': text}]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        neighbor_texts = [self.corpus[neighbor]['text'] for neighbor in neighbors]
        return text, neighbor_embeddings, neighbor_texts, sample['input_ids_length']

    def __len__(self):
        return len(self.datasets)

    def _collate_fn(self, elems):
        texts, neighbor_embeddings, neighbor_texts, _ = zip(*elems)
        batch = self.tokenizer(texts,
                                add_special_tokens=False,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        # num_of_elements = random.randint(1, len(neighbor_texts[0]))
        # random_indices = random.sample(range(len(neighbor_texts[0])), num_of_elements)
        random_indices = [0]
        all_neighbor_texts = []
        for neighbor_text in neighbor_texts:
            all_neighbor_texts += [neighbor_text[i] for i in random_indices]
        neighbor_batch = self.tokenizer(all_neighbor_texts,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids']
        # ret_pos = batch['input_ids'].size(1) // 2
        ret_pos = 0
        shape = (batch['input_ids'].size(0), 1)
        batch['retrieval_position'] = torch.full(shape, ret_pos, dtype=torch.long)
        batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
        batch['knowledge_input_ids'] = neighbor_batch['input_ids']
        return batch

@register_class
class RAGPretrainFromAFSDataset(RAGPretrainDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

    def setup_datasets(self):
        data_name_or_path = self.args['data_name_or_path']
        self.epoch = self.epoch%self.args['num_epochs']
        data_name_or_path = data_name_or_path.format(self.epoch)
        print_rank_0(f'[!] load dataset from {data_name_or_path}')
        self.datasets = load_dataset('arrow', data_files=data_name_or_path, split='train')
        self.num_samples = len(self.datasets)
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.setup_datasets()
        print_rank_0(f'[!] set epoch to {epoch}')

@register_class
class DialogSFTDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.epoch = 0
        self.setup_datasets()

    def setup_datasets(self):
        self.split = self.args['train_or_test']
        self.datasets = load_dataset(self.args['data_name_or_path'], split=self.split)
        self.num_samples = len(self.datasets)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        print_rank_0(f'[!] set epoch to {epoch}')

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = sample['prompt'] + sample['chosen']
        return text

    def __len__(self):
        return self.num_samples

    def _collate_fn(self, elems):
        batch = self.tokenizer(elems,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids']
        return batch

@register_class
class CorpusPretrainDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)        
        
    def setup_datasets(self):
        data_name_or_path = self.args['data_name_or_path']
        self.datasets = load_from_disk(data_name_or_path)
        # self.datasets = self.datasets.filter(self.filter_empty)
        self.num_samples = len(self.datasets)

    def filter_empty(self, example):
        return example['text_length'] >= 10
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = sample['text'].replace("<|endoftext|>", "")
        # filtering the non-English characters except the punctuation and digits
        # text = re.sub(r"[^a-zA-Z0-9',.?!]", " ", text)
        # text = re.sub(r"\s+", " ", text)  # remove extra spaces
        text = text.strip()  # remove leading and trailing spaces
        return text

@register_class
class CorpusPretrainFromAfsDataset(CorpusPretrainDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)        
        
    def setup_datasets(self):
        data_name_or_path = self.args['data_name_or_path']
        data_name_or_path = data_name_or_path.format(self.epoch)
        print_rank_0(f'[!] load dataset from {data_name_or_path}')
        self.datasets = load_dataset('arrow', data_files=data_name_or_path, split='train')
        # self.datasets = self.datasets.filter(self.filter_empty)
        self.num_samples = len(self.datasets)
    
    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.setup_datasets()

@register_class
class LongDocumentSummarizationSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)        
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = "Please write an abstract for this article:\n"+sample['article']+"\nAbstract:\n"+sample['abstract']
        return text

@register_class
class DocumentSummarizationSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)        
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = "Please write an abstract for this article:\n"+sample['article']+"\nAbstract:\n"+sample['highlights']
        return text
    
    def setup_datasets(self):
        self.split = self.args['train_or_test']
        if os.path.exists(self.args['data_name_or_path']):
            self.datasets = load_from_disk(self.args['data_name_or_path'])[self.split]
        else:
            self.datasets = load_dataset(self.args['data_name_or_path'], '3.0.0',  split=self.split)
        self.num_samples = len(self.datasets)

@register_class
class ReGPTDialogSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.negatives = np.loadtxt(args['negative_path']).reshape((-1, args['negative_depth_in_pool'], 4))
        # 将最后一列大于FNTP_threshold的位置的第二列位置替换为0到self.negatives.shape[0]的随机数，防止假负例过多
        mask = self.negatives[:, :, 3] > self.args['FNTP_threshold']
        self.negatives[mask, 1] = np.random.randint(0, self.negatives.shape[0], size=mask.sum())
    
    def _collate_fn(self, elems):
        batch = super()._collate_fn(elems)
        input_ids = batch['input_ids'].numpy()
        predict_from_last = self.args['predict_from_last']
        predict_from_last = min(predict_from_last, len(input_ids[0])-1)  # 防止predict_from_last大于句子长度
        negative_infos = self.negatives[input_ids[:,-predict_from_last:]] # (batch_size, predict_from_last, negative_depth_in_pool, 4)
        negative_ids = negative_infos[:, :, :, 1] # (batch_size, predict_from_last, negative_depth_in_pool)
        # 从负样本中随机选取self.args['negative_depth']个
        negative_ids = negative_ids[:, :, np.random.choice(negative_ids.shape[2], self.args['negative_depth'], replace=False)].astype(int) # (batch_size, predict_from_last, negative_depth)
        # 使用negs_of_eos填充
        batch['negative_ids'] = torch.from_numpy(negative_ids).long()
        return batch

@register_class
class ReGPTCorpusPretrainDataset(ReGPTDialogSFTDataset, CorpusPretrainDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return CorpusPretrainDataset.__getitem__(self, idx)
    
    def setup_datasets(self):
        CorpusPretrainDataset.setup_datasets(self)

@register_class
class ReGPTCorpusPretrainFromAfsDataset(ReGPTDialogSFTDataset, CorpusPretrainFromAfsDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return CorpusPretrainFromAfsDataset.__getitem__(self, idx)
    
    def setup_datasets(self):
        CorpusPretrainFromAfsDataset.setup_datasets(self)

@register_class
class ReGPTLongDocumentSummarizationSFTDataset(ReGPTDialogSFTDataset, LongDocumentSummarizationSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return LongDocumentSummarizationSFTDataset.__getitem__(self, idx)

@register_class
class ReGPTDocumentSummarizationSFTDataset(ReGPTDialogSFTDataset, DocumentSummarizationSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return DocumentSummarizationSFTDataset.__getitem__(self, idx)
    
    def setup_datasets(self):
        DocumentSummarizationSFTDataset.setup_datasets(self)

@register_class
class QADataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.setup_datasets()
        self.qa_prmt = QA_PROMPT
    
    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        self.num_samples = len(self.datasets)
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        if 'query' in sample:
            query = sample['query']
        else:
            query = sample['question']
        query = self.qa_prmt.format(question=query)
        if 'answers' in sample:
            answer = sample['answers'][0]
        else:
            answer = sample['answer']
        neighbor_embeddings = sample.get('neighbor_embeddings')
        return query, answer, neighbor_embeddings, sample['input_ids_length']
    
    def __len__(self):
        return self.num_samples
    
    def _collate_fn(self, elems):
        qrys, anss, neighbor_embeddings, _ = zip(*elems)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        batch = self.tokenizer(qrys, anss,
                                    max_length=self.args['max_seq_len'],
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt")
        ans_lens = [len(self.tokenizer(ans)['input_ids']) for ans in anss]
        batch["labels"] = batch['input_ids'].clone()
        for i in range(len(batch['labels'])):
            batch['labels'][i, :-ans_lens[i]] = -100
        retrieval_positions = []
        seq_len = batch['input_ids'].size(1)
        for i in range(len(batch['labels'])):
            ans_len = ans_lens[i]
            retrieval_position = seq_len - ans_len
            if retrieval_position <= 0:
                retrieval_position = seq_len//2
            retrieval_positions.append(retrieval_position)
        batch['retrieval_position'] = torch.tensor(retrieval_positions).reshape(-1, 1)
        batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
        return batch
    
    def filter_empty(self, example):
        return len(example['answers']) > 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        print_rank_0(f'[!] set epoch to {epoch}')

    def update_total_tokens(self):
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
        self.num_samples = len(self.datasets)
    
@register_class
class QADataset4Chat(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.inference_with_explict_docs_for_test = args['inference_with_explict_docs_for_test']
        self.number_of_docs = args['number_of_docs']
        print('number_of_docs:', self.number_of_docs, 'path:', args['data_name_or_path'])
        self.tokenizer = tokenizer
        self.setup_datasets()
        self.corpus = load_from_disk(args['corpus'])
        # self.searcher = LuceneSearcher(self.args['index_path'])
        # self.searcher.set_bm25(0.82, 0.68)
    
    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        # # 将datasets顺序前后颠倒
        # self.datasets = self.datasets.select(range(len(self.datasets)-1, -1, -1))
        # # flantten the datasets
        # self.datasets = self.datasets.flatten_indices()
        self.num_samples = len(self.datasets)
        # input_ids_lengths = self.datasets['input_ids_length']
        # input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        # self.total_tokens = sum(input_ids_lengths)
    
    def __getitem__(self, idx):
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
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
        retrieved_docs = self.corpus[sample['neighbors']]['text'][:self.number_of_docs]
        neighbor_batch_input_ids = self.tokenizer(retrieved_docs,
                                max_length=64,
                                padding=True,
                                truncation=True).input_ids
        retrieved_docs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in neighbor_batch_input_ids]
        references = "References:\n"
        for doc in retrieved_docs:
            references += doc+'\n'
        if not self.inference_with_explict_docs_for_test:
            references = ''
        query = references + query + '\nThe answer MUST in ONE OR FEW WORDS.'
        chat = [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': answer}]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        neighbor_embeddings = None
        return chat, neighbor_embeddings, retrieved_docs, 0
    
    def __len__(self):
        return self.num_samples
    
    def _collate_fn(self, elems):
        texts, neighbor_embeddings, retrieved_docs, _ = zip(*elems)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        batch = self.tokenizer(texts,
                                add_special_tokens=False,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        all_retrieved_docs = []
        # number_of_docs = random.randint(1, 10)
        for docs in retrieved_docs:
            all_retrieved_docs += docs[:self.number_of_docs]
        neighbor_batch = self.tokenizer(all_retrieved_docs,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids']
        # batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
        batch['knowledge_input_ids'] = neighbor_batch['input_ids']
        if self.inference_with_explict_docs_for_test:
            batch['knowledge_input_ids'] = None
        return batch

    def filter_empty(self, example):
        return len(example['answers']) > 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        print_rank_0(f'[!] set epoch to {epoch}')

    def update_total_tokens(self):
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
        self.num_samples = len(self.datasets)

@register_class
class QADataset4ChatTest(QADataset4Chat):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        # 将datasets shard
        num_samples = len(self.datasets)
        # 1000 samples per shard
        num_shards = max(num_samples//1000, 1)
        self.datasets = self.datasets.shard(num_shards=num_shards, index=0)
        # flantten the datasets
        self.datasets = self.datasets.flatten_indices()
        self.num_samples = len(self.datasets)
        # input_ids_lengths = self.datasets['input_ids_length']
        # input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        # self.total_tokens = sum(input_ids_lengths)
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        if 'query' in sample:
            query = sample['query']
        else:
            query = sample['question']
        if 'answers' in sample:
            answers = sample['answers']
        else:
            answers = [sample['answer']]
        # hits = self.searcher.search(query, 5)
        retrieved_docs = self.corpus[sample['neighbors']]['text'][:self.number_of_docs]
        neighbor_batch_input_ids = self.tokenizer(retrieved_docs,
                                max_length=64,
                                padding=True,
                                truncation=True).input_ids
        retrieved_docs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in neighbor_batch_input_ids]
        references = "References:\n"
        for doc in retrieved_docs:
            references += doc+'\n'
        if not self.inference_with_explict_docs_for_test:
            references = ''
        query = references + query+'\nThe answer MUST in ONE OR FEW WORDS.'
        chat = [{'role': 'user', 'content': query}]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        neighbor_embeddings = None
        return chat, answers, retrieved_docs, neighbor_embeddings, 0
    
    def _collate_fn(self, elems):
        texts, answers, retrieved_docs, neighbor_embeddings, _ = zip(*elems)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        batch = self.tokenizer(texts,
                                add_special_tokens=False,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        all_retrieved_docs = []
        for docs in retrieved_docs:
            all_retrieved_docs += docs
        neighbor_batch = self.tokenizer(all_retrieved_docs,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch['knowledge_input_ids'] = neighbor_batch['input_ids']
        if self.inference_with_explict_docs_for_test:
            batch['knowledge_input_ids'] = None
        batch['answers'] = answers
        return batch

    def set_epoch(self, epoch):
        pass

@register_class
class QADataset4ChatTestwHiddenStates(QADataset4Chat):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        # 将datasets shard
        num_samples = len(self.datasets)
        # 1000 samples per shard
        num_shards = max(num_samples//1000, 1)
        self.datasets = self.datasets.shard(num_shards=num_shards, index=0)
        # flantten the datasets
        self.datasets = self.datasets.flatten_indices()
        self.num_samples = len(self.datasets)
        # input_ids_lengths = self.datasets['input_ids_length']
        # input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        # self.total_tokens = sum(input_ids_lengths)
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        if 'query' in sample:
            query = sample['query']
        else:
            query = sample['question']
        if 'answers' in sample:
            answers = sample['answers']
        else:
            answers = [sample['answer']]
        # hits = self.searcher.search(query, 5)
        retrieved_docs = self.corpus[sample['neighbors']]['text'][:self.number_of_docs]
        neighbor_batch_input_ids = self.tokenizer(retrieved_docs,
                                max_length=64,
                                padding=True,
                                truncation=True).input_ids
        retrieved_docs = [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in neighbor_batch_input_ids]
        references = "References:\n"
        for doc in retrieved_docs:
            references += doc+'\n'
        if not self.inference_with_explict_docs_for_test:
            references = ''
        query = references + query+'\nThe answer MUST in ONE OR FEW WORDS.'
        chat = [{'role': 'user', 'content': query}]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        neighbor_embeddings = np.array(sample['neighbors_hidden_states'])
        return chat, answers, retrieved_docs, neighbor_embeddings, 0
    
    def _collate_fn(self, elems):
        texts, answers, retrieved_docs, neighbor_embeddings, _ = zip(*elems)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        batch = self.tokenizer(texts,
                                add_special_tokens=False,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        all_retrieved_docs = []
        for docs in retrieved_docs:
            all_retrieved_docs += docs
        neighbor_batch = self.tokenizer(all_retrieved_docs,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch['knowledge_input_ids'] = neighbor_batch['input_ids']
        if self.inference_with_explict_docs_for_test:
            batch['knowledge_input_ids'] = None
        batch['answers'] = answers
        batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
        return batch

    def set_epoch(self, epoch):
        pass

@register_class
class QADataset4Contrastive(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.setup_datasets()
    
    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        self.num_samples = len(self.datasets)
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        chat = sample['prompt']
        retrieved_docs = [sample['background_knowledge']]
        return chat, retrieved_docs
    
    def __len__(self):
        return self.num_samples
    
    def _collate_fn(self, elems):
        texts, retrieved_docs = zip(*elems)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        batch = self.tokenizer(texts,
                                add_special_tokens=False,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        all_retrieved_docs = []
        for docs in retrieved_docs:
            all_retrieved_docs += docs
        neighbor_batch = self.tokenizer(all_retrieved_docs,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids']
        batch['knowledge_input_ids'] = neighbor_batch['input_ids']
        return batch
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        print_rank_0(f'[!] set epoch to {epoch}')

@register_class
class QASFTDataset(QADataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        self.datasets = self.datasets.sort('input_ids_length', reverse=False)
        # self.datasets = self.datasets.select(range(len(self.datasets)//2))
        self.num_samples = len(self.datasets)
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)

    def _collate_fn(self, elems):
        qrys, anss, neighbor_embeddings, _ = zip(*elems)
        self.tokenizer.padding_side = 'right'  # fix weired overflow bug
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        qa_pairs = [f"{qry}{ans}{self.tokenizer.eos_token}" for qry, ans in zip(qrys, anss)]
        batch = self.tokenizer(qa_pairs,
                                    max_length=self.args['max_seq_len'],
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt")
        ans_lens = [len(self.tokenizer(ans)['input_ids']) for ans in anss]
        batch["labels"] = batch['input_ids'].clone()
        retrieval_positions = []
        if self.tokenizer.padding_side=='right':
            seq_lens = batch['attention_mask'].sum(dim=1).tolist()
        else:
            seq_lens = [batch['input_ids'].size(1)] * len(batch['input_ids'])
        for i in range(len(batch['labels'])):
            ans_len = ans_lens[i]
            seq_len = seq_lens[i]
            retrieval_position = seq_len - ans_len
            if retrieval_position <= 0:
                retrieval_position = seq_len//2
            retrieval_positions.append(retrieval_position)
            batch['labels'][i, :retrieval_position] = -100
            batch['labels'][i, seq_len:] = -100
        batch['retrieval_position'] = torch.tensor(retrieval_positions).reshape(-1, 1)
        batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
        return batch

@register_class
class QAEvalDataset(QADataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        # 仅保留前1000个样本
        if len(self.datasets) > 1000:
            # 随机选取1000个样本
            shard_num = len(self.datasets) // 1000
            self.datasets = self.datasets.shard(num_shards=shard_num, index=0)
        self.update_total_tokens()
    
    def _collate_fn(self, elems):
        qrys, anss, _, _ = zip(*elems)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        batch = self.tokenizer(qrys,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch['answers'] = self.tokenizer(anss,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")['input_ids']
        return batch

@register_class
class TrufulQADataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.datasets = load_from_disk(args['data_name_or_path'])
        self.num_samples = len(self.datasets)
        self.tokenizer = tokenizer
        self.qa_prmt = QA_PROMPT

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        query = sample['question']
        query = self.qa_prmt.format(question=query)
        answers = sample['mc1_targets']['choices']
        labels = sample['mc1_targets']['labels']
        return query, answers, labels
    
    def __len__(self):
        return self.num_samples
    
    def _collate_fn(self, elems):
        qrys, anss, labels = zip(*elems)
        ans_lens = []
        pairs = []
        targets = []
        idxs = []
        idx = -1
        for qry, ans, target in zip(qrys, anss, labels):
            targets+=target
            idx+=1
            for i in range(len(ans)):
                qa_pair = qry + ans[i]
                pairs.append(qa_pair)
                ans_lens.append(len(self.tokenizer(ans[i])['input_ids']))
                idxs.append(idx)
        batch = self.tokenizer(pairs,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids'].clone()
        for i in range(len(batch['labels'])):
            batch['labels'][i, :-ans_lens[i]] = -100
        batch['targets'] = torch.tensor(targets)
        batch['idxs'] = torch.tensor(idxs)
        return batch

@register_class
class OpenBookQADataset(TrufulQADataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        query = sample['question_stem']
        query = self.qa_prmt.format(question=query)
        answers = sample['choices']['text']
        labels = [int(l==sample['answerKey']) for l in sample['choices']['label']]
        return query, answers, labels

@register_class
class QueryDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.args = args
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        try:
            self.collection = load_from_disk(args.dev_query)['query']
        except:
            self.collection = load_from_disk(args.dev_query)['question']
        total_cnt = len(self.collection)
        shard_cnt = total_cnt//self.n_procs
        if self.rank!=self.n_procs-1:
            self.collection = self.collection[self.rank*shard_cnt:(self.rank+1)*shard_cnt]
        else:
            self.collection = self.collection[self.rank*shard_cnt:]
        self.num_samples = len(self.collection)
        
    def _collate_fn(self, qrys):
        return self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)

    def __getitem__(self, idx):
        return self.collection[idx]

    def __len__(self):
        return self.num_samples

@register_class
class PassageDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
        self.args = args
        self.collection = load_from_disk(args.collection)['text']
        total_cnt = len(self.collection)
        shard_cnt = total_cnt//self.n_procs
        if self.rank!=self.n_procs-1:
            self.collection = self.collection[self.rank*shard_cnt:(self.rank+1)*shard_cnt]
        else:
            self.collection = self.collection[self.rank*shard_cnt:]
        self.num_samples = len(self.collection)
        print('rank:',self.rank,'samples:',self.num_samples)

    def _collate_fn(self, psgs):
        p_records = self.tokenizer(psgs, padding=True, truncation=True, return_tensors="pt", max_length=self.args.p_max_seq_len)
        return p_records

    def __getitem__(self, idx):
        psg = self.collection[idx]
        return psg

    def __len__(self):
        return self.num_samples
