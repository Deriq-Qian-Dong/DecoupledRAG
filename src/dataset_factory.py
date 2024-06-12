import os
import re
import torch
import numpy as np
from utils import print_rank_0
from torch.utils.data import DataLoader, Dataset, BatchSampler, DistributedSampler, Sampler
from transformers import DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset, load_from_disk
import math

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
class RAGPretrainDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.epoch = 0
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
        if 'text' in sample:
            text = sample['text']
        else:
            text = sample['query']+"\n\nThe answer is:"+sample['answers'][0]
        neighbor_embeddings = sample['neighbor_embeddings']
        return text, neighbor_embeddings, sample['input_ids_length']

    def __len__(self):
        return len(self.datasets)

    def _collate_fn(self, elems):
        texts, neighbor_embeddings, _ = zip(*elems)
        batch = self.tokenizer(texts,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids']
        ret_pos = batch['input_ids'].size(1) // 2
        shape = (batch['input_ids'].size(0), 1)
        batch['retrieval_position'] = torch.full(shape, ret_pos, dtype=torch.long)
        batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
        return batch

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

class LongDocumentSummarizationSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)        
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = "Please write an abstract for this article:\n"+sample['article']+"\nAbstract:\n"+sample['abstract']
        return text
    
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
    
class ReGPTCorpusPretrainDataset(ReGPTDialogSFTDataset, CorpusPretrainDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return CorpusPretrainDataset.__getitem__(self, idx)
    
    def setup_datasets(self):
        CorpusPretrainDataset.setup_datasets(self)

class ReGPTCorpusPretrainFromAfsDataset(ReGPTDialogSFTDataset, CorpusPretrainFromAfsDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return CorpusPretrainFromAfsDataset.__getitem__(self, idx)
    
    def setup_datasets(self):
        CorpusPretrainFromAfsDataset.setup_datasets(self)

class ReGPTLongDocumentSummarizationSFTDataset(ReGPTDialogSFTDataset, LongDocumentSummarizationSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return LongDocumentSummarizationSFTDataset.__getitem__(self, idx)
    
class ReGPTDocumentSummarizationSFTDataset(ReGPTDialogSFTDataset, DocumentSummarizationSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
    def __getitem__(self, idx):
        return DocumentSummarizationSFTDataset.__getitem__(self, idx)
    
    def setup_datasets(self):
        DocumentSummarizationSFTDataset.setup_datasets(self)

class QADataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.setup_datasets()
    
    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
        self.num_samples = len(self.datasets)
        input_ids_lengths = self.datasets['input_ids_length']
        input_ids_lengths = [min(self.args['max_seq_len'], length) for length in input_ids_lengths]
        self.total_tokens = sum(input_ids_lengths)
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        query = sample['query']+"\n\nThe answer is:"
        answer = sample['answers'][0]
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
    

class QASFTDataset(QADataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
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
    
class QAEvalDataset(QADataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
    
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
    
class QueryDataset(Dataset):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)
        self.args = args
        self.collection = load_from_disk(args.dev_query)['query']
        self.num_samples = len(self.collection)
        
    def _collate_fn(self, qrys):
        return self.tokenizer(qrys, padding=True, truncation=True, return_tensors="pt", max_length=self.args.q_max_seq_len)

    def __getitem__(self, idx):
        return self.collection[idx]

    def __len__(self):
        return self.num_samples
    
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
