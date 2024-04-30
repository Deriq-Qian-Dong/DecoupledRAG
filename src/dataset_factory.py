import os
import re
import torch
import numpy as np
from utils import print_rank_0
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk

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
        self.num_samples = len(self.datasets)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        print_rank_0(f'[!] set epoch to {epoch}')

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = sample['text']
        neighbor_dr_embeddings = sample['neighbor_embeddings']
        neighbor_gpt_embeddings = sample.get('neighbor_gpt2_embeddings', None)
        return text, neighbor_dr_embeddings, neighbor_gpt_embeddings

    def __len__(self):
        return self.num_samples

    def _collate_fn(self, elems):
        texts, p_reps, neighbor_embeddings = zip(*elems)
        batch = self.tokenizer(texts,
                                max_length=self.args['max_seq_len'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        batch["labels"] = batch['input_ids']
        batch['retrieval_position'] = torch.tensor(batch['input_ids'].size(1) // 2)
        if neighbor_embeddings[0] is not None:
            batch['neighbor_embeddings'] = torch.tensor(neighbor_embeddings)
            batch['p_reps'] = torch.tensor(p_reps)
        else:
            batch['neighbor_embeddings'] = torch.tensor(p_reps)
            batch['p_reps'] = None
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
