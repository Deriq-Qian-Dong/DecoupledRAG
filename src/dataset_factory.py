import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk

class DialogSFTDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.datasets = load_dataset(args['data_name_or_path'])
        self.split = args['train_or_test']
        self.tokenizer = tokenizer
        self.num_samples = len(self.datasets[self.split])
        self.args = args
        self.hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

    def __getitem__(self, idx):
        sample = self.datasets[self.split][idx]
        text = sample['prompt'] + sample['chosen']+self.tokenizer.eos_token
        tokenized_text = self.tokenizer(text,
                                   max_length=self.args['max_seq_len'],
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt")
        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0) # (1, L) -> (L)
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
        return tokenized_text

    def __len__(self):
        return self.num_samples

    def _collate_fn(self, elems):
        batch = self.hf_collate_fn(
                {"input_ids": [e["input_ids"] for e in elems], "attention_mask": [e["attention_mask"] for e in elems]}
            )
        batch["labels"] = batch['input_ids']
        return batch
    
class CorpusPretrainDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.split = args['train_or_test']
        data_name_or_path = args['data_name_or_path']
        # self.datasets = load_dataset('text', data_files={'train': f'{data_name_or_path}/corpus.tsv', 'test':f'{data_name_or_path}/test.txt'})
        self.datasets = load_from_disk(data_name_or_path)
        self.datasets = self.datasets.filter(self.filter_empty)
        self.tokenizer = tokenizer
        self.num_samples = len(self.datasets[self.split])
        self.args = args
        self.hf_collate_fn = DataCollatorWithPadding(self.tokenizer)

    def filter_empty(self, example):
        return example['text_length'] >= 10
    
    def __getitem__(self, idx):
        sample = self.datasets[self.split][idx]
        text = sample['text'] + self.tokenizer.eos_token
        tokenized_text = self.tokenizer(text,
                                   max_length=self.args['max_seq_len'],
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt")
        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
        return tokenized_text

    def __len__(self):
        return self.num_samples
    
    def _collate_fn(self, elems):
        batch = self.hf_collate_fn(
                {"input_ids": [e["input_ids"] for e in elems], "attention_mask": [e["attention_mask"] for e in elems]}
            )
        batch["labels"] = batch['input_ids']
        return batch

class ReGPTDialogSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.negatives = np.loadtxt(args['negative_path']).reshape((-1, args['negative_depth_in_pool'], 4))
        self.negs_of_eos = self.negatives[self.tokenizer.eos_token_id] # (negative_depth_in_pool, 4)
        # 取第FNTP_threshold个以后的负样本，过滤False Negative和True Positive
        self.negs_of_eos = self.negs_of_eos[self.args['FNTP_threshold']:, :] # (negative_depth_in_pool-FNTP_threshold, 4)
        self.negs_of_eos = self.negs_of_eos[:,1] # (negative_depth_in_pool-FNTP_threshold, )

    def __getitem__(self, idx):
        tokenized_text = super().__getitem__(idx)
        input_ids = tokenized_text['input_ids'].tolist()
        negative_infos = self.negatives[input_ids[1:]]  # (L-1, negative_depth_in_pool, 4)
        # 取第FNTP_threshold个以后的负样本，过滤False Negative和True Positive
        negative_infos = negative_infos[:, self.args['FNTP_threshold']:, :]
        negative_ids = negative_infos[:, :, 1] # (L-1, negative_depth_in_pool)
        # 从负样本中随机选取self.args['negative_depth']个
        negative_ids = negative_ids[:, np.random.choice(negative_ids.shape[1], self.args['negative_depth'], replace=False)].astype(int) # (L-1, negative_depth)
        tokenized_text['negative_ids'] = negative_ids
        return tokenized_text
    
    def _collate_fn(self, elems):
        batch = super()._collate_fn(elems)
        negs_of_eos = self.negs_of_eos[np.random.choice(self.negs_of_eos.shape[0], self.args['negative_depth'], replace=False)] # (negative_depth, )
        negative_ids = [e['negative_ids'] for e in elems] # (batch_size, L-1, negative_depth)
        # negative_ids中所有元素的第一个纬度上padding到同样的长度
        max_len = max([e.shape[0] for e in negative_ids])
        # 使用negs_of_eos填充
        negative_ids = [np.concatenate([e, np.tile(negs_of_eos, (max_len-e.shape[0], 1))], axis=0) for e in negative_ids] # (batch_size, max_len, negative_depth)
        negative_ids = np.stack(negative_ids, axis=0) # (batch_size, max_len, negative_depth)
        batch['negative_ids'] = torch.from_numpy(negative_ids).long()
        return batch
    
class ReGPTCorpusPretrainDataset(CorpusPretrainDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.negatives = np.loadtxt(args['negative_path']).reshape((-1, args['negative_depth_in_pool'], 4))
        self.negs_of_eos = self.negatives[self.tokenizer.eos_token_id] # (negative_depth_in_pool, 4)
        # 取第FNTP_threshold个以后的负样本，过滤False Negative和True Positive
        self.negs_of_eos = self.negs_of_eos[self.args['FNTP_threshold']:, :] # (negative_depth_in_pool-FNTP_threshold, 4)
        self.negs_of_eos = self.negs_of_eos[:,1] # (negative_depth_in_pool-FNTP_threshold, )

    def __getitem__(self, idx):
        tokenized_text = super().__getitem__(idx)
        input_ids = tokenized_text['input_ids'].tolist()
        negative_infos = self.negatives[input_ids[1:]] # (L-1, negative_depth_in_pool, 4)
        # 取第FNTP_threshold个以后的负样本，过滤False Negative和True Positive
        negative_infos = negative_infos[:, self.args['FNTP_threshold']:, :]
        negative_ids = negative_infos[:, :, 1] # (L-1, negative_depth_in_pool)
        # 从负样本中随机选取self.args['negative_depth']个
        negative_ids = negative_ids[:, np.random.choice(negative_ids.shape[1], self.args['negative_depth'], replace=False)].astype(int) # (L-1, negative_depth)
        tokenized_text['negative_ids'] = negative_ids
        return tokenized_text
    
    def _collate_fn(self, elems):
        batch = super()._collate_fn(elems)
        negs_of_eos = self.negs_of_eos[np.random.choice(self.negs_of_eos.shape[0], self.args['negative_depth'], replace=False)] # (negative_depth, )
        negative_ids = [e['negative_ids'] for e in elems] # (batch_size, L-1, negative_depth)
        # negative_ids中所有元素的第一个纬度上padding到同样的长度
        max_len = max([e.shape[0] for e in negative_ids])
        # 使用negs_of_eos填充
        negative_ids = [np.concatenate([e, np.tile(negs_of_eos, (max_len-e.shape[0], 1))], axis=0) for e in negative_ids] # (batch_size, max_len, negative_depth)
        negative_ids = np.stack(negative_ids, axis=0) # (batch_size, max_len, negative_depth)
        batch['negative_ids'] = torch.from_numpy(negative_ids).long()
        return batch
