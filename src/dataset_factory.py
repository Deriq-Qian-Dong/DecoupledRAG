import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk

class DialogSFTDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.split = args['train_or_test']
        self.tokenizer = tokenizer
        self.setup_datasets()
        self.num_samples = len(self.datasets)

    def setup_datasets(self):
        self.datasets = load_dataset(self.args['data_name_or_path'], split=self.split)
        
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
        # self.datasets = load_dataset('text', data_files={'train': f'{data_name_or_path}/corpus.tsv', 'test':f'{data_name_or_path}/test.txt'})
        self.datasets = load_from_disk(data_name_or_path)[self.split]
        self.datasets = self.datasets.filter(self.filter_empty)

    def filter_empty(self, example):
        return example['text_length'] >= 10
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        text = sample['text']
        return text


class ReGPTDialogSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.negatives = np.loadtxt(args['negative_path']).reshape((-1, args['negative_depth_in_pool'], 4))
    
    def _collate_fn(self, elems):
        batch = super()._collate_fn(elems)
        input_ids = batch['input_ids'].numpy()
        predict_from_last = self.args['predict_from_last']
        predict_from_last = min(predict_from_last, len(input_ids[0])-1)  # 防止predict_from_last大于句子长度
        negative_infos = self.negatives[input_ids[:,-predict_from_last:]] # (batch_size, predict_from_last, negative_depth_in_pool, 4)
        # 取第FNTP_threshold个以后的负样本，过滤False Negative和True Positive
        negative_infos = negative_infos[:, :, self.args['FNTP_threshold']:, :]
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

