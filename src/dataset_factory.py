import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk

class DialogSFTDataset(Dataset):
    def __init__(self, tokenizer, args):
        try:
            self.rank = torch.distributed.get_rank()
            self.n_procs = torch.distributed.get_world_size() 
        except:
            self.rank = self.n_procs = 0
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
    

class ReGPTDialogSFTDataset(DialogSFTDataset):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.negatives = np.loadtxt(args['negatives_path']).reshape((-1, args['negative_depth_in_pool'], 4))

    def __getitem__(self, idx):
        tokenized_text = super().__getitem__(idx)
        input_ids = tokenized_text['input_ids'].tolist()
        negative_infos = self.negatives[input_ids[1:]]  # (L-1, negative_depth_in_pool, 4)
        # 取第FNTP_threshold个以后的负样本，过滤False Negative和True Positive
        negative_infos = negative_infos[:, self.args['FNTP_threshold']:, :]
        negative_ids = negative_infos[:, :, 1] # (L-1, negative_depth_in_pool)
        # 从负样本中随机选取self.args['negative_depth']个
        negative_ids = negative_ids[:, np.random.choice(negative_ids.shape[1], self.args['negative_depth'], replace=False)].astype(int) # (L-1, negative_depth)
        tokenized_text['negative_ids'] = torch.LongTensor(negative_ids)
        return tokenized_text
    
    def _collate_fn(self, elems):
        batch = super()._collate_fn(elems)
        batch['negative_ids'] = torch.stack([e['negative_ids'] for e in elems], dim=0)
        return batch
    