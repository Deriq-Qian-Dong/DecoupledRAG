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
        self.datasets = load_dataset(args['dataset_name_or_path'])
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