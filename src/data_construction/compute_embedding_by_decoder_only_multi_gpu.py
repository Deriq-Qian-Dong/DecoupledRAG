import os
import sys
import json
import torch
import numpy as np
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from tqdm import tqdm

os.environ['http_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'
os.environ['https_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'
def get_config(path="config/rellama_config.yaml"):
    with open(path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
def computing_for_decoder_only_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.eod_id
    try:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        pass
    tokenizer.truncation_side='left'
    tokenizer.padding_side='left'
    model = AutoModel.from_pretrained(config['model_name_or_path'])
    dataset = load_dataset('csv', data_files={'train': config['data_name_or_path']}, delimiter='\t',column_names=['pid', 'text'])['train']
    def collate_fn(elems):
        pids = []
        texts = []
        for elem in elems:
            pids.append(elem['pid'])
            texts.append(elem['text'])
        batch = tokenizer(texts,
                            max_length=config['max_length'],
                            padding=True,
                            truncation=True,
                            return_tensors="pt")
        batch['pids'] = pids
        return batch
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        print('start computing...')
        for key in config:
            print(f'{key}: {config[key]}')
    model = accelerator.prepare_model(model)
    model.eval()
    data = []
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    # dataset = dataset.select(range(100000))
    dataset = dataset.shard(num_shards=world_size, index=local_rank, contiguous=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=collate_fn, shuffle=False)
    # dataloader = accelerator.prepare_data_loader(dataloader)
    all_embeddings = []
    all_pids = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='computing', disable=not accelerator.is_local_main_process):
            pids = batch.pop('pids')
            # move batch to device
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = model(**batch).last_hidden_state
            embeddings = outputs[:, -1, :].detach().cpu().numpy()
            all_embeddings.append(embeddings)
            all_pids.extend(pids)
        all_embeddings = np.concatenate(all_embeddings)
        all_pids = np.array(all_pids)
        np.save(config['output_path']+f'_{local_rank}.npy', all_embeddings)
        np.save(config['output_path']+f'_pids_{local_rank}.npy', all_pids)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # merge all data
            merged_embeddings = []
            merged_pids = []
            for rank in range(world_size):
                data = np.load(config['output_path']+f'_{rank}.npy')
                merged_embeddings.append(data)
                pids = np.load(config['output_path']+f'_pids_{rank}.npy')
                merged_pids.extend(pids.tolist())
            merged_embeddings = np.concatenate(merged_embeddings)
            merged_pids = np.array(merged_pids)
            np.save(config['output_path']+".npy", merged_embeddings)
            np.save(config['output_path']+"_pids.npy", merged_pids)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    if config['model_type'] == 'decoder_only':
        computing_for_decoder_only_model(config)
    else:
        raise ValueError('model_type must be decoder_only or encoder_decoder')