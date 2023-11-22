import os
import sys
import torch
import datetime
import numpy as np
from utils import *
from model import *
from tqdm import tqdm
from time import time
from torch import Tensor, nn
from dataset_factory import *
import torch_optimizer as optim
import torch.distributed as dist
from dataclasses import dataclass
from accelerate import Accelerator
from transformers.file_utils import ModelOutput
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple, Union, Dict
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'
sys.path.append('/root/paddlejob/workspace/env_run/ReGPT')

config = get_config('config/regpt_config.yaml')
ReGPT_kwargs = config['ReGPT_kwargs']
config['training'].update(ReGPT_kwargs)
config['dataset']['train'].update(ReGPT_kwargs)
config['dataset']['test'].update(ReGPT_kwargs)
generation_kwargs = config['generation_kwargs']
config['training'].update(generation_kwargs)
train_config = config['training']
dataset_config = config['dataset']
train_config['negative_depth'] = dataset_config['train']['negative_depth']
dataset_config['train']['predict_from_last'] = train_config['predict_from_last']
dataset_config['test']['predict_from_last'] = train_config['predict_from_last']

tokenizer = AutoTokenizer.from_pretrained(train_config['tokenizer_name_or_path'])
tokenizer.pad_token = tokenizer.eos_token
train_config['eos_token_id'] = tokenizer.eos_token_id

dataset_class = {"DialogSFTDataset": DialogSFTDataset, "CorpusPretrainDataset": CorpusPretrainDataset, "ReGPTDialogSFTDataset": ReGPTDialogSFTDataset, "ReGPTCorpusPretrainDataset": ReGPTCorpusPretrainDataset}

config['training']['model_name_or_path'] = 'output/SFT-3'
print_args(config)
model = ReGPTForCausalLM(train_config)
model.cuda()
model = model.half()

def move_to_cuda(kwargs):
    for key in kwargs:
        kwargs[key] = kwargs[key].cuda()
model.train_config['do_sample'] = True
def generate(text='Tsinghua University is a '):
    kwargs = tokenizer([text],return_tensors='pt')
    move_to_cuda(kwargs)
    output_ids = model.generate(**kwargs)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output_ids.cpu().numpy().squeeze())),output_ids

model.train_config['max_length'] = 100
print(generate('Beijing is located in '))
