import os
import sys
import torch
import random
import datetime
import numpy as np
from utils import *
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
try:
    from transformers import GPT2LMandRetrievalHeadsModel, LlamaWithRetrievalHeadForCausalLM
except:
    GPT2LMandRetrievalHeadsModel, LlamaWithRetrievalHeadForCausalLM = None, None
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

tokenizer = AutoTokenizer.from_pretrained('./output/SFT-new/')
args = {'data_name_or_path':'../data_of_ReGPT/marco/sorted_datasets_train_llama2/','max_seq_len':256}
data = RAGPretrainDataset(tokenizer, args)
max_tokens = 16384  # 设置每个batch的最大token数
accelerator = Accelerator(log_with='tensorboard', project_dir='output_test/')
print('world size: %d\nlocal rank: %d' % (accelerator.num_processes, accelerator.process_index))
sampler = DynamicBatchSampler(data, max_tokens, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
dataloader = DataLoader(data, batch_sampler=sampler, shuffle=False, collate_fn=data._collate_fn)
print('dataloader length: %d\nlocal rank: %d' % (len(dataloader), accelerator.process_index))
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
accelerator.init_trackers(project_name=f'test_{timestamp}')
# dataloader = accelerator.prepare_data_loader(dataloader)
pbar = tqdm(total=len(dataloader))
for step, batch in enumerate(dataloader):
    batch_size = batch.input_ids.shape[0]
    stats = {'batch_size': batch_size}
    accelerator.log(stats, step)
    if accelerator.is_main_process:
        pbar.update(1)

