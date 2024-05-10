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
%cd /root/paddlejob/workspace/env_run/ReGPT
config = get_config('config/rellama_config.yaml')
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

config['training']['model_name_or_path'] = 'output/SFT-step-4000'
print_args(config)
model = ReGPTForCausalLM(train_config)
model.cuda()
model = model.half()


def move_to_cuda(kwargs, device='cuda:0'):
    for key in kwargs:
        kwargs[key] = kwargs[key].to(device)
def generate(text='Tsinghua University is a '):
    text+=' '
    kwargs = tokenizer([text],return_tensors='pt')
    move_to_cuda(kwargs)
    output_ids = model.generate(**kwargs)
#     print(tokenizer.convert_ids_to_tokens(output_ids.cpu().numpy().squeeze()))
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(output_ids.cpu().numpy().squeeze()))

from time import time
start = time()
model.train_config['do_sample'] = True
model.train_config['top_k'] = 5
model.train_config['top_p'] = 0.8
model.train_config['max_length'] = 100
print(generate('Beethoven'))
print(time()-start)

llama = AutoModelForCausalLM.from_pretrained("../llama2-7b/")
llama_tokenizer = AutoTokenizer.from_pretrained("../llama2-7b/")
llama=llama.to("cuda:1").half()

def generate_llama(text, model, tokenizer, max_length):
    start = time()
    model.eval()
    text+=' '
    dtype = model.parameters().__next__().dtype
    kwargs = tokenizer([text],return_tensors='pt')
    move_to_cuda(kwargs, 'cuda:1')
    input_ids = kwargs.pop('input_ids')
    attention_mask = kwargs.pop('attention_mask')
    kwargs['output_hidden_states'] = True
    with torch.no_grad():
        for step in range(max_length):
            # 使用模型生成下一个标记
            output = model.generate(input_ids, max_new_tokens=1, top_k=50, top_p=0.95)

            # 获取生成的标记
            next_token_id = output[:1, -1:]

            # 将生成的标记添加到输入中，用于下一步的生成
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 解码生成的文本
            generated_text = tokenizer.decode(input_ids[0])
        print(step)
#             print(f"Step {step + 1}: {generated_text}")
#     print('用时：',time()-start)
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy().squeeze()))

start = time()
generated_text = generate_llama("Western", llama, llama_tokenizer, 100)
print(generated_text)
print('单词个数:', len(generated_text.split()))
print('用时：',time()-start)

from transformers import AutoConfig, LlamaWithRetrievalHeadForInference, AutoTokenizer
model_path = '../../rag_llama2/two_ca_layer/SFT-best/'
config = AutoConfig.from_pretrained(model_path)
config.add_cross_attention = True
config.faiss_dimension = 768
config.cross_attention_activation_function = 'silu'
config.add_cross_attention_layer_number = 1
config.negatives_x_device = True

config.kb_path = '../../data_of_ReGPT/marco/phrases_embeddings.npy'
config.retrieval_step = 10
config.topk = 6

model = LlamaWithRetrievalHeadForInference.from_pretrained(model_path, config=config)          
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)


input_text = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

input_ids = input_ids[:,:30]
outputs = model.generate(input_ids,max_new_tokens=100)
from datasets import load_dataset
dataset = load_dataset('csv', data_files={'train': '../../data_of_ReGPT/marco/collection.tsv'}, delimiter='\t',column_names=['pid', 'text'])['train']



