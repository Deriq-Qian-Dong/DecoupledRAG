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
config.negatives_x_device = True

config.kb_path = '../../data_of_ReGPT/marco/phrases_embeddings.npy'
config.retrieval_step = 10
config.topk = 6
config.q_reps_cache_type = 'RunningAverageQRepsCache'
config.q_reps_cache_window_size = 10

model = LlamaWithRetrievalHeadForInference.from_pretrained(model_path, config=config)          
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)


input_text = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

input_ids = input_ids[:,:30]
outputs = model.generate(input_ids,max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
from datasets import load_dataset, load_from_disk
dataset = load_from_disk('../../data_of_ReGPT/marco/collection/')


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
    from transformers import GPT2LMandRetrievalHeadsModel, LlamaWithRetrievalHeadForCausalLM, LlamaWithRetrievalHeadForInference
except:
    GPT2LMandRetrievalHeadsModel, LlamaWithRetrievalHeadForCausalLM, LlamaWithRetrievalHeadForInference = None, None, None
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
config = get_config('../config/rag_llama_test_config.yaml')
config['dataset']['test']
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('../output/SFT-best/')
%cd /root/paddlejob/workspace/env_run/ReGPT
dataset = RAGPretrainDataset(tokenizer, config['dataset']['test'])

RAG_kwargs = config['RAG_kwargs']
generation_kwargs = config['generation_kwargs']
config['training'].update(RAG_kwargs)
config['training'].update(generation_kwargs)
config['dataset']['train'].update(RAG_kwargs)
config['dataset']['test'].update(RAG_kwargs)

train_config = config['training']
model_config = AutoConfig.from_pretrained(train_config['model_name_or_path'])
model_config.kb_path = train_config['kb_path']
model_config.retrieval_step = train_config['retrieval_step']
model_config.topk = train_config['topk']
model_config.q_reps_cache_type = train_config['q_reps_cache_type']
model_config.q_reps_cache_window_size = train_config['q_reps_cache_window_size']

model = LlamaWithRetrievalHeadForInference.from_pretrained(config['training']['model_name_or_path'], config=model_config)
tokenizer = AutoTokenizer.from_pretrained(config['training']['tokenizer_name_or_path'])
dataset_config = config['dataset']
dataset_class = {"RAGPretrainDataset": RAGPretrainDataset, "DialogSFTDataset": DialogSFTDataset, "CorpusPretrainDataset": CorpusPretrainDataset, "ReGPTDialogSFTDataset": ReGPTDialogSFTDataset, "ReGPTCorpusPretrainDataset": ReGPTCorpusPretrainDataset, "ReGPTLongDocumentSummarizationSFTDataset": ReGPTLongDocumentSummarizationSFTDataset,"ReGPTDocumentSummarizationSFTDataset":ReGPTDocumentSummarizationSFTDataset, "ReGPTCorpusPretrainFromAfsDataset":ReGPTCorpusPretrainFromAfsDataset}
test_dataset = dataset_class[dataset_config['test']['dataset_name']](tokenizer, dataset_config['test'])
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset._collate_fn)
def _prepare_inputs(record):
    prepared = {}
    local_rank = "cuda:0"
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        elif x is None:
            prepared[key] = x
        elif isinstance(x, bool):
            prepared[key] = x
        elif isinstance(x, tuple):
            prepared[key] = x
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared
model = model.cuda()
model.eval()
total_loss = 0
pbar = tqdm(test_dataloader, desc=f"Evaluation")
with torch.no_grad():
    for step,batch in enumerate(test_dataloader):
        neighbor_embeddings = batch.pop('neighbor_embeddings')
        retrieval_position = batch.pop('retrieval_position')
        batch.pop('attention_mask')
        retrieval_position = int(retrieval_position)
        retrieval_step = config['training']['retrieval_step']
        input_ids = batch.pop('input_ids')
        seq_len = input_ids.size(1)
        labels = batch.pop('labels')
        if retrieval_step<0:
            retrieval_step = retrieval_position
        model.config.retrieval_step = retrieval_step
        # generate with teacher forcing and retrieval for each retrieval_step
        neighbor_embeddings = None
        past_key_values = None
        for i in range(retrieval_step, seq_len+retrieval_step, retrieval_step):
            batch['input_ids'] = input_ids[:, i-retrieval_step:i]
            batch['labels'] = labels[:, i-retrieval_step:i]
            batch['encoder_hidden_states'] = neighbor_embeddings
            batch['past_key_values'] = past_key_values
            model_inputs = model.prepare_inputs_for_generation(**batch)
            batch = _prepare_inputs(model_inputs)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.cpu().detach().float().numpy().mean()
            neighbor_embeddings = outputs.encoder_hidden_states
            past_key_values = outputs.past_key_values
        model._reset_q_reps_cache()
        pbar.update(1)
        pbar.set_description(f"Step {step} | Loss: {total_loss/(step+1):.4f} | Perplexity: {np.exp(total_loss/(step+1)):.4f} | Retrieval Position: {retrieval_position}")
