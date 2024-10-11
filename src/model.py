import os
import sys
import torch
import random
import datetime
import itertools
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
from prompt_templates import QA_Reasoning_PROMPT
from transformers.file_utils import ModelOutput
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Tuple, Union, Dict
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from registry import registry, register_class
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
try:
    from transformers import GPT2LMandRetrievalHeadsModel, LlamaWithRetrievalHeadForCausalLM, LlamaWithRetrievalHeadForInference, LlamaWithRetrievalHeadAndKnowledgeInjectorForCausalLM
except:
    GPT2LMandRetrievalHeadsModel, LlamaWithRetrievalHeadForCausalLM, LlamaWithRetrievalHeadForInference = None, None, None
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from evaluation import Evaluator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Process, Pool

torch.manual_seed(random.randint(0, 1000000))

optimizer_class = {"AdamW": FusedAdam, "Lamb": optim.Lamb, "DeepSpeedCPUAdam": DeepSpeedCPUAdam}
scheduler_class = {"CosineAnnealingLR": CosineAnnealingLR, "LinearLR": LinearLR}

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def dataset_class(class_name):
    cls = registry.get_class(class_name)
    if cls:
        return cls
    else:
        raise ValueError(f"Class {class_name} not found")

# dataset_class = {"RAGPretrainDataset": RAGPretrainDataset, "DialogSFTDataset": DialogSFTDataset, "CorpusPretrainDataset": CorpusPretrainDataset, "ReGPTDialogSFTDataset": ReGPTDialogSFTDataset, "ReGPTCorpusPretrainDataset": ReGPTCorpusPretrainDataset, "ReGPTLongDocumentSummarizationSFTDataset": ReGPTLongDocumentSummarizationSFTDataset,"ReGPTDocumentSummarizationSFTDataset":ReGPTDocumentSummarizationSFTDataset, "ReGPTCorpusPretrainFromAfsDataset":ReGPTCorpusPretrainFromAfsDataset, "QADataset":QADataset, "QASFTDataset":QASFTDataset, "QAEvalDataset":QAEvalDataset}

@dataclass
class ReGPTOutput(ModelOutput):
    loss: Optional[Tensor] = None

@register_class
class ReGPTForCausalLM(nn.Module):
    def __init__(self, train_config):
        super(ReGPTForCausalLM, self).__init__()
        model = AutoModel.from_pretrained(train_config['model_name_or_path'], use_cache=not train_config['gradient_checkpointing'])            
        freeze_bottom_causal_layers(model.base_model, train_config['num_layers_unfrozen'])
        if train_config['gradient_checkpointing']:
            # model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if 'lora_model_name_or_path' in train_config:    
            lora_config = LoraConfig.from_pretrained(train_config['lora_model_name_or_path'])
            torch.cuda.is_available = lambda : False
            model = PeftModel.from_pretrained(model, train_config['lora_model_name_or_path'], config=lora_config, is_trainable=True)
            torch.cuda.is_available = lambda : True
            model.print_trainable_parameters()
        else:
            model.base_model.get_input_embeddings().weight.requires_grad = False
        # model = model.merge_and_unload()
        self.model = model
        matrix = np.load(open(train_config['faiss']['matrix_path'], 'rb'))
        self.matrix = matrix
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.train_config = train_config
        
    def forward(self, **kwargs):
        self.dtype = self.model.parameters().__next__().dtype
        input_ids = kwargs.pop('input_ids')
        negative_ids = kwargs.pop('negative_ids')
        labels = kwargs.pop('labels')
        predict_from_last = self.train_config['predict_from_last']
        predict_from_last = min(predict_from_last, input_ids.size(1)-1)
        labels = labels[:, -predict_from_last:].contiguous()  # [batch_size, predict_from_last]
        inputs_embeds = self.matrix[input_ids.cpu()]
        inputs_embeds = torch.from_numpy(inputs_embeds).to(input_ids.device) # [batch_size, seq_len, hidden_size]
        inputs_embeds = inputs_embeds.to(self.dtype)
        negative_embeds = self.matrix[negative_ids.cpu()]
        negative_embeds = torch.from_numpy(negative_embeds).to(input_ids.device) 
        positive_embeds = self.matrix[labels.cpu()]  # [batch_size, predict_from_last, hidden_size]
        positive_embeds = torch.from_numpy(positive_embeds).to(input_ids.device) 
        embeds_for_contrastive_training = torch.cat([positive_embeds.unsqueeze(2), negative_embeds], dim=2).to(self.dtype).contiguous()  # [batch_size, predict_from_last, 1+negative_depth, hidden_size]
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['output_hidden_states'] = True
        # set requires_grad to True
        kwargs['inputs_embeds'].requires_grad = True
        outputs = self.model(**kwargs)

        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        last_hidden_state = last_hidden_state[:, -predict_from_last-1:-1, :].contiguous()  # [batch_size, predict_from_last, hidden_size]
        q_reps = last_hidden_state.view(-1, embeds_for_contrastive_training.shape[-1])  # [batch_size*predict_from_last, hidden_size]
        # l2 norm
        # q_reps = q_reps / torch.norm(q_reps, dim=-1, keepdim=True)
        p_reps = embeds_for_contrastive_training.view(-1, embeds_for_contrastive_training.shape[-1])  # [batch_size*predict_from_last*(1+negative_depth), hidden_size]
        if self.train_config['negatives_in_device']:
            scores = self.compute_similarity(q_reps, p_reps)  # [batch_size*predict_from_last, batch_size*predict_from_last*(1+negative_depth)]
            scores = scores.view(q_reps.size(0), -1)  # [batch_size*predict_from_last, batch_size*predict_from_last*(1+negative_depth)]
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)  # [batch_size*predict_from_last]
            target = target * (p_reps.size(0) // q_reps.size(0))  # [batch_size*predict_from_last]
        else:
            p_reps = p_reps.view(-1, 1+self.train_config['negative_depth'], p_reps.size(-1))  # [batch_size*predict_from_last, 1+negative_depth, hidden_size]
            scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))  # [batch_size*predict_from_last, 1, 1+negative_depth]
            scores = scores.squeeze(1)  # [batch_size*predict_from_last, 1+negative_depth]
            target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)  # [batch_size*predict_from_last]
        # ignore_index=-1 where label==self.train_config['eos_token_id']
        labels = labels.view(-1)  # [batch_size*predict_from_last]
        target[labels==self.train_config['eos_token_id']] = -1
        loss = self.cross_entropy(scores, target)
        return ReGPTOutput(loss=loss)
    
    def save_gate_state(self, gate_crossattention, path):
        """保存每一层的 gate_crossattention"""
        torch.save(gate_crossattention.state_dict(), path)
    
    def save_pretrained(self, directory):
        self.model.knowledge_injector.save_pretrained(directory) 
        # 创建保存目录（如果不存在）
        os.makedirs(directory, exist_ok=True)
        # 使用 ThreadPoolExecutor 并行存储
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for i in range(self.train_config['add_cross_attention_layer_number'] + 1):
                gate_crossattention = self.model.model.layers[i].gate_crossattention
                path = f"{directory}/gate_{i}.pt"
                
                # 提交任务给线程池
                futures.append(executor.submit(self.save_gate_state, gate_crossattention, path))
    

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    
    def generate(self, **kwargs):
        self.model.eval()
        self.dtype = self.model.parameters().__next__().dtype
        input_ids = kwargs.pop('input_ids')
        attention_mask = kwargs.pop('attention_mask')
        kwargs['output_hidden_states'] = True
        p_reps = self.matrix  # [phrases_size, hidden_size]
        p_reps = torch.from_numpy(p_reps).to(input_ids.device).to(self.dtype)  # [phrases_size, hidden_size]
        with torch.no_grad():
            for _ in range(self.train_config['max_length']):
                inputs_embeds = self.matrix[input_ids.cpu()]
                inputs_embeds = torch.from_numpy(inputs_embeds).to(input_ids.device) # [batch_size, cur_seq_len, hidden_size]
                inputs_embeds = inputs_embeds.to(self.dtype)
                kwargs['inputs_embeds'] = inputs_embeds
                outputs = self.model(**kwargs)
                last_hidden_state = outputs.last_hidden_state  # [batch_size, cur_seq_len, hidden_size]
                last_hidden_state = last_hidden_state[:, -1:, :].contiguous()  # [batch_size, 1, hidden_size]
                q_reps = last_hidden_state.view(-1, self.matrix.shape[-1])  # [batch_size, hidden_size]
                # l2 norm
                # q_reps = q_reps / torch.norm(q_reps, dim=-1, keepdim=True)
                q_reps = q_reps.unsqueeze(1)  # [batch_size, 1, hidden_size]
                scores = self.compute_similarity(q_reps, p_reps)  # [batch_size, phrases_size]
                scores = scores.squeeze(1)  # [batch_size, phrases_size]
                scores = torch.softmax(scores, dim=-1)  # [batch_size, phrases_size]
                if self.train_config['do_sample']:
                    # top-k or top-p sampling
                    scores = scores.squeeze(0)  # [phrases_size]
                    filtered_logits = top_k_top_p_filtering(scores, top_k=self.train_config['top_k'], top_p=self.train_config['top_p'], filter_value=0)
                    next_input_ids = torch.multinomial(filtered_logits, num_samples=1).reshape(-1, 1) # [batch_size, 1]
                else:
                    # greedy decoding
                    next_input_ids = torch.argmax(scores, dim=-1).reshape(-1, 1) # [batch_size, 1]
                input_ids = torch.cat([input_ids, next_input_ids], dim=-1)
        return input_ids

MODEL_CLASS = {'gpt2': GPT2LMandRetrievalHeadsModel, 'llama': LlamaWithRetrievalHeadAndKnowledgeInjectorForCausalLM}
@register_class
class RAGForCausalLM(nn.Module):
    def __init__(self, train_config):
        super(RAGForCausalLM, self).__init__()
        config = AutoConfig.from_pretrained(train_config['model_name_or_path'])
        config.add_cross_attention = True
        config.faiss_dimension = train_config['faiss']['dimension']
        config.cross_attention_activation_function = train_config['cross_attention_activation_function']
        config.add_cross_attention_layer_number = train_config['add_cross_attention_layer_number']
        config.negatives_x_device = train_config['negatives_x_device']
        config.output_hidden_states = True
        config.kg_model_name_or_path = train_config['kg_model_name_or_path']
        config.freeze_retrieval_head = train_config['freeze_retrieval_head']
        self.model_config = config
        model = MODEL_CLASS[train_config['model_type']].from_pretrained(train_config['model_name_or_path'], config=config)          
        # import os
        # if os.path.exists(os.path.join(train_config['kg_model_name_or_path'], 'adapter_config.json')):
        #     model.model.load_adapter(train_config['kg_model_name_or_path'], "knowledge_injector")
        # else:
        #     peft_config = LoraConfig(
        #         lora_alpha=32,
        #         lora_dropout=0.1,
        #         r=16,
        #         bias='none',
        #         task_type="CAUSAL_LM"
        #     )
        #     model.model.add_adapter(peft_config, "knowledge_injector")
        freeze_non_crossattention_parameters(model, train_config['freeze_retrieval_head'], train_config['freeze_lm_head'])
        if train_config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if train_config['model_type']=='gpt2':
            model.base_model.wpe.weight.requires_grad = False
            model.base_model.get_input_embeddings().weight.requires_grad = False
        else:
            model.base_model.embed_tokens.weight.requires_grad = False
        # print_trainable_params_stats(model)
        self.model = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.train_config = train_config
        
    def forward(self, **kwargs):
        self.dtype = self.model.parameters().__next__().dtype
        neighbor_embeddings = kwargs.pop('neighbor_embeddings', None)
        outputs = self.model(**kwargs)
        return outputs
    
    def save_pretrained(self, directory):
        # self.model.model.save_pretrained(directory) 
        os.makedirs(directory, exist_ok=True)
        for i in range(self.train_config['add_cross_attention_layer_number']+1):
            # gate_scores.append(float(self.model.model.layers[i].gate_crossattention.cpu().detach().float().numpy()[0]))
            state_dict = self.model.model.layers[i].gate_crossattention.state_dict()
            # move to cpu
            state_dict = {key: value.cpu() for key, value in state_dict.items()}
            torch.save(state_dict, f"{directory}/gate_{i}.pt")

@register_class
class LanguageModelTrainer:
    def __init__(self, config):
        config['training']['kg_model_name_or_path'] = os.path.join(config['training']['project_dir'], config['training']['kg_model_name_or_path'])
        self.config = config
        generation_kwargs = config['generation_kwargs']
        self.config['training'].update(generation_kwargs)
        for key in self.config['dataset']['test']:
            self.config['dataset']['test'][key]['number_of_docs'] = self.config['dataset']['number_of_docs']
            self.config['dataset']['test'][key]['inference_with_explict_docs_for_test'] = self.config['dataset']['inference_with_explict_docs_for_test']
            self.config['dataset']['test'][key]['compare_speed'] = self.config.get('compare_speed', False)
        for key in self.config['dataset']['train']:
            self.config['dataset']['train'][key]['number_of_docs'] = self.config['dataset']['number_of_docs']
            self.config['dataset']['train'][key]['inference_with_explict_docs_for_test'] = self.config['dataset']['inference_with_explict_docs_for_test']
        self.setup()
        self.best_perplexity = 1e10
        self.sampler = None
        self.best_metric = 0.0

    def run(self):
        self.test()
        # for epoch in range(self.train_config['start_from'], self.train_config['num_epochs']):
        #     assert self.config.get('compare_speed', False) == False
        #     self.epoch = epoch
        #     self.set_epoch_to_dataset()
        #     self.train()
        #     self.test()

    def set_epoch_to_dataset(self):
        number_of_docs_lst = [20]
        # number_of_docs_lst = [1]
        number_of_docs = number_of_docs_lst[self.epoch%len(number_of_docs_lst)]
        for key in self.dataset_config['train']:
            self.dataset_config['train'][key]['number_of_docs'] = number_of_docs
        for key in self.dataset_config['test']:
            self.dataset_config['test'][key]['number_of_docs'] = number_of_docs
        self.setup_test_dataloader()
        self.setup_train_dataloader()

    def setup_train_dataloader(self):
        self.train_dataloaders = {}
        for key in self.dataset_config['train']:
            dataset_args = self.dataset_config['train'][key]
            dataloader = self.get_dataloader(dataset_args)
            self.train_dataloaders[key] = dataloader

    def setup_test_dataloader(self, number_of_docs=None):
        if number_of_docs is not None:
            for key in self.dataset_config['test']:
                self.dataset_config['test'][key]['number_of_docs'] = number_of_docs
        self.test_dataloaders = {}
        for key in self.dataset_config['test']:
            dataset_args = self.dataset_config['test'][key]
            dataloader = self.get_dataloader(dataset_args)
            self.test_dataloaders[key] = dataloader
    
    def get_dataloader(self, dataset_args):
        dataset = dataset_class(dataset_args['dataset_name'])(self.tokenizer, dataset_args)
        dataset.set_epoch(self.epoch)
        if dataset_args['dynamic_sampler']:
            sampler = DynamicBatchSampler(dataset, dataset_args['max_tokens'], num_replicas=self.accelerator.num_processes, rank=self.accelerator.process_index)            
            dataloader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, collate_fn=dataset._collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=dataset_args['batch_size'], shuffle=True, collate_fn=dataset._collate_fn)
            dataloader = self.accelerator.prepare_data_loader(dataloader)
        return dataloader

    def setup_model(self, train_config):
        model = AutoModelForCausalLM.from_pretrained(train_config['model_name_or_path'], use_cache=not train_config['gradient_checkpointing'])
        self.model_config = model.config
        # freeze_bottom_causal_layers(model.base_model, train_config['num_layers_unfrozen'])
        # try:
        #     # llama2
        #     model.base_model.embed_tokens.weight.requires_grad = train_config['num_layers_unfrozen']<=0
        # except:
        #     # gpt2
        #     model.base_model.wte.weight.requires_grad = train_config['num_layers_unfrozen']<=0
        #     model.base_model.wpe.weight.requires_grad = train_config['num_layers_unfrozen']<=0
        # print_trainable_params_stats(model)
        if train_config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if os.path.exists(os.path.join(train_config['kg_model_name_or_path'], 'adapter_config.json')):
            model.load_adapter(train_config['kg_model_name_or_path'], "finetune")
        else:
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="CAUSAL_LM"
            )
            model.add_adapter(peft_config, "finetune")
        self.model = model

    def setup_config(self, train_config, dataset_config):
        return train_config, dataset_config
    
    def _prepare_inputs(self, record):
        prepared = {}
        local_rank = self.accelerator.process_index
        for key in record:
            x = record[key]
            if isinstance(x, torch.Tensor):
                prepared[key] = x.to(local_rank)
                # if prepared[key].dtype in [torch.float32, torch.float64, torch.float16]:
                    # prepared[key].requires_grad = True
            elif x is None:
                prepared[key] = x
            elif isinstance(x, bool):
                prepared[key] = x
            elif isinstance(x, tuple):
                prepared[key] = x
            else:
                prepared[key] = self._prepare_inputs(x)
        return prepared

    def setup(self):
        config = self.config
        train_config = config['training']
        dataset_config = config['dataset']
        train_config, dataset_config = self.setup_config(train_config, dataset_config)
        tokenizer = AutoTokenizer.from_pretrained(train_config['tokenizer_name_or_path'])
        tokenizer.pad_token = tokenizer.eos_token
        train_config['eos_token_id'] = tokenizer.eos_token_id
        
        self.setup_model(train_config)
        model = self.model

        train_config["optimizer"]["kwargs"]['eps'] = float(train_config["optimizer"]["kwargs"]['eps'])
        train_config["optimizer"]["kwargs"]['lr'] = float(train_config["optimizer"]["kwargs"]['lr'])
        params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
        params = {'params': [v for k, v in params]}
        optimizer = optimizer_class[train_config["optimizer"]["name"]](
            [params],
            **train_config["optimizer"]["kwargs"],
        )
        if train_config["scheduler"]["name"] == "CosineAnnealingLR":
            train_config["scheduler"]["kwargs"]['eta_min'] = train_config['optimizer']['kwargs']['lr'] * 0.1
        scheduler = scheduler_class[train_config["scheduler"]["name"]](optimizer, **train_config["scheduler"]["kwargs"])

        accelerator = Accelerator(log_with=train_config['log_with'], project_dir=train_config['project_dir'])
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        accelerator.init_trackers(project_name=f'{train_config["project_name"]}_{timestamp}')
        if accelerator.is_main_process:
            print_trainable_params_stats(model)
        if tokenizer.chat_template is None:
            template = '''{% set loop_messages = messages %}\n{% for message in loop_messages %}\n{% if message['role'] == 'user' %}\nQuestion: {{ message['content'] | trim }}\n{% elif message['role'] == 'assistant' %}\nAnswer: {{ message['content'] | trim }}{% if loop.last and not add_generation_prompt %}</s>{% endif %}\n{% elif message['role'] == 'system' %}\nInstruction: {{ message['content'] | trim }}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\nAnswer: {% endif %}'''
            tokenizer.chat_template = template
        self.tokenizer = tokenizer
        self.epoch = 0
        self.accelerator = accelerator
        self.dataset_config = dataset_config
        self.setup_test_dataloader()
        key = list(dataset_config['test'].keys())[0]
        dataset_args = dataset_config['test'][key]
        test_dataset = dataset_class(dataset_args['dataset_name'])(self.tokenizer, dataset_args)
        test_dataloader = DataLoader(test_dataset, batch_size=dataset_args['batch_size'], shuffle=False, collate_fn=test_dataset._collate_fn)
        (model, optimizer, _, _) = accelerator.prepare(model, optimizer, test_dataloader, test_dataloader)
        # Upcasting trainable params to float32.
        for param in filter(lambda p: p.requires_grad, model.module.parameters()):
            param.data = param.data.to(torch.float32)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_config = train_config
        self.iter_count = 0
        print_args(config)

    def compute_loss(self, outputs):
        loss = outputs.loss
        stats = {"loss": float(loss.cpu().detach().float().numpy())}
        return loss, stats
    
    def task_specific_stats(self, stats, model):
        return stats

    def pop_unused_keys(self, record):
        record.pop('knowledge_input_ids', None)
        record.pop('retrieval_position', None)
        record.pop('neighbor_embeddings', None)
        return record
    
    def create_combined_loader(self, train_dataloaders):
        # 创建一个计数器以跟踪每个 DataLoader 的结束次数
        counters = {name: 0 for name in train_dataloaders}
        loaders = {name: iter(loader) for name, loader in train_dataloaders.items()}

        while True:
            for key in itertools.cycle(loaders.keys()):
                # 如果所有的 dataloader 都遍历过，输出信息并终止while循环
                if all(count >= 1 for count in counters.values()):
                    return
                try:
                    # 尝试获取数据
                    batch = next(loaders[key])
                    yield key, batch
                except StopIteration:
                    # 该 dataloader 到末尾，重置迭代器并增加计数器
                    loaders[key] = iter(train_dataloaders[key])
                    counters[key] += 1

    def train(self):
        model, optimizer, train_dataloaders, scheduler, accelerator, epoch = self.model, self.optimizer, self.train_dataloaders, self.scheduler, self.accelerator, self.epoch
        model.train()
        # train_dataloaders is a dict
        number_of_steps = sum([len(dataloader) for dataloader in train_dataloaders.values()])
        pbar = tqdm(total=number_of_steps)
        local_rank = accelerator.process_index
        print(f"Number of steps of process {local_rank}: {number_of_steps}")
        step = 0
        combined_loader = self.create_combined_loader(train_dataloaders)
        # for key in train_dataloaders:
            # print(f"\033[31mProcess {local_rank} | Dataset: {key} | Number of steps: {len(train_dataloaders[key])}\033[0m")
        # for batch in train_dataloaders[key]:
        for (dataset_name, batch) in combined_loader:
            # if self.iter_count==0 and step<self.train_config['skip_steps']:
            #     step+=1
            #     if accelerator.is_main_process:
            #         pbar.update(1)
            #         pbar.set_description(f"Epoch {epoch} | Skiping {step}/{self.train_config['skip_steps']}")
            #     continue
            self.iter_count += 1
            total_time = time()
            batch = self.pop_unused_keys(batch)
            seq_len = batch['input_ids'].size(1)
            batch_size = batch.input_ids.shape[0]
            batch = self._prepare_inputs(batch)
            batch = accelerator.prepare(batch)
            forward_time = time()
            outputs = model(**batch)
            forward_time = time() - forward_time
            loss, stats = self.compute_loss(outputs)
            stats["training/seq_len"] = seq_len
            stats["training/batch_size"] = batch_size
            stats = self.task_specific_stats(stats, model)
            backward_time = time()
            accelerator.backward(loss)
            backward_time = time() - backward_time
            stats["time/forward"] = forward_time
            stats["time/backward"] = backward_time
            opt_time = time()
            for group_number, lr in enumerate(scheduler.get_last_lr()):
                stats[f"training/learning_rate"] = lr
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_description(f"Epoch {epoch} | Step {step} | Loss: {loss.cpu().detach().float().numpy():.4f} | Dataset_name: {dataset_name}")
                step += 1
            if self.iter_count%self.train_config['eval_step']==0:
                self.test()
            opt_time = time() - opt_time
            stats["time/optimization"] = opt_time
            total_time = time() - total_time
            stats["time/total"] = total_time
            data_loading_time = total_time - forward_time - backward_time - opt_time
            stats["time/data_loading"] = data_loading_time
            accelerator.log(stats, step=self.iter_count)
        pbar.close()

    def generate(self, batch, key):
        model = self.model
        if self.config.get('compare_speed', False):
            outputs = model.module.generate(**batch, max_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'], do_sample=False, pad_token_id=self.tokenizer.eos_token_id, min_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'])
        else:
            outputs = model.module.generate(**batch, max_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'], do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return outputs
    
    def _compute_f1(self, prediction, answer, tokenizer):
        # Tokenize the prediction and answer
        pred_tokens = tokenizer.tokenize(prediction)
        answer_tokens = tokenizer.tokenize(answer)

        # Calculate true positives, false positives, and false negatives
        common_tokens = set(pred_tokens) & set(answer_tokens)
        true_positive = len(common_tokens)
        false_positive = len(set(pred_tokens) - common_tokens)
        false_negative = len(set(answer_tokens) - common_tokens)

        # Calculate precision, recall, and F1 score
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1
    
    def compute_f1(self, prediction, answers, tokenizer):
        assert isinstance(answers, list)
        best_f1 = 0.0
        for answer in answers:
            best_f1 = max(self._compute_f1(prediction, answer, tokenizer), best_f1)
        return best_f1
    
    def compute_accuracy(self, prediction, answers, tokenizer):
        assert isinstance(answers, list)
        for answer in answers:
            if answer == prediction:
                return 1.0
        return 0.0
    
    @torch.no_grad()
    def compute_hidden_states(self, batch, num_docs, batch_size=20):
        model = self.model
        model_config = self.model_config
        start_time = time()

        # Split input_ids into smaller batches if batch_size is specified
        input_ids = batch['knowledge_input_ids']
        total_samples = input_ids.size(0)

        # Prepare a list to accumulate hidden states for all layers
        accumulated_hidden_states = [[] for _ in range(model_config.num_hidden_layers)]

        for i in range(0, total_samples, batch_size):
            batch_input_ids = input_ids[i:i + batch_size]

            outputs = model.model(input_ids=batch_input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.past_key_values
            if hidden_states is None:
                print("Hidden states is None")
                exit()

            for layer_idx in range(len(hidden_states)):
                encoder_hidden_states = hidden_states[layer_idx]
                key_states = encoder_hidden_states[0]
                value_states = encoder_hidden_states[1]

                # Accumulate the hidden states for this layer (before any reshaping or manipulation)
                accumulated_hidden_states[layer_idx].append((key_states, value_states))

        # After all batches are processed, concatenate the hidden states along the first dimension
        new_hidden_states = []
        for layer_hidden_states in accumulated_hidden_states:
            # Concatenate key and value states for each layer
            layer_key_states = torch.cat([item[0] for item in layer_hidden_states], dim=0)
            layer_value_states = torch.cat([item[1] for item in layer_hidden_states], dim=0)

            # Perform operations after concatenation
            kv_len = layer_key_states.size(2)
            bsz = layer_key_states.size(0) // num_docs
            num_heads = model_config.num_attention_heads
            head_dim = model_config.hidden_size // num_heads
            num_key_value_heads = model_config.num_key_value_heads

            # Reshape key and value states
            key_states = layer_key_states.view(bsz, num_docs, num_key_value_heads, kv_len, head_dim)\
                                        .transpose(1, 2)\
                                        .reshape(bsz, num_key_value_heads, num_docs * kv_len, head_dim)
            value_states = layer_value_states.view(bsz, num_docs, num_key_value_heads, kv_len, head_dim)\
                                            .transpose(1, 2)\
                                            .reshape(bsz, num_key_value_heads, num_docs * kv_len, head_dim)

            num_key_value_groups = num_heads // num_key_value_heads
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            new_hidden_states.append((key_states, value_states))

        return new_hidden_states, time() - start_time

    def test(self):
        model, tokenizer, optimizer, scheduler, test_dataloaders, accelerator, iter_count = self.model, self.tokenizer, self.optimizer, self.scheduler, self.test_dataloaders, self.accelerator, self.iter_count
        model.eval()
        
        # Define the available metrics and corresponding functions
        metrics_function_dict = {
            'f1': self.compute_f1,
            'accuracy': self.compute_accuracy,
        }

        # Determine which metrics to calculate from the config
        metrics = self.config['training']['metrics']  # List of metrics to calculate, e.g., ['accuracy', 'f1']
        target_metric = self.config['training']['target_metric']  # Used to save the best checkpoint, either 'accuracy' or 'f1'
        
        metrics_dict = {metric: {} for metric in metrics}

        with torch.no_grad():
            # for number_of_docs in [40, 50]:
            for number_of_docs in [2,1,3,5,10,20,30,40,50]:
            # for number_of_docs in [20]:
                metrics_results_dict = {metric: [] for metric in metrics}
                self.setup_test_dataloader(number_of_docs=number_of_docs)
                test_dataloaders = self.test_dataloaders
                results = []
                start_time = time()
                hidden_states_time = 0
                total_generated_tokens = 0
                for key in test_dataloaders:
                    number_of_docs = test_dataloaders[key].dataset.number_of_docs
                    # Prepare to accumulate metrics for this dataset
                    metrics_accumulated = {metric: 0.0 for metric in metrics}
                    total_sample_count = 0
                    accelerator.print(f"Dataset: {key} | Number of steps: {len(test_dataloaders[key])}")
                    test_dataloader = test_dataloaders[key]
                    for batch in tqdm(test_dataloader, desc=f"dataset: {key} | number_of_docs: {number_of_docs}", disable=not accelerator.is_main_process):
                        batch = self.pop_unused_keys(batch)
                        answers = batch.pop('answers')
                        batch = accelerator.prepare(batch)
                        if 'knowledge_input_ids' in batch:
                            knowledge_outputs, compute_time = self.compute_hidden_states(batch, number_of_docs)
                            batch['knowledge_outputs'] = knowledge_outputs
                            hidden_states_time += compute_time
                            batch.pop('knowledge_input_ids')
                        # outputs = model.module.model.generate(**batch, max_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'], do_sample=False)
                        outputs = self.generate(batch, key)
                        outputs = outputs[:, batch['input_ids'].size(1):]
                        total_generated_tokens += outputs.size(1)*outputs.size(0)
                        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        inputs = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                        # Calculate metrics for each sample
                        for i in range(len(outputs)):
                            total_sample_count += 1
                            # if accelerator.is_main_process:
                                # print(f"Answer: {answers[i]} | Prediction: {outputs[i]} | Input: {inputs[i]}")
                            for metric in metrics:
                                metric_function = metrics_function_dict[metric]
                                metrics_accumulated[metric] += metric_function(outputs[i], answers[i], tokenizer)
                    
                    accelerator.wait_for_everyone()
                    # Normalize metrics by total sample count
                    for metric in metrics:
                        metrics_accumulated[metric] /= total_sample_count
                        metrics_accumulated[metric] = torch.tensor(metrics_accumulated[metric]).to(accelerator.device)

                        gathered_metric = accelerator.gather(metrics_accumulated[metric])
                        metric_result = gathered_metric.mean().item()
                        metrics_results_dict[metric].append(float(metric_result))

                        # Log the metric results
                        accelerator.log({f"test/{key}/{number_of_docs}/{metric}": metric_result}, step=iter_count)
                        accelerator.print(f"Step {iter_count} | Dataset: {key} | {metric.capitalize()}: {metric_result:.4f}")
                end_time = time()
                gpu_time = (end_time - start_time - hidden_states_time)*self.accelerator.num_processes
                tokens_per_second = self.accelerator.num_processes*total_generated_tokens/gpu_time
                accelerator.log({f"test/{number_of_docs}/total_tokens": self.accelerator.num_processes*total_generated_tokens}, step=iter_count)
                accelerator.log({f"test/{number_of_docs}/tokens_per_second": tokens_per_second}, step=iter_count)
                accelerator.log({f"test/{number_of_docs}/gpu_time": gpu_time}, step=iter_count)
                accelerator.log({f"test/{number_of_docs}/hidden_states_time": hidden_states_time*self.accelerator.num_processes}, step=iter_count)
                total_time = (end_time - start_time)*self.accelerator.num_processes
                accelerator.log({f"test/{number_of_docs}/total_time": total_time}, step=iter_count)
                # Log overall mean for each metric
                for metric in metrics:
                    print(metrics_results_dict[metric])
                    mean_metric = np.mean(metrics_results_dict[metric])
                    accelerator.log({f"test/mean_{metric}_{number_of_docs}": mean_metric}, step=iter_count)
                    print(f"Step {iter_count} | Mean {metric.capitalize()}: {mean_metric:.4f}")
                    metrics_dict[metric][number_of_docs] = mean_metric

            # Decide which metric to use for checkpoint saving based on target_metric
            best_metric = metrics_dict[target_metric][20]
            accelerator.print(f"\033[31mStep {iter_count} | Best {target_metric.capitalize()}: {best_metric:.4f}\033[0m")
            accelerator.log({f"test/target_metric_{target_metric}": best_metric}, step=iter_count)

            # Save best model checkpoint based on the target metric
            if accelerator.is_main_process:
                if best_metric > self.best_metric:
                    self.best_metric = best_metric
                    accelerator.print(f"New best {target_metric.capitalize()}: {best_metric:.4f}")
                    accelerator.unwrap_model(model).save_pretrained(f"{self.config['training']['project_dir']}/RAG-best/")
                # Save the latest model checkpoint regardless of performance
                accelerator.unwrap_model(model).save_pretrained(f"{self.config['training']['project_dir']}/RAG-new/")
        
        model.train()
                    

@register_class
class ReGPTLanguageModelTrainer(LanguageModelTrainer):
    def __init__(self, config):
        self.config = config
        ReGPT_kwargs = config['ReGPT_kwargs']
        generation_kwargs = config['generation_kwargs']
        self.config['training'].update(ReGPT_kwargs)
        self.config['training'].update(generation_kwargs)
        self.config['dataset']['train'].update(ReGPT_kwargs)
        self.config['dataset']['test'].update(ReGPT_kwargs)
        self.setup()
        self.best_perplexity = 1e10

    def setup_model(self, train_config):
        self.model = ReGPTForCausalLM(train_config)

    def setup_config(self, train_config, dataset_config):
        train_config['negative_depth'] = dataset_config['train']['negative_depth']
        dataset_config['train']['predict_from_last'] = train_config['predict_from_last']
        dataset_config['train']['train_or_test'] = 'train'
        dataset_config['test']['predict_from_last'] = train_config['predict_from_last']
        dataset_config['test']['train_or_test'] = 'test'
        return train_config, dataset_config

@register_class
class RAGLanguageModelTrainer(LanguageModelTrainer):
    def __init__(self, config):
        self.config = config
        RAG_kwargs = config['RAG_kwargs']
        self.config['training'].update(RAG_kwargs)
        super(RAGLanguageModelTrainer, self).__init__(config)

    def setup_model(self, train_config):
        self.model = RAGForCausalLM(train_config)
        self.model_config = self.model.model_config

    def compute_loss(self, outputs):
        lm_loss = outputs.loss
        retrieval_loss = outputs.retrieval_loss
        loss = lm_loss + retrieval_loss
        stats = {"training/lm_loss": float(lm_loss.cpu().detach().float().numpy()), "training/retrieval_loss": float(retrieval_loss.cpu().detach().float().numpy()), "training/loss": float(loss.cpu().detach().float().numpy())}
        return loss, stats
    
    def task_specific_stats(self, stats, model):
        # for i in range(self.config['training']['add_cross_attention_layer_number']):
            # stats[f'gate_score/{i}'] = float(self.accelerator.unwrap_model(model).model.base_model.layers[i].gate_crossattention.cpu().detach().float().numpy()[0])
        return stats
    
    def pop_unused_keys(self, record):
        return record

    def generate(self, batch, key):
        model = self.model
        if self.config.get('compare_speed', False):
            outputs = model.module.model.generate(**batch, max_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'], do_sample=False, pad_token_id=self.tokenizer.eos_token_id, min_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'])
        else:
            outputs = model.module.model.generate(**batch, max_new_tokens=self.config['dataset']['test'][key]['max_new_tokens'], do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return outputs
