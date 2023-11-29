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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


torch.manual_seed(random.randint(0, 1000000))

optimizer_class = {"AdamW": FusedAdam, "Lamb": optim.Lamb, "DeepSpeedCPUAdam": DeepSpeedCPUAdam}
scheduler_class = {"CosineAnnealingLR": CosineAnnealingLR, "LinearLR": LinearLR}
dataset_class = {"DialogSFTDataset": DialogSFTDataset, "CorpusPretrainDataset": CorpusPretrainDataset, "ReGPTDialogSFTDataset": ReGPTDialogSFTDataset, "ReGPTCorpusPretrainDataset": ReGPTCorpusPretrainDataset, "ReGPTLongDocumentSummarizationSFTDataset": ReGPTLongDocumentSummarizationSFTDataset,"ReGPTDocumentSummarizationSFTDataset":ReGPTDocumentSummarizationSFTDataset}

@dataclass
class ReGPTOutput(ModelOutput):
    loss: Optional[Tensor] = None

class ReGPTForCausalLM(nn.Module):
    def __init__(self, train_config):
        super(ReGPTForCausalLM, self).__init__()
        model = AutoModel.from_pretrained(train_config['model_name_or_path'], use_cache=not train_config['gradient_checkpointing'])
        freeze_bottom_causal_layers(model.base_model, train_config['num_layers_unfrozen'])
        try:
            # llama2
            model.base_model.embed_tokens.weight.requires_grad = False
        except:
            # gpt2
            model.base_model.wte.weight.requires_grad = False
            model.base_model.wpe.weight.requires_grad = False
        self.input_linear_proj = nn.Linear(train_config['faiss']['dimension'], model.config.hidden_size)
        self.linear_proj = nn.Linear(model.config.hidden_size, train_config['faiss']['dimension'])
        if os.path.exists(os.path.join(train_config['model_name_or_path'], 'input_linear_proj.pt')):
            self.input_linear_proj.load_state_dict(torch.load(os.path.join(train_config['model_name_or_path'], 'input_linear_proj.pt'), map_location='cpu'))
        if os.path.exists(os.path.join(train_config['model_name_or_path'], 'linear_proj.pt')):
            self.linear_proj.load_state_dict(torch.load(os.path.join(train_config['model_name_or_path'], 'linear_proj.pt'), map_location='cpu'))
        print_trainable_params_stats(model)
        if train_config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        self.model = model
        # self.searcher = Searcher(train_config['faiss']['index_type'], dimension=train_config['faiss']['dimension'], nprobe=train_config['faiss']['nprobe'])
        # phrases = np.load(open(train_config['faiss']['phrases_path'], 'rb'))
        matrix = np.load(open(train_config['faiss']['matrix_path'], 'rb'))
        # self.searcher._build(matrix, phrases, speedup=False)
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
        inputs_embeds = self.input_linear_proj(inputs_embeds) # [batch_size, seq_len, hidden_size]
        negative_embeds = self.matrix[negative_ids.cpu()]
        negative_embeds = torch.from_numpy(negative_embeds).to(input_ids.device) 
        positive_embeds = self.matrix[labels.cpu()]  # [batch_size, predict_from_last, hidden_size]
        positive_embeds = torch.from_numpy(positive_embeds).to(input_ids.device) 
        embeds_for_contrastive_training = torch.cat([positive_embeds.unsqueeze(2), negative_embeds], dim=2).to(self.dtype).contiguous()  # [batch_size, predict_from_last, 1+negative_depth, hidden_size]
        kwargs['inputs_embeds'] = inputs_embeds
        kwargs['output_hidden_states'] = True
        outputs = self.model(**kwargs)

        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        last_hidden_state = last_hidden_state[:, -predict_from_last-1:-1, :].contiguous()  # [batch_size, predict_from_last, hidden_size]
        q_reps = self.linear_proj(last_hidden_state).view(-1, embeds_for_contrastive_training.shape[-1])  # [batch_size*predict_from_last, hidden_size]
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
    
    def save_pretrained(self, directory):
        self.model.save_pretrained(directory)
        torch.save(self.input_linear_proj.state_dict(), os.path.join(directory, 'input_linear_proj.pt'))
        torch.save(self.linear_proj.state_dict(), os.path.join(directory, 'linear_proj.pt'))

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
                inputs_embeds = self.input_linear_proj(inputs_embeds) # [batch_size, cur_seq_len, hidden_size]
                kwargs['inputs_embeds'] = inputs_embeds
                outputs = self.model(**kwargs)
                last_hidden_state = outputs.last_hidden_state  # [batch_size, cur_seq_len, hidden_size]
                last_hidden_state = last_hidden_state[:, -1:, :].contiguous()  # [batch_size, 1, hidden_size]
                q_reps = self.linear_proj(last_hidden_state).view(-1, self.matrix.shape[-1])  # [batch_size, hidden_size]
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



class LanguageModelTrainer:
    def __init__(self, config):
        self.config = config
        self.setup()

    def run(self):
        self.test()
        for epoch in range(1, 1+self.train_config['num_epochs']):
            self.epoch = epoch
            self.train()
            self.test()
    
    def setup_model(self, train_config):
        model = AutoModelForCausalLM.from_pretrained(train_config['model_name_or_path'], use_cache=not train_config['gradient_checkpointing'])
        freeze_bottom_causal_layers(model.base_model, train_config['num_layers_unfrozen'])
        try:
            # llama2
            model.base_model.embed_tokens.weight.requires_grad = train_config['num_layers_unfrozen']<=0
        except:
            # gpt2
            model.base_model.wte.weight.requires_grad = train_config['num_layers_unfrozen']<=0
            model.base_model.wpe.weight.requires_grad = train_config['num_layers_unfrozen']<=0
        print_trainable_params_stats(model)
        if train_config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        self.model = model

    def setup_dataloader(self, dataset_config, tokenizer):
        train_dataset = dataset_class[dataset_config['train']['dataset_name']](tokenizer, dataset_config['train'])
        self.train_dataloader = DataLoader(train_dataset, batch_size=dataset_config['train']['batch_size'], shuffle=True, collate_fn=train_dataset._collate_fn)
        test_dataset = dataset_class[dataset_config['test']['dataset_name']](tokenizer, dataset_config['test'])
        self.test_dataloader = DataLoader(test_dataset, batch_size=dataset_config['test']['batch_size'], shuffle=False, collate_fn=test_dataset._collate_fn)

    def setup(self):
        config = self.config
        train_config = config['training']
        dataset_config = config['dataset']
        train_config['negative_depth'] = dataset_config['train']['negative_depth']
        dataset_config['train']['predict_from_last'] = train_config['predict_from_last']
        dataset_config['train']['train_or_test'] = 'train'
        dataset_config['test']['predict_from_last'] = train_config['predict_from_last']
        dataset_config['test']['train_or_test'] = 'test'
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

        self.setup_dataloader(dataset_config, tokenizer)

        accelerator = Accelerator(log_with=train_config['log_with'], project_dir=train_config['project_dir'])
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        accelerator.init_trackers(project_name=f'{train_config["project_name"]}_{timestamp}')
        (model, optimizer, self.train_dataloader, self.test_dataloader) = accelerator.prepare(model, optimizer, self.train_dataloader, self.test_dataloader)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.epoch = 0
        self.iter_count = 0
        print_args(config)

    def train(self):
        model, optimizer, train_dataloader, scheduler, accelerator, epoch = self.model, self.optimizer, self.train_dataloader, self.scheduler, self.accelerator, self.epoch
        model.train()
        step = 0
        pbar = tqdm(total=len(train_dataloader))
        for batch in train_dataloader:
            self.iter_count += 1
            total_time = time()
            seq_len = batch['input_ids'].size(1)
            batch = accelerator.prepare(batch)
            forward_time = time()
            outputs = model(**batch)
            forward_time = time() - forward_time
            loss = outputs.loss
            stats = {"loss": float(loss.cpu().detach().float().numpy())}
            stats["seq_len"] = seq_len
            backward_time = time()
            accelerator.backward(loss)
            backward_time = time() - backward_time
            stats["time/forward"] = forward_time
            stats["time/backward"] = backward_time
            opt_time = time()
            for group_number, lr in enumerate(scheduler.get_last_lr()):
                stats[f"learning_rate"] = lr
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.is_main_process:
                pbar.update(1)
                step += 1
                pbar.set_description(f"Epoch {epoch} | Step {step} | Loss: {loss.cpu().detach().float().numpy():.4f}")
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

    def test(self):
        model, tokenizer, optimizer, scheduler, test_dataloader, accelerator, iter_count = self.model, self.tokenizer, self.optimizer, self.scheduler, self.test_dataloader, self.accelerator, self.iter_count
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Evaluation of step {iter_count}", disable=not accelerator.is_main_process):
                batch = accelerator.prepare(batch)
                outputs = model(**batch)
                loss = outputs.loss
                loss = accelerator.gather_for_metrics(loss)
                total_loss += loss.cpu().detach().float().numpy().mean()
        total_loss /= len(test_dataloader)
        perplexity = np.exp(total_loss)
        accelerator.print(f"Step {iter_count} | Perplexity: {perplexity:.4f} | Loss: {total_loss:.4f}")
        directory = f"output/SFT-step-{iter_count}/"
        accelerator.wait_for_everyone()
        stats = {"test/perplexity": perplexity, "test/loss": total_loss}
        accelerator.log(stats, step=self.iter_count)
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(directory)
            tokenizer.save_pretrained(directory)
        model.train()

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

    def setup_model(self, train_config):
        self.model = ReGPTForCausalLM(train_config)
