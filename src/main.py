import os
import sys
import torch
import datetime
import numpy as np
from utils import *
from tqdm import tqdm
from time import time
import torch_optimizer as optim
from accelerate import Accelerator
from dataset_factory import DialogSFTDataset
from torch.utils.data import DataLoader, Dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

optimizer_class = {"AdamW": FusedAdam, "Lamb": optim.Lamb, "DeepSpeedCPUAdam": DeepSpeedCPUAdam}
scheduler_class = {"CosineAnnealingLR": CosineAnnealingLR, "LinearLR": LinearLR}

class LanguageModelTrainer:
    def __init__(self, config):
        self.config = config
        self.setup()

    def run(self):
        # self.test()
        for epoch in range(1, 1+self.train_config['num_epochs']):
            self.epoch = epoch
            self.train()
            self.test()
    
    def setup(self):
        config = self.config
        print_args(config)
        train_config = config['training']
        dataset_config = config['dataset']
        model = AutoModelForCausalLM.from_pretrained(train_config['model_name_or_path'], use_cache=not train_config['gradient_checkpointing'])
        tokenizer = AutoTokenizer.from_pretrained(train_config['model_name_or_path'])
        tokenizer.pad_token = tokenizer.eos_token
        freeze_bottom_causal_layers(model.base_model, train_config['num_layers_unfrozen'])
        try:
            # llama2
            model.base_model.embed_tokens.weight.requires_grad = train_config['num_layers_unfrozen']>0
        except:
            # gpt2
            model.base_model.wte.weight.requires_grad = train_config['num_layers_unfrozen']>0
            model.base_model.wpe.weight.requires_grad = train_config['num_layers_unfrozen']>0
        print_trainable_params_stats(model)
        train_config["optimizer"]["kwargs"]['eps'] = float(train_config["optimizer"]["kwargs"]['eps'])
        params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
        params = {'params': [v for k, v in params]}
        optimizer = optimizer_class[train_config["optimizer"]["name"]](
            [params],
            **train_config["optimizer"]["kwargs"],
        )
        scheduler = scheduler_class[train_config["scheduler"]["name"]](optimizer, **train_config["scheduler"]["kwargs"])
        train_dataset = DialogSFTDataset(tokenizer, dataset_config['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=train_dataset._collate_fn)
        test_dataset = DialogSFTDataset(tokenizer, dataset_config['test'])
        test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=test_dataset._collate_fn)
        accelerator = Accelerator(log_with=train_config['log_with'], project_dir=train_config['project_dir'])
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        accelerator.init_trackers(project_name=f'{train_config["project_name"]}_{timestamp}')
        (model, optimizer, train_dataloader, test_dataloader) = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)
        if train_config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.accelerator = accelerator
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.epoch = 0
        self.iter_count = 0

    def train(self):
        model, optimizer, train_dataloader, scheduler, accelerator, epoch = self.model, self.optimizer, self.train_dataloader, self.scheduler, self.accelerator, self.epoch
        model.train()
        step = 0
        pbar = tqdm(total=len(train_dataloader))
        for batch in train_dataloader:
            self.iter_count += 1
            batch = accelerator.prepare(batch)
            forward_time = time()
            outputs = model(**batch)
            forward_time = time() - forward_time
            loss = outputs.loss
            stats = {"loss": float(loss.cpu().detach().numpy())}
            backward_time = time()
            accelerator.backward(loss)
            backward_time = time() - backward_time
            stats["time/forward"] = forward_time
            stats["time/backward"] = backward_time
            for group_number, lr in enumerate(scheduler.get_last_lr()):
                stats[f"learning_rate"] = lr
            accelerator.log(stats, step=self.iter_count)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.is_main_process:
                pbar.update(1)
                step += 1
                pbar.set_description(f"Epoch {epoch} | Step {step} | Loss: {loss.cpu().detach().numpy():.4f}")
        pbar.close()

    def test(self):
        model, tokenizer, optimizer, scheduler, test_dataloader, accelerator, epoch = self.model, self.tokenizer, self.optimizer, self.scheduler, self.test_dataloader, self.accelerator, self.epoch
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Evaluation of epoch {epoch}"):
                batch = accelerator.prepare(batch)
                outputs = model(**batch)
                loss = outputs.loss
                loss = accelerator.gather_for_metrics(loss)
                total_loss += loss.cpu().detach().numpy().mean()
        total_loss /= len(test_dataloader)
        perplexity = np.exp(total_loss)
        accelerator.print(f"Epoch {epoch} | Perplexity: {perplexity:.4f} | Loss: {total_loss:.4f}")
        directory = f"output/SFT-{epoch}/"
        accelerator.wait_for_everyone()
        stats = {"test/perplexity": perplexity, "test/loss": total_loss}
        accelerator.log(stats, step=self.iter_count)
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(directory)
            tokenizer.save_pretrained(directory)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    trainer = LanguageModelTrainer(config)
    trainer.run()
