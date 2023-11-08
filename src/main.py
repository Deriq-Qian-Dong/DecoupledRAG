from utils import *
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch_optimizer as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from dataset_factory import DialogSFTDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
os.environ['http_proxy'] = 'http://172.19.57.45:3128'
os.environ['https_proxy'] = 'http://172.19.57.45:3128'

optimizer_class = {"AdamW": torch.optim.AdamW, "Lamb": optim.Lamb}
scheduler_class = {"CosineAnnealingLR": CosineAnnealingLR, "LinearLR": LinearLR}

def main():
    config = get_config()
    print_args(config)
    train_config = config['training']
    dataset_config = config['dataset']
    model = AutoModelForCausalLM.from_pretrained(train_config['model_name_or_path'])
    tokenizer = AutoTokenizer.from_pretrained(train_config['model_name_or_path'])
    tokenizer.pad_token = tokenizer.eos_token
    freeze_bottom_causal_layers(model.base_model, train_config['num_layers_unfrozen'])
    model.base_model.embed_tokens.weight.requires_grad = False
    print_trainable_params_stats(model)
    train_config["optimizer"]["kwargs"]['eps'] = float(train_config["optimizer"]["kwargs"]['eps'])
    optimizer = optimizer_class[train_config["optimizer"]["name"]](
        model.parameters(),
        **train_config["optimizer"]["kwargs"],
    )
    scheduler = scheduler_class[train_config["scheduler"]["name"]](optimizer, **train_config["scheduler"]["kwargs"])
    train_dataset = DialogSFTDataset(tokenizer, dataset_config['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=train_dataset._collate_fn)
    test_dataset = DialogSFTDataset(tokenizer, dataset_config['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=test_dataset._collate_fn)
    accelerator = Accelerator()
    (model, optimizer, train_dataloader, test_dataloader) = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)
    test(model, tokenizer, optimizer, scheduler, test_dataloader, accelerator, 0)
    for epoch in range(1, 1+train_config['num_epochs']):
        train(model, optimizer, train_dataloader, scheduler, accelerator, epoch)
        test(model, tokenizer, optimizer, scheduler, test_dataloader, accelerator, epoch)

def train(model, optimizer, train_dataloader, scheduler, accelerator, epoch):
    model.train()
    step = 0
    pbar = tqdm(total=len(train_dataloader))
    for batch in train_dataloader:
        batch = accelerator.prepare(batch)
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if accelerator.is_main_process:
            pbar.update(1)
            step += 1
            pbar.set_description(f"Epoch {epoch} | Step {step} |  Loss: {loss.cpu().detach().numpy():.4f}")
    pbar.close()

def test(model, tokenizer, optimizer, scheduler, test_dataloader, accelerator, epoch):
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
    accelerator.save(optimizer.state_dict(), directory)
    accelerator.save(scheduler.state_dict(), directory)
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(directory)
        tokenizer.save_pretrained(directory)

if __name__ == "__main__":
    main()
