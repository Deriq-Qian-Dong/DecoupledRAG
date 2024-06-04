import os
import sys
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import argparse
import random
import subprocess
import tempfile
import time
from collections import defaultdict

import dataset_factory
import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_optimizer as optim
import utils
from modeling import DualEncoder
from torch import distributed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BertModel
from utils import build_engine, load_qid, merge, read_embed, search, save_to_json, load_from_json, get_config, dict_to_HParams

SEED = 2024
best_mrr=-1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def main_multi(args, model, optimizer, writer):
    epoch = 0
    local_rank = torch.distributed.get_rank()

    # 加载数据集
    query_dataset = dataset_factory.QueryDataset(args)
    query_loader = DataLoader(query_dataset, batch_size=args.dev_batch_size, collate_fn=query_dataset._collate_fn, num_workers=3)
    passage_dataset = dataset_factory.PassageDataset(args)
    passage_loader = DataLoader(passage_dataset, batch_size=args.dev_batch_size, collate_fn=passage_dataset._collate_fn, num_workers=3)
    validate_multi_gpu(model, query_loader, passage_loader, epoch, args, writer, args.corpus_name)


def validate_multi_gpu(model, query_loader, passage_loader, epoch, args, writer, corpus_name):
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    top_k = args.top_k
    q_output_file_name = f'{args.model_out_dir}/query.emb.step%d.npy'%epoch
    if local_rank==0:
        q_embs = []
        with torch.no_grad():
            model.eval()
            for records in query_loader:
                with autocast():
                    output = model(query_inputs=_prepare_inputs(records))
                    q_reps = output['q_reps']
                q_embs.append(q_reps.cpu().detach().numpy())
        emb_matrix = np.concatenate(q_embs, axis=0)
        np.save(q_output_file_name, emb_matrix)
        print("predict q_embs cnt: %s" % len(emb_matrix))
    with torch.no_grad():
        model.eval()
        para_embs = []
        for records in tqdm(passage_loader, disable=args.local_rank>0):
            with autocast():
                output = model(passage_inputs=_prepare_inputs(records))
                p_reps = output['p_reps']
            para_embs.append(p_reps.cpu().detach().numpy())
    torch.distributed.barrier() 
    para_embs = np.concatenate(para_embs, axis=0)
    print("predict embs cnt: %s" % len(para_embs))
    print('create index done!')
    if args.use_faiss:
        engine = build_engine(para_embs, dim=para_embs.shape[1])
    else:
        engine = torch.from_numpy(para_embs).cuda()
    qid_list = load_qid(args.dev_query)
    search(engine, q_output_file_name, qid_list, f"{args.model_out_dir}/res.top%d.part%d.step%d.%s"%(top_k, local_rank, epoch, corpus_name), top_k=top_k, use_faiss=args.use_faiss)
    torch.distributed.barrier() 
    if local_rank==0:
        f_list = []
        for part in range(world_size):
            f_list.append(f'{args.model_out_dir}/res.top%d.part%d.step%d.%s' % (top_k, part, epoch, corpus_name))
        shift = para_embs.shape[0]
        merge(world_size, shift, top_k, epoch, corpus_name, args.model_out_dir)

        
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt

def _prepare_inputs(record):
    prepared = {}
    local_rank = torch.distributed.get_rank()
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        elif x is None:
            prepared[key] = x
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared

def main_cli(config_path):
    args = get_config(config_path)
    args = dict_to_HParams(args)
    args.learning_rate = float(args.learning_rate)
    args.model_name_or_path = args.retriever_model_name_or_path
    config = args
    # 加载到多卡
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    args.local_rank = local_rank
    writer = None
    if local_rank==0:
        args.print_config()
        
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = DualEncoder(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    params = {'params': [v for k, v in params]}
    # optimizer = torch.optim.Adam([params], lr=args.learning_rate, weight_decay=0.0)
    optimizer = optim.Lamb([params], lr=args.learning_rate, weight_decay=0.0)

    if args.warm_start_from:
        print('warm start from ', args.warm_start_from)
        state_dict = torch.load(args.warm_start_from, map_location=device)
        for k in list(state_dict.keys()):
            state_dict[k.replace('module.','')] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print("model loaded on GPU%d"%local_rank)
    print(args.model_out_dir)
    os.makedirs(args.model_out_dir, exist_ok=True)


    main_multi(args, model, optimizer, writer)

if __name__ == '__main__':
    config_path = sys.argv[1]
    main_cli(config_path)
