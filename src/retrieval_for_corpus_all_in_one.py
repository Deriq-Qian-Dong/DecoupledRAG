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
from utils import merge, read_embed, search, get_config, dict_to_HParams
from datasets import load_from_disk

SEED = 2024
best_mrr=-1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


def main_multi(args, model, optimizer, writer, corpus_embeddings=None):
    epoch = 0
    local_rank = torch.distributed.get_rank()

    # 加载数据集
    validate_multi_gpu(model, None, None, epoch, args, writer, args.corpus_name, corpus_embeddings)

def validate_multi_gpu(model, query_loader, passage_loader, epoch, args, writer, corpus_name, corpus_embeddings=None):
    local_start = time.time()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    top_k = args.top_k    
    shard_cnt = corpus_embeddings.shape[0]//world_size
    if local_rank==world_size-1:
        para_embs = corpus_embeddings[local_rank*shard_cnt:]
    else:
        para_embs = corpus_embeddings[local_rank*shard_cnt:(local_rank+1)*shard_cnt]
    q_output_file_name = f'{args.model_out_dir}/para_embs.part{local_rank}.{corpus_name}.npy'
    np.save(q_output_file_name, para_embs)
    engine = torch.from_numpy(corpus_embeddings).cuda()
    qid_list = list(range(0, corpus_embeddings.shape[0]))
    qid_list = [str(qid) for qid in qid_list]
    shard_cnt = len(qid_list)//world_size
    if local_rank==world_size-1:
        qid_list = qid_list[local_rank*shard_cnt:]
    else:
        qid_list = qid_list[local_rank*shard_cnt:(local_rank+1)*shard_cnt]
    search(engine, q_output_file_name, qid_list, f"{args.model_out_dir}/res.top%d.part%d.step%d.%s"%(top_k, local_rank, epoch, corpus_name), top_k=top_k, use_faiss=args.use_faiss)
    torch.distributed.barrier() 
    if local_rank==0:
        f_list = []
        for part in range(world_size):
            f_list.append(f'{args.model_out_dir}/res.top%d.part%d.step%d.%s' % (top_k, part, epoch, corpus_name))
        res = []
        for f in f_list:
            with open(f, 'r') as f:
                res += f.readlines()
        with open(f'{args.model_out_dir}/res.top%d.step%d.%s'%(top_k, epoch, corpus_name), 'w') as f:
            f.writelines(res)

        
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

    print(args.model_out_dir)
    os.makedirs(args.model_out_dir, exist_ok=True)
    corpus_embeddings = None
    if args.corpus_embeddings:
        corpus_embeddings = np.load(args.corpus_embeddings)
    main_multi(args, None, None, writer, corpus_embeddings)

if __name__ == '__main__':
    config_path = sys.argv[1]
    main_cli(config_path)
