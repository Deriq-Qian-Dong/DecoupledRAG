import functools
import torch.nn as nn
from typing import Dict, MutableMapping, Tuple, Union
import yaml
import torch.distributed as dist
import torch
import faiss
import joblib
import torch.nn.functional as F
import numpy as np
import json
def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_from_json(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def print_rank_0(msg):
    """Print from process with rank 0 only."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg, flush=True)
    else:
        print(msg, flush=True)

def print_args(args: MutableMapping[str, object], depth: int = 0):
    """Prints the arguments passed to the script."""
    prefix = "\t" * depth
    for k, v in args.items():
        if isinstance(v, Dict):
            print_rank_0(f"{prefix}{k}:")
            print_args(v, depth + 1)
        else:
            print_rank_0(f"{prefix}{k}: {v}")

def print_trainable_params_stats(model: nn.Module):
    """Prints the number of trainable parameters in the specified model."""
    num_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"Number of parameters: {num_params/1000000000:.2}B")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Number of trainable parameters: {trainable_params/1000000}M")
    print_rank_0(f"Ratio of trainable parameters: {trainable_params / num_params:.2%}")
    for para_name, para in model.named_parameters():
        if para.requires_grad==True:
            print_rank_0(f"Trainable parameter: {para_name}")


# 加载 YAML 配置文件
def get_config(path="config/rellama_config.yaml"):
    with open(path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")
    
def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "embed_tokens",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "transformer.blocks",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)

def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)

def freeze_non_crossattention_parameters(model: nn.Module, freeze_retrieval_head=False, freeze_lm_head=True):
    """Freezes non cross-attention parameters of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    hidden_layers_to_processing = list(hidden_layers)
    if freeze_lm_head:
        hidden_layers_to_processing.append(findattr(model, ("lm_head", "model.lm_head")))
    if freeze_retrieval_head:
        try:
            hidden_layers_to_processing.append(findattr(model, ("retrieval_head", "model.retrieval_head")))
        except:
            pass
    hidden_layers_to_processing.append(findattr(model, ("model.norm",)))
    for layer in hidden_layers_to_processing:
        for para_name, para in layer.named_parameters():
            if "crossattention" not in para_name:
                para.requires_grad_(False)
            if 'lora' in para_name or 'crossattention' in para_name or 'knowledge_injector' in para_name:
                para.requires_grad_(True)

class Searcher:

    def __init__(self, index_type, dimension=4096, nprobe=1):
        # self.searcher = faiss.index_factory(dimension, index_type, faiss.METRIC_INNER_PRODUCT)
        self.searcher = faiss.index_factory(dimension, index_type, faiss.METRIC_INNER_PRODUCT)
        self.corpus = []
        self.nprobe = nprobe
        self.index_type = index_type

    def _build(self, matrix, corpus, speedup=False):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        if speedup:
            self.move_to_gpu()
        # self.searcher.train(matrix)
        self.searcher.add(matrix)
        # if speedup:
            # self.move_to_cpu()
        print_rank_0(f'[!] build collection with {self.searcher.ntotal} samples')
    
    def _search(self, vector, topk=20):
        # self.searcher.nprobe = self.nprobe
        D, I = self.searcher.search(vector, topk)
        rest = [[self.corpus[i] for i in N] for N in I]
        distance = [[i for i in N] for N in D]
        return rest, distance

    def save(self, path_faiss, path_corpus, path_source_corpus=None):
        faiss.write_index(self.searcher, path_faiss)
        with open(path_corpus, 'wb') as f:
            joblib.dump(self.corpus, f)

    def load(self, path_faiss, path_corpus, path_source_corpus=None):
        self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)
        print_rank_0(f'[!] load {len(self.corpus)} utterances from {path_faiss} and {path_corpus}')

    def add(self, vectors, texts):
        '''the whole source information are added in _build'''
        self.searcher.add(vectors)
        self.corpus.extend(texts)
        print_rank_0(f'[!] add {len(texts)} dataset over')

    def move_to_gpu(self, device=0):
        res = faiss.StandardGpuResources()
        self.searcher = faiss.index_cpu_to_gpu(res, device, self.searcher)
        print_rank_0(f'[!] move index to GPU device: {device} over')
    
    def move_to_cpu(self):
        self.searcher = faiss.index_gpu_to_cpu(self.searcher)
        print_rank_0(f'[!] move index from GPU to CPU over')


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


def search(index, emb_file, qid_list, outfile, top_k, use_faiss=False):
    search_torch(index, emb_file, qid_list, outfile, top_k)

def search_torch(index, emb_file, qid_list, outfile, top_k):
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in read_embed(emb_file):
            q_emb_matrix = np.array(batch_vec)
            q_emb_matrix = torch.from_numpy(q_emb_matrix)
            q_emb_matrix = q_emb_matrix.cuda()
            top_k = min(top_k, len(index))
            res_dist, res_p_id = topk_query_passage(q_emb_matrix, index, top_k)
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, j+1, score))
                q_idx += 1

from tqdm import tqdm
def read_embed(file_name, dim=768, bs=4):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        with tqdm(total=len(emb_np)//bs+1) as pbar:
            while(i < len(emb_np)):
                vec_list = emb_np[i:i+bs]
                i += bs
                pbar.update(1)
                yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in tqdm(inp):
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                assert len(vector) == dim
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list
import torch
def topk_query_passage(query_vector, passage_vector, k):
    """
    对query vector和passage vector进行内积计算，并返回top k的索引

    Args:
        query_vector (torch.Tensor): query向量，形状为 (batch_size, query_dim)
        passage_vector (torch.Tensor): passage向量，形状为 (batch_size, passage_dim)
        k (int): 返回的top k值

    Returns:
        torch.Tensor: top k值的索引，形状为 (batch_size, k)
    """
    # 计算query向量和passage向量的内积
    scores = torch.matmul(query_vector, passage_vector.t())  # 形状为 (batch_size, batch_size)

    # 对每个batch进行排序，取top k值
    k = min(k, scores.size(1))
    res_dist, res_p_id = torch.topk(scores, k=k, dim=1)  # 形状为 (batch_size, k)

    return res_dist.cpu().numpy(), res_p_id.cpu().numpy()

def merge(total_part, shift, top, eval_cnts, query_dataset_name, output):
    f_list = []
    for part in range(total_part):
        f0 = open(f'{output}/res.top%d.part%d.step%d.%s' % (top, part, eval_cnts, query_dataset_name))
        f_list.append(f0)

    line_list = []
    for part in range(total_part):
        line = f_list[part].readline()
        line_list.append(line)

    out = open(f'{output}/res.top%d.step%d.%s' % (top, eval_cnts, query_dataset_name), 'w')
    last_q = ''
    ans_list = {}
    while len(line_list):
        cur_list = []
        for line in line_list:
            sub = line.strip().split('\t')
            cur_list.append(sub)

        if last_q == '':
            last_q = cur_list[0][0]
        if cur_list[0][0] != last_q:
            rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
            for i in range(min(top, len(rank))):
                out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
            ans_list = {}
        for i, sub in enumerate(cur_list):
            ans_list[int(sub[1]) + shift*i] = float(sub[-1])
        last_q = cur_list[0][0]

        line_list = []
        for f0 in f_list:
            line = f0.readline()
            sub = line.strip().split('\t')
            if sub[-1]=='':
                continue
            line_list.append(line)

    rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
    for i in range(min(top, len(rank))):
        out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
    out.close()


def dict_to_HParams(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_HParams(value)
    return HParams(**d)
class HParams(object):
    """Hyper paramerter"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise ValueError('key(%s) not in HParams.' % key)
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.to_dict())

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    @classmethod
    def from_json(cls, json_str):
        """doc"""
        d = json.loads(json_str)
        if type(d) != dict:
            raise ValueError('json object must be dict.')
        return HParams.from_dict(d)

    def get(self, key, default=None):
        """doc"""
        return self.__dict__.get(key, default)

    @classmethod
    def from_dict(cls, d):
        """doc"""
        if type(d) != dict:
            raise ValueError('input must be dict.')
        hp = HParams(**d)
        return hp

    def to_json(self):
        """doc"""
        return json.dumps(self.__dict__)

    def to_dict(self):
        """doc"""
        return self.__dict__
    
    def print_config(self):
        for key,value in self.__dict__.items():
            print(key+":",value)

    def join(self, other):
        """doc"""
        if not isinstance(other, HParams):
            raise ValueError('input must be HParams instance.')
        self.__dict__.update(**other.__dict__)
        return self
