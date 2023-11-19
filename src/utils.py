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