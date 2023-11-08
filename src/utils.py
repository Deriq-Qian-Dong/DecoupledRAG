import functools
import torch.nn as nn
from typing import Dict, MutableMapping, Tuple, Union
import yaml
import torch.distributed as dist
import torch

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
    print_rank_0(f"Ratio of trainable parameters: {trainable_params / num_params:.2f}")


# 加载 YAML 配置文件
def get_config():
    with open("scripts/config.yaml", "r") as yaml_file:
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
