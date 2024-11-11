import os
import sys
from utils import *
from model import *
from registry import registry

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'

def TrainerClass(class_name):
    cls = registry.get_class(class_name)
    if cls:
        return cls
    else:
        raise ValueError(f"Class {class_name} not found")

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    trainer = TrainerClass(config['trainer'])(config)
    trainer.run()
