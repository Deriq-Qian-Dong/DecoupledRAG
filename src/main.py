import os
import sys
from utils import *
from model import LanguageModelTrainer, ReGPTLanguageModelTrainer, RAGLanguageModelTrainer, RAGLanguageModelTester, RAGQATester

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'

TrainerClass = {'LanguageModelTrainer': LanguageModelTrainer, 'ReGPTLanguageModelTrainer': ReGPTLanguageModelTrainer, "RAGLanguageModelTrainer":RAGLanguageModelTrainer, "RAGLanguageModelTester":RAGLanguageModelTester, "RAGQATester":RAGQATester}

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    trainer = TrainerClass[config['trainer']](config)
    trainer.run()
