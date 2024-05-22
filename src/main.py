import os
import sys
from utils import *
from model import LanguageModelTrainer, ReGPTLanguageModelTrainer, RAGLanguageModelTrainer, RAGLanguageModelTester

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['http_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'
os.environ['https_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'

TrainerClass = {'LanguageModelTrainer': LanguageModelTrainer, 'ReGPTLanguageModelTrainer': ReGPTLanguageModelTrainer, "RAGLanguageModelTrainer":RAGLanguageModelTrainer, "RAGLanguageModelTester":RAGLanguageModelTester}

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    trainer = TrainerClass[config['trainer']](config)
    trainer.run()
