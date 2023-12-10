import os
import sys
from utils import *
from model import LanguageModelTrainer, ReGPTLanguageModelTrainer


os.environ['http_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'
os.environ['https_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'

TrainerClass = {'LanguageModelTrainer': LanguageModelTrainer, 'ReGPTLanguageModelTrainer': ReGPTLanguageModelTrainer}

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    trainer = TrainerClass[config['trainer']](config)
    trainer.run()
