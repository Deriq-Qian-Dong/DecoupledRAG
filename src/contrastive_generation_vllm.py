from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm

class QADataset4Chat(Dataset):
    def __init__(self, tokenizer, args):
        self.args = args
        self.number_of_docs = args['number_of_docs']
        print('number_of_docs:', self.number_of_docs, 'path:', args['data_name_or_path'])
        self.tokenizer = tokenizer
        self.setup_datasets()
        self.corpus = load_from_disk(args['corpus'])

    def setup_datasets(self):
        self.datasets = load_from_disk(self.args['data_name_or_path'])
    
    def __getitem__(self, idx):
        sample = self.datasets[idx]
        query = sample.get('query', sample.get('question', ''))
        answer = sample.get('answers', [sample.get('answer', '')])[0]
        retrieved_docs = self.corpus[sample['neighbors']]['text']
        query = query + '\nThe answer MUST in ONE OR FEW WORDS.'
        chat = [{'role': 'user', 'content': query}]
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return chat
    
    def __len__(self):
        return len(self.datasets)

    def get_batch(self, indices):
        return [self.__getitem__(idx) for idx in indices]

def initialize_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

def initialize_llm(model_path, tensor_parallel_size, batch_size):
    return LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, max_num_seqs=8192, gpu_memory_utilization=0.99)

def process_batches(datasets, llm, batch_size, sampling_params):
    new_data = []
    num_batches = len(datasets) // batch_size + (1 if len(datasets) % batch_size != 0 else 0)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(datasets))
        batch = datasets.get_batch(range(start_idx, end_idx))
        batch_outputs = llm.generate(batch, sampling_params, use_tqdm=False)
        
        for outputs in batch_outputs:
            sample = {'prompt': outputs.prompt, "answers": [out.text for out in outputs.outputs]}
            new_data.append(sample)
    
    return new_data

def save_dataset(data, path):
    dataset = HFDataset.from_list(data)
    dataset.save_to_disk(path)

def main(data_name_or_path, output_path):
    model_name_or_path = "../llama3-chat"
    dataset_config = {
        "number_of_docs": 1,
        "data_name_or_path": data_name_or_path,
        "corpus": '../data_of_ReGPT/Wiki-corpus/train',
        "max_seq_len": 512,
    }
    tokenizer = initialize_tokenizer(model_name_or_path)
    datasets = QADataset4Chat(tokenizer, dataset_config)
    batch_size = 8192
    llm = initialize_llm(model_name_or_path, tensor_parallel_size=8, batch_size=batch_size)
    
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, top_k=100, n=4, max_tokens=256)
    new_data = process_batches(datasets, llm, batch_size, sampling_params)
    
    save_dataset(new_data, output_path)

if __name__ == "__main__":
    candidates = ['2WikiMultihopQA/', 'hotpotqa/', 'nq/', 'openbookqa/', 'truthful_qa/']
    for candidate in candidates:
        print(f"Processing {candidate}...")
        data_path = f"../data_of_ReGPT/QA_datasets_woEmb/{candidate}/sorted_datasets_train"
        output_path = data_path.replace("QA_datasets_woEmb", "QA_datasets_contrastive")
        main(data_path, output_path)

