from torch.utils.data import DataLoader, Dataset
from datasets import concatenate_datasets, load_from_disk, Dataset as HFDataset
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
    
class QADataset4ChatWithBackgroundKnowledge(QADataset4Chat):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)

    def __getitem__(self, idx):
        sample = self.datasets[idx]
        tokenizer = self.tokenizer
        query = tokenizer.decode(tokenizer.encode(sample['prompt'], add_special_tokens=False), skip_special_tokens=True).replace('user\n\n', '').replace('assistant\n\n', '')
        answers = sample['answers']
        chats = []
        for answer in answers:
            chat = [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': answer}]
            chat.append({'role': 'user', 'content': 'Please provide the background knowledge for the answer. The background knowledge MUST be within 256 tokens.'})
            chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            chats.append(chat)
        return chats
    
    def get_batch(self, indices):
        batchs = []
        for idx in indices:
            batchs.extend(self.__getitem__(idx))
        return batchs


def initialize_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

def initialize_llm(model_path):
    return LLM(model=model_path, tensor_parallel_size=1, max_num_seqs=8192, gpu_memory_utilization=0.99, swap_space=512)

def process_batches(datasets, llm, batch_size, sampling_params, key_name="answers", replace_str="", num_gpu=8, local_rank=0):
    new_data = []
    num_batches = len(datasets) // batch_size + (1 if len(datasets) % batch_size != 0 else 0)

    # Split datasets for each GPU
    gpu_batch_indices = [i for i in range(local_rank, num_batches, num_gpu)]


    for batch_idx in tqdm(gpu_batch_indices):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(datasets))
        batch = datasets.get_batch(range(start_idx, end_idx))
        batch_outputs = llm.generate(batch, sampling_params, use_tqdm=False)
        
        for outputs in batch_outputs:
            generated = [out.text for out in outputs.outputs]
            prompt = outputs.prompt
            if replace_str:
                prompt = prompt.replace(replace_str, '')
            if len(generated) == 1:
                generated = generated[0]
            sample = {'prompt': prompt, key_name: generated}
            new_data.append(sample)
    
    return new_data

def save_dataset(data, path):
    dataset = HFDataset.from_list(data)
    dataset.save_to_disk(path)

def _generate_contrastive_answers(data_name_or_path, output_path, llm, model_name_or_path, num_gpu=8, local_rank=0):
    dataset_config = {
        "number_of_docs": 1,
        "data_name_or_path": data_name_or_path,
        "corpus": '../data_of_ReGPT/Wiki-corpus/train',
        "max_seq_len": 512,
    }
    tokenizer = initialize_tokenizer(model_name_or_path)
    datasets = QADataset4Chat(tokenizer, dataset_config)
    batch_size = 1024
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, top_k=100, n=4, max_tokens=256)
    new_data = process_batches(datasets, llm, batch_size, sampling_params, "answers", "", num_gpu, local_rank)
    
    save_dataset(new_data, output_path)

def generate_contrastive_answers(num_gpu=8, local_rank=0):
    candidates = ['hotpotqa/', '2WikiMultihopQA/']
    model_name_or_path = "../llama3_chat_qa_sft"
    llm = initialize_llm(model_name_or_path)
    for candidate in candidates:
        print(f"Processing {candidate}...")
        data_path = f"../data_of_ReGPT/QA_datasets_woEmb/{candidate}/sorted_datasets_train"
        output_path = data_path.replace("QA_datasets_woEmb", "QA_datasets_contrastive")
        _generate_contrastive_answers(data_path, output_path, llm, model_name_or_path, num_gpu, local_rank)

def _generate_background_knowledge_for_answers(data_name_or_path, output_path, llm, model_name_or_path, num_gpu=8, local_rank=0):
    dataset_config = {
        "number_of_docs": 1,
        "data_name_or_path": data_name_or_path,
        "corpus": '../data_of_ReGPT/Wiki-corpus/train',
        "max_seq_len": 512,
    }
    tokenizer = initialize_tokenizer(model_name_or_path)
    datasets = QADataset4ChatWithBackgroundKnowledge(tokenizer, dataset_config)
    # batch_size = 512
    batch_size = 128
    sampling_params = SamplingParams(temperature=0.9, n=1, max_tokens=256)
    new_data = process_batches(datasets, llm, batch_size, sampling_params, "background_knowledge", "<|start_header_id|>user<|end_header_id|>\n\nPlease provide the background knowledge for the answer. The background knowledge MUST be within 256 tokens.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", num_gpu, local_rank)
    
    save_dataset(new_data, output_path)

def generate_background_knowledge_for_answers(num_gpu=8, local_rank=0):
    model_name_or_path = "../llama3-chat"
    llm = initialize_llm(model_name_or_path)
    candidates = ['2WikiMultihopQA', 'hotpotqa/']
    for candidate in candidates:
        print(f"Processing {candidate}...")
        data_path = f"../data_of_ReGPT/QA_datasets_contrastive/{candidate}/sorted_datasets_train"
        output_path = data_path.replace("QA_datasets_contrastive", f"QA_datasets_contrastive_with_background_knowledge_{local_rank}GPU")
        _generate_background_knowledge_for_answers(data_path, output_path, llm, model_name_or_path, num_gpu, local_rank)


# 函数：合并单个数据集的多个GPU数据
def merge_single_dataset(base_dir, dataset_name, gpu_ids):
    datasets = []
    for gpu_id in gpu_ids:
        path = f'{base_dir}{gpu_id}/{dataset_name}/sorted_datasets_train/'
        dataset = load_from_disk(path)
        datasets.append(dataset)
    # 合并所有数据集，并扁平化索引
    merged_dataset = concatenate_datasets(datasets).flatten_indices()
    # 保存合并后的数据集
    merged_dataset.save_to_disk(f'{base_dir[:-1]}/{dataset_name}/sorted_datasets_train/')
    return merged_dataset

# 函数：合并多个数据集
def merge_datasets(base_dir, datasets_names, gpu_ids):
    merged_datasets = {}
    for dataset_name in datasets_names:
        print(f"正在合并 {dataset_name} 数据集...")
        merged_datasets[dataset_name] = merge_single_dataset(base_dir, dataset_name, gpu_ids)
    return merged_datasets

if __name__ == "__main__":
    import sys
    num_gpu = int(sys.argv[1])
    local_rank = int(sys.argv[2])
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
    generate_contrastive_answers(num_gpu, local_rank)
    # generate_background_knowledge_for_answers(num_gpu, local_rank)

    # # 示例调用
    # base_dir = '../data_of_ReGPT/QA_datasets_contrastive_with_background_knowledge_'  # 定义基础路径
    # gpu_ids = ['0GPU', '1GPU', '2GPU', '3GPU']  # 定义GPU数目
    # datasets_names = ['nq', '2WikiMultihopQA', 'hotpotqa']  # 定义要合并的数据集名称

    # # 调用函数合并多个数据集
    # merged_datasets = merge_datasets(base_dir, datasets_names, gpu_ids)

    # # 输出查看合并后的结果
    # for dataset_name, dataset in merged_datasets.items():
    #     print(f"Merged {dataset_name} dataset:", dataset)
