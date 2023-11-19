from datasets import load_dataset

def filter_empty(example):
    return len(example['text']) > 0

def add_text_length(example):
    example["text_length"] = len(example["text"].split())
    return example

dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl", "test": "data/test.jsonl"}, field="data")

dataset = dataset.filter(filter_empty)

dataset = dataset.map(add_text_length)

dataset = dataset.sort("text_length", reverse=True)

dataset.save_to_disk('data/processed')



