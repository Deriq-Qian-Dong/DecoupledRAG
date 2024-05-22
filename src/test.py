from transformers import AutoConfig, LlamaWithRetrievalHeadForInference, AutoTokenizer
model_path = './output/SFT-best/'
config = AutoConfig.from_pretrained(model_path)
config.negatives_x_device = True

config.kb_path = '../data_of_ReGPT/marco/phrases_embeddings.npy'
config.retrieval_step = 10
config.topk = 6
config.q_reps_cache_type = 'QRepsCache'
config.q_reps_cache_window_size = 10

model = LlamaWithRetrievalHeadForInference.from_pretrained(model_path, config=config)          
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)


input_text = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

input_ids = input_ids[:,:30]
outputs = model.generate(input_ids,max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
from datasets import load_dataset, load_from_disk
dataset = load_from_disk('../data_of_ReGPT/marco/collection/')
while True:
    input_text = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.'
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    
    input_ids = input_ids[:,:30]
    outputs = model.generate(input_ids,max_new_tokens=100)