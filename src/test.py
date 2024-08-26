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


from transformers import AutoConfig, LlamaWithRetrievalHeadAndKnowledgeInjectorForCausalLM, AutoTokenizer
model_path = './llama3-chat'
config = AutoConfig.from_pretrained(model_path)
config.kg_model_name_or_path = './ReGPT/output/SFT-best/'
config.cross_attention_activation_function = 'silu'
config.add_cross_attention = True
config.add_cross_attention_layer_number = 31
config.faiss_dimension = 4096

model = LlamaWithRetrievalHeadAndKnowledgeInjectorForCausalLM.from_pretrained(model_path, config=config)
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_text = 'Who is the president of the United States?'
knowledge = 'The president of the United States is Putin.'
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
knowledge_input_ids = tokenizer(knowledge, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids, knowledge_input_ids=knowledge_input_ids, max_new_tokens=10, num_beams=1, do_sample=False)
print(tokenizer.decode(outputs[0]))
