import numpy as np
import faiss
import os
from flask import Flask, request, jsonify
import requests


class FaissServer:
    def __init__(self, index_path, cache_path, dimension=768, topk=6):
        self.topk = topk
        self.dimension = dimension
        self.index_path = index_path
        self.cache_path = cache_path

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.add_url_rule('/search', 'search', self.search, methods=['POST'])

        index_paths = [f'{cache_path}/gpu_{i}.index' for i in range(faiss.get_num_gpus())]
        self.gpu_resources = [faiss.StandardGpuResources() for _ in range(faiss.get_num_gpus())]
        if not os.path.exists(cache_path) or not all([os.path.exists(p) for p in index_paths]):
            # Initialize FAISS
            self.gpu_indices = self.create_gpu_indices()
            os.makedirs(self.cache_path, exist_ok=True)
            print('Index not found, initializing server...')
            # Load vectors and add to the indices
            self.load_vectors()
            print('Server initialized successfully!')
            # Save the index to disk
            self.save_index()
        else:
            print('Index found, loading server...')
            self.load_index()
            print('Server loaded successfully!')

    
    def load_index(self):
        self.gpu_indices = []
        self.kb = np.load(self.index_path).astype(np.float32)
        for i in range(faiss.get_num_gpus()):
            gpu_index = faiss.read_index(self.cache_path + f'/gpu_{i}.index')
            # Convert the index to GPU
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources[i], i, gpu_index)
            self.gpu_indices.append(gpu_index)
        self.chunk_size = self.gpu_indices[0].ntotal

    def save_index(self):
        for i, gpu_index in enumerate(self.gpu_indices):
            # Re-convert the index back to CPU
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), self.cache_path + f'/gpu_{i}.index')

    def create_gpu_indices(self):
        gpu_indices = []
        for i in range(faiss.get_num_gpus()):
            index_flat = faiss.IndexFlatL2(self.dimension)
            gpu_index_flat = faiss.index_cpu_to_gpu(self.gpu_resources[i], i, index_flat)
            gpu_indices.append(gpu_index_flat)
        return gpu_indices

    def load_vectors(self):
        kb = np.load(self.index_path).astype(np.float32)
        num_vectors = kb.shape[0]
        num_gpus = len(self.gpu_indices)
        chunk_size = num_vectors // num_gpus
        self.chunk_size = chunk_size
        self.kb = kb

        for i, gpu_index in enumerate(self.gpu_indices):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_gpus - 1 else num_vectors
            gpu_index.add(kb[start:end])
            print(f'Number of vectors in GPU {i} index: {gpu_index.ntotal}')

    def search(self):
        query_vector = np.array(request.json['vector']).reshape(1, -1).astype(np.float32)
        k = self.topk

        distances, indices = [], []
        for idx, gpu_index in enumerate(self.gpu_indices):
            D, I = gpu_index.search(query_vector, k)
            distances.append(D)
            I += idx * self.chunk_size
            indices.append(I)

        distances = np.hstack(distances)  # shape: (batch_size, num_gpus*k)
        indices = np.hstack(indices)  # shape: (batch_size, num_gpus*k)
        # Get the top k results across all GPUs
        indices_to_gather = np.argsort(distances, axis=1)[:, :k]
        topk_distances = np.take_along_axis(distances, indices_to_gather, axis=1)
        topk_indices = np.take_along_axis(indices, indices_to_gather, axis=1)
        topk_kb = self.kb[topk_indices]

        return jsonify({
            'distances': topk_distances.tolist(),
            'indices': topk_indices.tolist(),
            'neighbor_embeddings': topk_kb.tolist()
        })
    
    def search_by_vector(self, vector):
        response = requests.post('http://localhost:5000/search', json={'vector': vector})
        return response.json()

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    index_path = '../data_of_ReGPT/Wiki-corpus/phrases_embeddings.npy'
    cache_path = './faiss_cache'
    server = FaissServer(index_path, cache_path)
    server.run()
