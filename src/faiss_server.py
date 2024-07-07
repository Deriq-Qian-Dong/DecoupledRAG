import numpy as np
import faiss
from flask import Flask, request, jsonify

class FaissServer:
    def __init__(self, index_path, dimension=768):
        self.dimension = dimension
        self.index_path = index_path

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.add_url_rule('/search', 'search', self.search, methods=['POST'])

        # Initialize FAISS
        self.gpu_resources = [faiss.StandardGpuResources() for _ in range(faiss.get_num_gpus())]
        self.gpu_indices = self.create_gpu_indices()

        # Load vectors and add to the indices
        self.load_vectors()
        print('Server initialized successfully!')

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
        k = int(request.json.get('k', 5))

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

    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)

if __name__ == '__main__':
    index_path = '../data_of_ReGPT/Wiki-corpus/phrases_embeddings.npy'
    server = FaissServer(index_path)
    server.run()
