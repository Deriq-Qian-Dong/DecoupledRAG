# Decoupling Knowledge and Context: An Efficient and Effective Retrieval Augmented Generation Framework via Cross Attention

## Overview
Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) with external knowledge. However, traditional RAG methods face challenges: they create lengthy contexts leading to slow inference, risk degrading LLM capabilities, and suffer from knowledge permutation sensitivity.

We present DecoupledRAG, a novel framework that addresses these limitations by separating external knowledge from the context. Our approach uses cross-attention to inject knowledge directly during LLM inference, without modifying model parameters or input context. This innovation enables efficient and robust knowledge integration while maintaining model performance.

## Installation
To set up the environment, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
sh scripts/update_transformers.sh
```

## Training
To train the DecoupledRAG framework, run the following command:

```bash
sh scripts/rag_llama_run.sh
```

The evaluation results will be presented in tensorboard.

### Configuration
All configuration files can be found in the `config` directory. You can check and modify the corresponding yaml files according to your needs.

### Training Scripts
We provide several training scripts in the `scripts` directory:

- `llama_run.sh`: Train the base LLaMA model
- `rag_llama_run.sh`: Train the DecoupledRAG model with LLaMA

To start training, simply run the corresponding script. For example:

```bash
# Train DecoupledRAG with LLaMA
sh scripts/rag_llama_run.sh
```

### Monitoring Training Progress
You can monitor the training progress using TensorBoard. We provide a script to launch TensorBoard:

```bash
sh scripts/tensorboard.sh
```

This will start TensorBoard server and display various metrics including:
- Training and validation loss curves
- Evaluation metrics (F1 score, accuracy)

Access the TensorBoard interface through your web browser to visualize these metrics in real-time.
