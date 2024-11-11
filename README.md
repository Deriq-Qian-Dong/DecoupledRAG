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
