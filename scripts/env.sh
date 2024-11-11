unset https_proxy
# pip install -r requirements.txt
cp ../data/packs_torch.tar ./
tar xvf packs_torch.tar
pip install --no-index --find-links=./packs_torch -r requirements.txt
git config --global user.email "dongqian"
git config --global user.name "dongqian"
export https_proxy='http://agent.baidu.com:8891'
export http_proxy='http://agent.baidu.com:8891'
mkdir -p ../data_of_ReGPT
# mkdir -p ../data_of_ReGPT/En-Wiki
# cp -r ../data/data_of_ReGPT/marco/sorted_datasets_train_llama2/ ../data_of_ReGPT/marco
# cp -r ../data/data_of_ReGPT/marco/sorted_datasets_test_llama2/ ../data_of_ReGPT/marco
# cp -r ../data/data_of_ReGPT/marco/phrases_embeddings.npy ../data_of_ReGPT/marco
# cp -r ../data/data_of_ReGPT/marco/collection/ ../data_of_ReGPT/marco
# cp -r ../data/data_of_ReGPT/msmarco_qa/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/WikiText-103/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/phrases_WikiText-103/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/llama2-7b-phrase-tokenizer-trained-on-WikiText103/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/En-Wiki/ ../data_of_ReGPT/
# cp -r ../data/llama2-7b/ ../
# cp -r ../data/data_of_ReGPT/c4_en/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/En-Wiki/sorted_datasets_test/ ../data_of_ReGPT/En-Wiki/
sh scripts/update_transformers.sh
# cp -r ../data/rag_llama2/24_qa/ ../
mkdir -p output
# mv ../24_qa/* output/
# cp -r ../data/rag_llama2/24_wiki/ ../
# mv ../24_wiki/* output/
mkdir -p ../data_of_ReGPT/Wiki-corpus
cp -r ../data/data_of_ReGPT/Wiki-corpus/train ../data_of_ReGPT/Wiki-corpus/
# cp -r ../data/data_of_ReGPT/Wiki-corpus/phrases_embeddings.npy ../data_of_ReGPT/Wiki-corpus/
cp -r ../data/data_of_ReGPT/QA_datasets_woEmb ../data_of_ReGPT/
cp -r ../data/data_of_ReGPT/QA_datasets_contrastive ../data_of_ReGPT/
# cp -r ../data/rag_llama2/24_wiki_qa/ ../
# cp -r ../data/llama3-chat ../
cp -r ../data/llama3_chat_qa_sft ../
# cp -r ../data/llama3-chat-top8 ../
cp ../data/anserini.tar.gz ../
cp ../data/jdk-11.0.13.zip ../
# cp -r ../data/data_of_ReGPT/Wiki-corpus/bm25_index/ ../data_of_ReGPT/Wiki-corpus/
cd ../
tar -xzvf anserini.tar.gz
unzip jdk-11.0.13.zip
echo 'export PATH=/root:$PATH' >> ~/.bashrc
echo 'PATH=$PATH:/root/paddlejob/workspace/env_run/anserini/target/appassembler/bin' >> ~/.bashrc
echo 'PATH=$PATH:/root/paddlejob/workspace/env_run/anserini/tools/eval/trec_eval.9.0.4/' >> ~/.bashrc
echo 'export JAVA_HOME=/root/paddlejob/workspace/env_run/jdk-11.0.13' >> ~/.bashrc
source ~/.bashrc
chmod -R +x /root/paddlejob/workspace/env_run/anserini
pip install pyserini

