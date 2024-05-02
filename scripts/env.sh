unset https_proxy
pip install -r requirements.txt
git config --global user.email "dongqian"
git config --global user.name "dongqian"
export https_proxy=http://172.19.57.45:3128
mkdir -p ../data_of_ReGPT
cp -r ../data/data_of_ReGPT/marco/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/WikiText-103/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/phrases_WikiText-103/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/llama2-7b-phrase-tokenizer-trained-on-WikiText103/ ../data_of_ReGPT/
# cp -r ../data/data_of_ReGPT/En-Wiki/ ../data_of_ReGPT/
# cp -r ../data/llama2-7b/ ../
# cp -r ../data/data_of_ReGPT/c4_en/ ../data_of_ReGPT/
