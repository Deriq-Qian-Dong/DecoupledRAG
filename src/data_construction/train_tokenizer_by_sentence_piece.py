import sys
import time
import sentencepiece as spm
corpus_name=sys.argv[1]
start_time = time.time()
spm.SentencePieceTrainer.train(
    input=f'../{corpus_name}/corpus.tsv',  # 输入文件
    model_prefix=f'llama-phrase-{corpus_name}',  # 模型前缀
    shuffle_input_sentence=True,  # 是否打乱句子
    train_extremely_large_corpus=True,
    # hyperparameters of tokenizer
    max_sentence_length=16384,  # 句子最大长度
    max_sentencepiece_length=128,
    model_type="BPE",
    vocab_size=1000000,
    split_by_whitespace=False,
    split_by_number=True,
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    allow_whitespace_only_pieces=True,
    remove_extra_whitespaces=False,
    normalization_rule_name="nfkc",
    num_threads=128,
)

end_time = time.time()
print(end_time - start_time)