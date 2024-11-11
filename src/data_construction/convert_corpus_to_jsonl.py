import json
import os
import argparse
from datasets import load_from_disk

def convert_collection(args):
    print('Converting collection...')
    file_index = 0
    corpus = load_from_disk(args.collection_path)
    for i, line in enumerate(corpus):
        doc_text = line['text']
        doc_id = i

        if i % args.max_docs_per_file == 0:
            if i > 0:
                output_jsonl_file.close()
            output_path = os.path.join(args.output_folder, 'docs{:02d}.json'.format(file_index))
            output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
            file_index += 1
        output_dict = {'id': doc_id, 'contents': doc_text}
        output_jsonl_file.write(json.dumps(output_dict) + '\n')

        if i % 100000 == 0:
            print(f'Converted {i:,} docs, writing into file {file_index}')

    output_jsonl_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert passage collection into jsonl files for Anserini.')
    parser.add_argument('--collection-path', required=True, help='Path to HF collection.')
    parser.add_argument('--output-folder', required=True, help='Output folder.')
    parser.add_argument('--max-docs-per-file', default=1000000, type=int,
                        help='Maximum number of documents in each jsonl file.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    convert_collection(args)
    print('Done!')