import argparse
import json
from statistics import mean
import transformers


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    if args.input is not None:
        print(tokenizer.tokenize(args.input))

    lens = []
    if args.dataset is not None:
        with open(args.dataset, 'r') as f:
            for line in f:
                data = json.loads(line)
                tokenized = {k: tokenizer.tokenize(str(v)) for k,v in data.items()}
                print(tokenized)
                lens.append(sum(len(v) for v in tokenized.values()))
    print(f'Mean length: {mean(lens)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-t', '--tokenizer', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    main(args)

