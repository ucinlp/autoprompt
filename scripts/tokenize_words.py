import argparse
import transformers


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    print(tokenizer.tokenize(args.input))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-t', '--tokenizer', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    main(args)

