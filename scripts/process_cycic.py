import argparse
import json
import re


REGEX = re.compile(r'(?<=True or False: ).*(?=\.)')


def main(args):
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            match = REGEX.search(data['question'])
            if match:
                data['question'] = match.group(0)
                print(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    main(args)

