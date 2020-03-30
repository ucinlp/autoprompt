import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate probes that combine context and prompt')
    parser.add_argument('in_file', type=str, help='JSONL file containing training data of subject object pairs')
    args = parser.parse_args()
