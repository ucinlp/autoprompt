import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import utils

def filter_data(args):
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter out invalid samples in TREx-train data for every relation')
    parser.add_argument('--data_dir', type=str, help='Directory containing TREx relation files')
    parser.add_argument('--out_dir', type=str, help='Directory to store output plots')
    parser.add_argument('--model_vocab_file', type=str, help="File containing a specific model's vocab")
    parser.add_argument('--common_vocab_file', type=str, help='File containing common vocab subset')
    args = parser.parse_args()
    filter_data(args)
