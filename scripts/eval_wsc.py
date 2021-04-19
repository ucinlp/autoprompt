import argparse
import json


def main(args):
    total_correct = 0
    total = 0
    with open(args.ground_truth, 'r') as f, \
         open(args.predictions, 'r') as g:
        for f_line, g_line in zip(f, g):
            f_data = json.loads(f_line)
            g_data = json.loads(g_line)
            label = f_data['label']
            prediction_tokens = set(x for x in g_data['prediction_tokens'] if x != '<pad>')
            label_tokens = set(x for x in g_data['label_tokens'] if x != '<pad>')
            if len(prediction_tokens) == 0:
                predicted_label = False
            else:
                predicted_label = prediction_tokens.issubset(label_tokens)
            correct = (label == predicted_label)
            total_correct += correct
            total += 1
    print(total_correct / total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth', type=str)
    parser.add_argument('predictions', type=str)
    args = parser.parse_args()

    main(args)

