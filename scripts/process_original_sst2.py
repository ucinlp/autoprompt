import csv
from pathlib import Path


ROOT = Path('glue_data/SST-2')

sentiment_labels = []
with open(ROOT / 'original' / 'sentiment_labels.txt', 'r') as f:
    reader = csv.DictReader(f, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        sentiment_labels.append(int(float(row['sentiment values']) > 0.5))


sent_to_label = {}
with open(ROOT / 'original' / 'dictionary.txt', 'r') as f:
    reader = csv.reader(f, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        sent, label_id = row
        sent_to_label[sent.lower().replace('\\', '')] = sentiment_labels[int(label_id)]


with open(ROOT / 'test.tsv', 'r') as f, \
     open(ROOT / 'test-labeled.tsv', 'w') as g:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    writer = csv.DictWriter(g, delimiter='\t', quoting=csv.QUOTE_NONE, fieldnames=['sentence', 'label'])
    writer.writeheader()
    for row in reader:
        sent = row['sentence']
        label = sent_to_label[sent]
        writer.writerow({'sentence': sent, 'label': label})
