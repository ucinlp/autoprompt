import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
from transformers import glue_processors as processors
import json
import numpy as np



USE_GPU = 1
# Device configuration
device = torch.device('cuda' if (torch.cuda.is_available() and USE_GPU) else 'cpu')

# Load pre-trained model tokenizer (vocabulary)
pretrained_model = 'bert-base-uncased'
logging.basicConfig(level=logging.INFO)

# model_em = BertModel.from_pretrained('bert-base-uncased')
# model_em.eval()
# model_em.to(device)

tokenizer = BertTokenizer.from_pretrained(pretrained_model)

model = BertForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
# model = BertModel.from_pretrained(pretrained_model)
model.eval()

sentences_a_dev = []
sentences_b_dev = []
dev_labels = []


#read data mnli
processor = processors["mnli"]()
data = processor.get_dev_examples("/home/yrazeghi/data/MNLI")
data_dev = [d.to_dict() for d in data]
sentences_a = [d.get("text_a")  for d in data_dev]
sentences_b = [ d.get("text_b")  for d in data_dev]
dev_labels_all = [d.get("label")  for d in data_dev]


#labels: 	not_entailment, entailment
for i, b in enumerate(dev_labels_all):
    # if dev_labels_all[i] == "entailment":
    sentences_a_dev.append(sentences_a[i])
    sentences_b_dev.append(sentences_b[i])
    dev_labels.append(dev_labels_all[i])


#alphaNLI
# with open('../data/dev.jsonl', 'r') as handle:
#     json_data_dev = [json.loads(line) for line in handle]
# sentences_a_dev = [data["obs1"] +  data["hyp2"]  for data in json_data_dev]
# sentences_b_dev = [ data["obs2"]  for data in json_data_dev]
# sentences_a_dev_h2 = [data["obs1"] +  data["hyp2"]  for data in json_data_dev]
# sentences_b_dev_h2 = [ data["obs2"]  for data in json_data_dev]



# with open('../data/dev-labels.lst') as f:
#     dev_labels = [int(x)-1 for x in f.read().splitlines()]

predicted_words = []
embedings_all_layers = []
predicted_words50 = []
embedings = []
for i, b in enumerate(dev_labels):
    sub = sentences_a_dev[i]
    sub = sub[:-1]
    sub = sub + ", "
    sub = sub + sentences_b_dev[i]
    sub = sub[:-1]
    sub = sub + " rumors proved "
    text = '[CLS]' + sub + " target ." + '[SEP]'
    tokenized_text = tokenizer.tokenize(text)
    # print("tokenized_text:", tokenized_text)
    # print("tokenized_text len :", len(tokenized_text))
    maskIndex = 0
    for j, t in enumerate(tokenized_text):
        if t == "target":
            maskIndex = j
    tokenized_text[maskIndex] = '[MASK]'
    # print("tokenized_text after mask:", tokenized_text)
    # print("tokenized_text after mask len :", len(tokenized_text))
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print("indexed_tokens ", indexed_tokens)
    # print("indexed_tokens len: ", len(indexed_tokens))
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0]*len(tokenized_text)
    # segments_ids.extend([1]*(len(tokenized_text)-maskIndex)) #todo can use this


    # print("segments_ids ", segments_ids)
    # print("segments_ids len: ", len(segments_ids))



    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    model.to(device)

# Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]
        embedding = outputs[1]
    sorted_token_ids = torch.argsort(predictions[0, maskIndex])
    top_50_tokens = sorted_token_ids[:50]
    predicted_index = torch.argmax(predictions[0, maskIndex]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #put this word in the sentence and get embedding of that one and kmeans

    predicted_words.append(predicted_token)
    indexed_tokens[maskIndex] = predicted_index
    tokens_tensor = torch.tensor([indexed_tokens])
    # embedded_word = get_embedding4layers(tokens_tensor, segments_tensors, maskIndex)
    # embedings.append(embedded_word)


    for j, p in enumerate(top_50_tokens):
        predicted_token = tokenizer.convert_ids_to_tokens([top_50_tokens[j]])[0]
        # print(predicted_token)
        predicted_words50.append(predicted_token)
        if predicted_token == "True" or predicted_token == "true":
            print("here is True")
        if predicted_token == "False" or predicted_token == "false":
            print("here is True")
        # print(dev_labels[i])



from collections import Counter
counter = Counter(predicted_words50)
print(counter)
