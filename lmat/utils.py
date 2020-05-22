import csv
import json
import logging

from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


class TriggerCollator:
    """
    An object for collating outputs of TriggerTemplatizer
    """
    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels = list(zip(*features))
        # Assume that all inputs have the same keys as the first
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key].squeeze(0) for x in model_inputs]
            padded = pad_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        return padded_inputs, labels


class TriggerTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """
    def __init__(self,
                 template,
                 tokenizer,
                 add_special_tokens=True):
        if not hasattr(tokenizer, 'predict_token') or \
           not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )
        self._template = template
        self._tokenizer = tokenizer
        self._add_special_tokens = add_special_tokens

    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    def __call__(self, format_kwargs):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        label = format_kwargs.pop('label')
        text = self._template.format(**format_kwargs)
        logger.debug(f'Formatted text: {text}')
        if label is None:
            raise Exception(f'Bad data: {text}')

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer.encode_plus(
            text,
            add_special_tokens=self._add_special_tokens,
            return_tensors="pt"
        )
        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        return model_inputs, label


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')


def templatize_tsv(fname, templatizer):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        instances = [templatizer(row) for row in reader]
    return instances


def templatize_jsonl(fname, templatizer):
    with open(fname, 'r') as f:
        instances = [templatizer(json.loads(line)) for line in f]
    return instances


def load_dataset(path, templatizer):
    if path.suffix == '.tsv':
        return templatize_tsv(path, templatizer)
    elif path.suffix == '.jsonl':
        return templatize_jsonl(path, templatizer)
    else:
        raise ValueError(f'File "{path}" not supported. Currently supported formats: .tsv, .jsonl')











# def load_GLUE_data(args, filename, is_train, glue_name, sentence_size, class_labels, masked_words, down_sample = False):
#     facts = []
#     # tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
#     processor = processors[glue_name.lower()]()
#     print("hahahah", class_labels)
#     print("beeee", masked_words)
#     #TOOD: make this filepath as input
#     #/home/yrazeghi/data
#     if is_train:
#         data = processor.get_train_examples(args+glue_name)
#     else:
#         data = processor.get_dev_examples(args+glue_name)
#     for d in data:
#         label = d.label
#         if label in class_labels: #todo change this
#             ind = class_labels.index(label)
#             premiss = d.text_a
#             premiss = premiss[:-1]
#             hypothesis = d.text_b
#             hypothesis = hypothesis[:-1]
#             sub = premiss + " *%* " + hypothesis
#             # sub = "pick a context sentence that has obj_surface equal equal equal equal "
#             # print("label:::", label)
#             # print("word::::", )
#             obj = masked_words[ind]
#             # print("word::::", obj)

#             # TODO: @rlogan - add back in using correct tokenizer...
#             # if len(tokenizer.tokenize(sub)) > sentence_size:
#             #     continue

#             if down_sample:
#                 r_rand = random.uniform(0, 1)
#                 if r_rand < 0.005:
#                     facts.append((sub, obj))
#             else:
#                 facts.append((sub, obj))
#         # print('Total facts before:', len(lines))
#         # print('Invalid facts:', num_invalid_facts)
#     print('Total facts after:', len(facts))
#     return facts


# def load_TREx_data(args, filename):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

#     facts = []
#     with open(filename, newline='') as f:
#         lines = f.readlines()
#         num_invalid_facts = 0
#         for line in lines:
#             sample = json.loads(line)
#             sub = sample['sub_label']
#             obj = sample['obj_label']
#             sub =  sub
#             obj = obj
#             print("sub: ", sub)
#             print("obj: ", obj)
#             """
#             evidences = sample['evidences']
#             # To make the number of samples used between open-book and closed-book probe
#             # settings, we need to only consider facts that include context sentences
#             valid_contexts = []
#             # For each evidence, replace [MASK] in the masked sentence with obj_surface. But the actual answer/object should be obj_label
#             for evidence in evidences:
#                 ctx = evidence['masked_sentence']
#                 obj_surface = evidence['obj_surface']
#                 # Only consider context samples where object surface == true object label, and grab the first one
#                 if obj_surface == obj:
#                     valid_contexts.append(ctx)
#             # Randomly pick a context sentence that has obj_surface equal to the obj_label
#             if not valid_contexts:
#                 # print('Invalid fact with no context - sub: {}, obj: {}'.format(sub, obj))
#                 num_invalid_facts += 1
#             else:
#                 context = random.choice(valid_contexts)
#                 context_words = context.split()
#                 if len(context_words) > constants.MAX_CONTEXT_LEN:
#                     # If context is too long, use the first X tokens (it's ok if obj isn't included)
#                     context = ' '.join(context_words[:constants.MAX_CONTEXT_LEN])
#                     # print('Sample context too long ({}), truncating.'.format(len(context_words)))
#                 context = context.replace(constants.MASK, obj_surface)
#                 facts.append((sub, obj, context))
#             """
#             # Skip facts with objects that are not single token
#             if len(tokenizer.tokenize(obj)) > 1:
#                 num_invalid_facts += 1
#                 continue

#             facts.append((sub, obj))
#         print('Total facts before:', len(lines))
#         print('Invalid facts:', num_invalid_facts)
#         print('Total facts after:', len(facts))
#     return facts


# def get_all_datasets(args):
#     datasets = []

#     train_file = os.path.join(args.data_dir, 'train.jsonl')
#     # train_data = load_TREx_data(args, train_file)
#     class_labels = args.class_labels.split('-')
#     masked_words = args.masked_words.split('-')
#     train_data = load_GLUE_data(args.data_dir, train_file , is_train=True, glue_name = args.dataset , down_sample = False, class_labels = class_labels , masked_words = masked_words, sentence_size = args.sentence_size)
#     print('Num samples in train data:', len(train_data))

#     # dev_file = os.path.join(args.data_dir, 'val.jsonl')
#     dev_file = os.path.join(args.data_dir, 'dev.jsonl')
#     # dev_data = load_TREx_data(args, dev_file)
#     dev_data = load_GLUE_data(args.data_dir, dev_file , is_train=False, glue_name = args.dataset , down_sample = False, class_labels = class_labels, masked_words = masked_words, sentence_size = args.sentence_size)
#     print('Num samples in dev data:', len(dev_data))

#     datasets.append((train_data, dev_data))

#     return datasets


# def iterate_batches(inputs, batch_size, shuffle=False):
#     """
#     Split data into batches and return them as a generator
#     """
#     size = len(inputs)
#     inputs = np.array(inputs)
#     if shuffle:
#         indices = np.arange(size)
#         np.random.shuffle(indices)
#     for start_idx in range(0, size, batch_size):
#         end_idx = min(start_idx + batch_size, size)
#         if shuffle:
#             excerpt = indices[start_idx:end_idx]
#         else:
#             excerpt = slice(start_idx, end_idx)
#         yield inputs[excerpt]


# def make_batch(tokenizer, batch, trigger_tokens, prompt_format, use_ctx, cls_token, sep_token, mask_token, pad_token, period_token, device): #this should be changed for Roberta
#     """
#     For BERT, [CLS] token marks the beginning of a sentence and [SEP] marks separation/end of sentences
#     """
#     source_tokens_batch = []
#     target_tokens_batch = []
#     trigger_mask_batch = []
#     segment_ids_batch = []

#     for sample in batch:
#         # print('PROMPT:', build_prompt(tokenizer, sample, trigger_tokens))
#         source_tokens = []
#         target_tokens = []
#         trigger_mask = []
#         segment_ids = [] # used to distinguish different sentences
#         # sub, obj, ctx = sample
#         sub, obj = sample
#         sub_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sub))
#         print(len(tokenizer.tokenize(sub)))
#         obj_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
#         trigger_idx = 0
#         # print('SUB TOKENIZED:', tokenizer.tokenize(sub))
#         # print('OBJ TOKENIZED:', tokenizer.tokenize(obj))

#         # Add CLS token at the beginning
#         source_tokens.extend(cls_token)
#         target_tokens.append(-1)
#         trigger_mask.append(0)
#         # Add context if probe setting is open-book (use context)
#         if use_ctx:
#             # From CLS token right before
#             segment_ids.append(0)
#             # Add context tokens
#             source_tokens.extend(context_tokens)
#             target_tokens.extend([-1] * len(context_tokens))
#             trigger_mask.extend([0] * len(context_tokens))
#             segment_ids.extend([0] * len(context_tokens))
#             # Add SEP token to distinguish sentences
#             source_tokens.extend(sep_token)
#             target_tokens.append(-1)
#             trigger_mask.append(0)
#             segment_ids.append(0)

#         for part in prompt_format:
#             if part == 'X':
#                 # Add subject
#                 source_tokens.extend(sub_tokens)
#                 target_tokens.extend([-1] * len(sub_tokens))
#                 trigger_mask.extend([0] * len(sub_tokens))
#             elif part == 'Y':
#                 # Add MASKED object
#                 source_tokens.extend(mask_token)
#                 target_tokens.extend(obj_tokens)
#                 trigger_mask.extend([0] * len(obj_tokens))
#             else:
#                 # Add triggers
#                 num_trigger_tokens = int(part)
#                 source_tokens.extend(trigger_tokens[trigger_idx:trigger_idx+num_trigger_tokens])
#                 target_tokens.extend([-1] * (num_trigger_tokens))
#                 trigger_mask.extend([1] * (num_trigger_tokens))
#                 # Update trigger idx
#                 trigger_idx += num_trigger_tokens

#         # Add period at end of prompt
#         source_tokens.extend(period_token)
#         target_tokens.append(-1)
#         trigger_mask.append(0)

#         # Add SEP token at the end
#         source_tokens.extend(sep_token)
#         target_tokens.append(-1)
#         trigger_mask.append(0)

#         if use_ctx:
#             segment_ids.extend([1] * len(source_tokens))
#         else:
#             segment_ids.extend([0] * len(source_tokens))

#         # Add encoded prompt to batch
#         source_tokens_batch.append(torch.tensor(source_tokens))
#         target_tokens_batch.append(torch.tensor(target_tokens))
#         trigger_mask_batch.append(torch.tensor(trigger_mask))
#         segment_ids_batch.append(torch.tensor(segment_ids))

#     # Get max length sequence for padding
#     seq_len = [s.size(0) for s in source_tokens_batch]
#     max_len = np.max(seq_len)

#     # Pad the batch
#     source_tokens_batch = torch.nn.utils.rnn.pad_sequence(source_tokens_batch, batch_first=True, padding_value=pad_token[0])
#     target_tokens_batch = torch.nn.utils.rnn.pad_sequence(target_tokens_batch, batch_first=True, padding_value=-1)
#     trigger_mask_batch = torch.nn.utils.rnn.pad_sequence(trigger_mask_batch, batch_first=True)
#     segment_ids_batch = torch.nn.utils.rnn.pad_sequence(segment_ids_batch, batch_first=True, padding_value=pad_token[0])

#     # Move to GPU
#     source_tokens_batch = source_tokens_batch.to(device)
#     target_tokens_batch = target_tokens_batch.to(device)
#     trigger_mask_batch = trigger_mask_batch.to(device)
#     segment_ids_batch = segment_ids_batch.to(device)

#     return source_tokens_batch, target_tokens_batch, trigger_mask_batch, segment_ids_batch


# def make_batch_glue(tokenizer, batch, trigger_tokens, prompt_format, use_ctx, cls_token, sep_token, mask_token, pad_token, period_token, device): #this should be changed for Roberta
#     """
#     For BERT, [CLS] token marks the beginning of a sentence and [SEP] marks separation/end of sentences
#     """
#     source_tokens_batch = []
#     target_tokens_batch = []
#     trigger_mask_batch = []
#     segment_ids_batch = []
#     labels = []

#     for sample in batch:
#         # print('PROMPT:', build_prompt(tokenizer, sample, trigger_tokens))
#         source_tokens = []
#         target_tokens = []
#         trigger_mask = []
#         segment_ids = [] # used to distinguish different sentences

#         # sub, obj, ctx = sample
#         sub, obj = sample
#         prem, hyp = sub.split("*%*")
#         prem_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prem))
#         hyp_tokens =  tokenizer.convert_tokens_to_ids(tokenizer.tokenize(hyp))

#         # sub_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sub))
#         obj_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
#         labels.append(obj_tokens)
#         trigger_idx = 0

#         # Add CLS token at the beginning
#         source_tokens.extend(cls_token)
#         target_tokens.append(-1)
#         trigger_mask.append(0)
#         # From CLS token right before
#         segment_ids.append(0)
#         # Add context if probe setting is open-book (use context)
#         SEN_FLAG = 1

#         for part in prompt_format:
#             if part == 'H':
#                 # Add Hypothesis
#                 source_tokens.extend(hyp_tokens)
#                 target_tokens.extend([-1] * len(hyp_tokens))
#                 trigger_mask.extend([0] * len(hyp_tokens))
#                 segment_ids.extend([1-SEN_FLAG] * len(hyp_tokens))
#             elif part == 'Y':
#                 # Add MASKED object
#                 source_tokens.extend(mask_token)
#                 target_tokens.extend(obj_tokens)
#                 trigger_mask.extend([0] * len(obj_tokens))
#                 segment_ids.extend([1-SEN_FLAG] * len(obj_tokens))
#             elif part =='P':
#                 source_tokens.extend(prem_tokens)
#                 target_tokens.extend([-1] * len(prem_tokens))
#                 trigger_mask.extend([0] * len(prem_tokens))
#                 segment_ids.extend([1-SEN_FLAG] * len(prem_tokens))
#             elif part == 'S':
#                 # Add SEP token to distinguish sentences
#                 source_tokens.extend(sep_token)
#                 target_tokens.append(-1)
#                 trigger_mask.append(0)
#                 segment_ids.append(1-SEN_FLAG)
#                 SEN_FLAG = 1-SEN_FLAG
#             else:
#                 # Add triggers
#                 num_trigger_tokens = int(part)
#                 source_tokens.extend(trigger_tokens[trigger_idx:trigger_idx+num_trigger_tokens])
#                 target_tokens.extend([-1] * (num_trigger_tokens))
#                 trigger_mask.extend([1] * (num_trigger_tokens))
#                 # Update trigger idx
#                 trigger_idx += num_trigger_tokens
#                 segment_ids.extend([1-SEN_FLAG] * num_trigger_tokens)


#         # Add period at end of prompt
#         source_tokens.extend(period_token)
#         target_tokens.append(-1)
#         trigger_mask.append(0)
#         segment_ids.append(1-SEN_FLAG)

#         # Add SEP token at the end
#         source_tokens.extend(sep_token)
#         target_tokens.append(-1)
#         trigger_mask.append(0)
#         segment_ids.append(1-SEN_FLAG)


#         # Add encoded prompt to batch
#         source_tokens_batch.append(torch.tensor(source_tokens))
#         target_tokens_batch.append(torch.tensor(target_tokens))
#         trigger_mask_batch.append(torch.tensor(trigger_mask))
#         segment_ids_batch.append(torch.tensor(segment_ids))

#     # Get max length sequence for padding
#     seq_len = [s.size(0) for s in source_tokens_batch]
#     max_len = np.max(seq_len)

#     # Pad the batch
#     source_tokens_batch = torch.nn.utils.rnn.pad_sequence(source_tokens_batch, batch_first=True, padding_value=pad_token[0])
#     target_tokens_batch = torch.nn.utils.rnn.pad_sequence(target_tokens_batch, batch_first=True, padding_value=-1)
#     trigger_mask_batch = torch.nn.utils.rnn.pad_sequence(trigger_mask_batch, batch_first=True)
#     segment_ids_batch = torch.nn.utils.rnn.pad_sequence(segment_ids_batch, batch_first=True, padding_value=pad_token[0])

#     # Move to GPU
#     source_tokens_batch = source_tokens_batch.to(device)
#     target_tokens_batch = target_tokens_batch.to(device)
#     trigger_mask_batch = trigger_mask_batch.to(device)
#     segment_ids_batch = segment_ids_batch.to(device)

#     return source_tokens_batch, target_tokens_batch, trigger_mask_batch, segment_ids_batch, labels

# def get_unique_objects(data):
#     """
#     @rloganiv - Used to identify the set of labels
#     """
#     objs = set()
#     for sample in data:
#         sub, obj = sample
#         # sub, obj, ctx = sample
#         # print('sub: {}, obj: {}, ctx: {}'.format(sub, obj, ctx))
#         objs.add(obj)
#     return list(objs)


# def load_vocab(vocab_filename):
#     with open(vocab_filename, "r") as f:
#         lines = f.readlines()
#     vocab = [x.strip() for x in lines]
#     return vocab
