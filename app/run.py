import csv
from dataclasses import dataclass
import io
import logging
import random
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st
import torch
import transformers
from tqdm import tqdm

from autoprompt import utils
import autoprompt.create_trigger as ct


# logging.getLogger("streamlit.caching").addHandler(logging.StreamHandler(sys.stdout))
# logging.getLogger("streamlit.caching").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


# Setting eq and frozen ensures that a __hash__ method is generated which is needed for caching to
# properly respond to changed args.
@dataclass(eq=True, frozen=True)
class Args:
    # Configurable
    template: str
    model_name: str
    iters: int
    num_cand: int
    accumulation_steps: int

    # Non-Configurable
    seed = 0
    sentence_size = 64
    tokenize_labels = True
    filter = False
    initial_trigger = None
    label_field = "label"
    bsz = 32
    eval_size = 1

    @classmethod
    def from_streamlit(cls):
        st.sidebar.image('assets/icon.png', width=150)
        st.sidebar.markdown('### Training Parameters')
        model_name = st.sidebar.selectbox(
            "Model",
            options=['bert-base-cased', 'roberta-large'],
            help="Language model used for training and evaluation."
        )
        iters = int(st.sidebar.number_input(
            "Iterations",
            value=3,
            min_value=1,
            max_value=100,
            help="Number of trigger search iterations. Larger values may yield better results."
        ))
        num_cand = int(st.sidebar.number_input(
            "Number of Candidates",
            value=10,
            min_value=1,
            max_value=100,
            help="Number of candidate trigger token replacements to evaluate during each search "
                 "iteration. Larger values may yield better results."
        ))
        accumulation_steps = int(st.sidebar.number_input(
            "Gradient Accumulation Steps",
            value=1,
            min_value=1,
            max_value=10,
            help="Number of gradient accumulation steps used during training. Larger values may yield "
                 "better results. Cannot be larger than half the dataset size."
        ))
        st.sidebar.markdown(
            """
            ### Template

            Templates define how task-specific inputs are combined with trigger tokens to create
            the prompt. They should contain the following placeholders:
            - `{sentence}`: Placeholders for the task-specific input fields contain the field name
              between curly brackets. For manually entered data the field name is `{sentence}`. For
              uploaded csv's, field names should correspond to columns in the csv.
            - `[T]`: Placeholder for a trigger token. These are learned from the training data.
            - `[P]`: Placeholder for where to insert the [MASK] token that the model will predict
              on.

            Templates can also include manually written text (e.g., `[CLS]` and `[SEP]` in the
            default input).
            """
        )
        template = st.sidebar.text_input("Template", "[CLS] {sentence} [T] [T] [T] [P] . [SEP]")
        return cls(
            template=template,
            model_name=model_name,
            iters=iters,
            num_cand=num_cand,
            accumulation_steps=accumulation_steps,
        )


# TODO(rloganiv): This probably could use a better name...
@dataclass
class GlobalData:
    device: torch.device
    config: transformers.PretrainedConfig
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    embeddings: torch.nn.Module
    embedding_gradient: ct.GradientStorage
    predictor: ct.PredictWrapper

    @classmethod
    @st.cache(allow_output_mutation=True)
    def from_pretrained(cls, model_name):
        logger.info(f'Loading pretrained model: {model_name}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config, model, tokenizer = ct.load_pretrained(model_name)
        model.to(device)
        embeddings = ct.get_embeddings(model, config)
        embedding_gradient = ct.GradientStorage(embeddings)
        predictor = ct.PredictWrapper(model)
        return cls(
            device,
            config,
            model,
            tokenizer,
            embeddings,
            embedding_gradient,
            predictor
        )


@dataclass
class Dataset:
    train: List[int]
    dev: List[int]
    label_map: Dict[str, str]


def load_trigger_dataset(dataset, templatizer):
    instances = []
    for x in dataset:
        instances.append(templatizer(x))
    return instances


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def run_autoprompt(args, dataset):
    ct.set_seed(args.seed)
    global_data = GlobalData.from_pretrained(args.model_name)

    templatizer = utils.TriggerTemplatizer(
        args.template,
        global_data.config,
        global_data.tokenizer,
        label_field=args.label_field,
        label_map=dataset.label_map,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
    )
    evaluation_fn = ct.AccuracyFn(global_data.tokenizer, dataset.label_map, global_data.device)

    # Do not allow for initial trigger specification.
    trigger_ids = [global_data.tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=global_data.device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # Load datasets
    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=global_data.tokenizer.pad_token_id)
    train_dataset = load_trigger_dataset(dataset.train, templatizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = load_trigger_dataset(dataset.dev, templatizer)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    progress = st.progress(0.0)
    trigger_placeholder = st.empty()
    best_dev_metric = -float('inf')
    for i in range(args.iters):
        logger.info(f'Iteration: {i}')
        progress.progress(float(i)/args.iters)
        
        current_trigger = ','.join(global_data.tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0)))
        trigger_placeholder.markdown(f'**Current trigger**: {current_trigger}')

        global_data.model.zero_grad()
        train_iter = iter(train_loader)
        averaged_grad = None

        # Compute gradient of loss
        for step in range(args.accumulation_steps):
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(global_data.device) for k, v in model_inputs.items()}
            labels = labels.to(global_data.device)
            predict_logits = global_data.predictor(model_inputs, trigger_ids)
            loss = ct.get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = global_data.embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps

        logger.info('Evaluating Candidates')
        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)

        token_to_flip = i % templatizer.num_trigger_tokens
        candidates = ct.hotflip_attack(averaged_grad[token_to_flip],
                                       global_data.embeddings.weight,
                                       increase_loss=False,
                                       num_candidates=args.num_cand)
        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=global_data.device)
        denom = 0
        for step in pbar:
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(global_data.device) for k, v in model_inputs.items()}
            labels = labels.to(global_data.device)
            with torch.no_grad():
                predict_logits = global_data.predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels)

            # Update current score
            current_score += eval_metric.sum()
            denom += labels.size(0)

            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for i, candidate in enumerate(candidates):

                # if candidate.item() in filter_candidates:
                #     candidate_scores[i] = -1e32
                #     continue

                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = global_data.predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)

                candidate_scores[i] += eval_metric.sum()

        if (candidate_scores > current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(global_data.device) for k, v in model_inputs.items()}
            labels = labels.to(global_data.device)
            with torch.no_grad():
                predict_logits = global_data.predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric


    progress.progress(1.0)
    current_trigger = ','.join(global_data.tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0)))
    trigger_placeholder.markdown(f'**Current trigger**: {current_trigger}')

    best_trigger_tokens = global_data.tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    dev_output = predict_test(map(lambda x: x['sentence'], dataset.dev), dataset.label_map,
                              templatizer, best_trigger_ids, global_data.tokenizer, global_data.predictor, args)
    st.dataframe(pd.DataFrame(dev_output).style.highlight_min(axis=1))
    return best_trigger_tokens, best_dev_metric, dataset.label_map, templatizer, best_trigger_ids, global_data.tokenizer, global_data.predictor, args


def predict_test(sentences, label_map, templatizer, best_trigger_ids, tokenizer, predictor, args):
    # Evaluate clean
    output = { 'sentences': [] }
    any_label = None
    for label in label_map.values():
        output[label] = []
        any_label = label
    output['prompt'] = []
    for sentence in sentences:
        model_inputs, _ = templatizer({'sentence': sentence, 'label': any_label})
        model_inputs = {k: v.to(best_trigger_ids.device) for k, v in model_inputs.items()}

        prompt_ids = ct.replace_trigger_tokens(
            model_inputs, best_trigger_ids, model_inputs['trigger_mask'])
        # st.write(prompt_ids)
        # st.write(prompt_ids.shape)

        prompt = ' '.join(tokenizer.convert_ids_to_tokens(prompt_ids['input_ids'][0]))
        output['prompt'].append(prompt)

        predict_logits = predictor(model_inputs, best_trigger_ids)
        output['sentences'].append(sentence)
        for label in label_map.values():
            label_id = utils.encode_label(tokenizer=tokenizer, label=label, tokenize=args.tokenize_labels)
            label_id = label_id.to(best_trigger_ids.device)
            label_loss = ct.get_loss(predict_logits, label_id)
            # st.write(sentence, label, label_loss)
            output[label].append(label_loss.item())
    return output


def manual_dataset():
    num_train_instances = st.slider("Number of Train Instances", 4, 50)
    any_empty = False
    dataset = []
    data_col, label_col = st.beta_columns([3,1])
    for i in range(num_train_instances):
        with data_col:
            data = st.text_input("Train Instance " + str(i+1))
        with label_col:
            label = st.text_input("Train Label " + str(i+1), max_chars=20)
        if data == "" or label == "":
            any_empty = True
        dataset.append({'sentence': data, 'label': label})

    num_eval_instances = st.slider("Number of Evaluation Instances", 2, 50)
    eval_dataset = []
    data_col, label_col = st.beta_columns([3,1])
    for i in range(num_eval_instances):
        with data_col:
            data = st.text_input("Eval Instance " + str(i+1))
        with label_col:
            label = st.text_input("Eval Label " + str(i+1), max_chars=20)
        if data == "" or label == "":
            any_empty = True
        eval_dataset.append({'sentence': data, 'label': label})

    label_set = set(map(lambda x: x['label'], dataset))
    label_map = dict(map(lambda x: (x, x), label_set))

    if any_empty:
        st.warning('Waiting for data to be added')
        st.stop()

    if len(label_set) < 2:
        st.warning('Not enough labels')
        st.stop()

    return Dataset(
        train=dataset,
        dev=eval_dataset,
        label_map=label_map
    )


def csv_dataset():
    st.markdown("""
        Please upload your training and evaluation csv files.

        Format restrictions:
        - The file is required to have a header
        - The column name of the output field should be `label`.
        - Each file should contain no more than 64 rows.
    """)
    train_csv = st.file_uploader('Train', accept_multiple_files=False)

    dev_csv = st.file_uploader('Dev', accept_multiple_files=False)

    if train_csv is None or dev_csv is None:
        st.stop()

    with io.StringIO(train_csv.getvalue().decode('utf-8')) as f:
        reader = csv.DictReader(f)
        train_dataset = list(reader)
    if len(train_dataset) > 64:
        raise ValueError('Train dataset is too large')

    with io.StringIO(dev_csv.getvalue().decode('utf-8')) as f:
        reader = csv.DictReader(f)
        dev_dataset = list(reader)
    if len(dev_dataset) > 64:
        raise ValueError('Dev dataset is too large')

    labels = set(x['label'] for x in train_dataset)
    label_map = {x: x for x in labels}

    return Dataset(
        train=train_dataset,
        dev=dev_dataset,
        label_map=label_map
    )


def run():
    st.title('AutoPrompt Demo')
    st.write("Give some examples, get a model!")
    st.markdown("See https://ucinlp.github.io/autoprompt/ for more details.")

    args = Args.from_streamlit()
    dataset_mode = st.radio('How would you like to input your training data?',
                            options=['Manual Input', 'From CSV'])

    if dataset_mode == 'Manual Input':
        dataset = manual_dataset()
    else:
        dataset = csv_dataset()

    # TODO(rloganiv): Way too many arguments...
    if st.button('Train'):
        trigger_tokens, dev_metric, label_map, templatizer, best_trigger_ids, tokenizer, predictor, args = run_autoprompt(args, dataset)
        logger.debug('Dev metric')
        st.write('Train accuracy: ' + str(round(dev_metric*100, 1)))
        st.write("### Let's test it ourselves!")
        sentence = st.text_input("Sentence", dataset.dev[1]['sentence'])
        pred_output = predict_test([sentence], label_map ,templatizer, best_trigger_ids, tokenizer, predictor, args)
        st.dataframe(pd.DataFrame(pred_output).style.highlight_min(axis=1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout)
    run()

