import csv
from dataclasses import dataclass
import io
import json
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


logger = logging.getLogger(__name__)


HIGHLIGHT_COLOR = '#eec66d'
with open('assets/sst2_train.jsonl', 'r') as f:
    DEFAULT_TRAIN = [json.loads(line) for line in f]


@dataclass
class CacheTest:
    """
    Stores whether the train button has been pressed for a given
    set of inputs to run_autoprompt.
    """
    is_test: bool


class CacheMiss(Exception):
    pass


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
            options=['roberta-large', 'bert-base-cased'],
            help="Language model used for training and evaluation."
        )
        iters = int(st.sidebar.number_input(
            "Iterations",
            value=10,
            min_value=1,
            max_value=100,
            help="Number of trigger search iterations. Larger values may yield better results."
        ))
        num_cand = int(st.sidebar.number_input(
            "Number of Candidates",
            value=25,
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

            Templates can also include manually written text (such as the
            period in the default example below).
            """
        )
        template = st.sidebar.text_input("Template", "{sentence} [T] [T] [T] [P].")
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
    label_map: Dict[str, str]


def load_trigger_dataset(dataset, templatizer):
    instances = []
    for x in dataset:
        instances.append(templatizer(x))
    return instances


@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs={CacheTest: lambda o: 0})
def run_autoprompt(args, dataset, cache_test):
    if cache_test.is_test:
        raise CacheMiss()

    ct.set_seed(args.seed)
    global_data = GlobalData.from_pretrained(args.model_name)

    templatizer = utils.TriggerTemplatizer(
        args.template,
        global_data.config,
        global_data.tokenizer,
        label_field=args.label_field,
        label_map=dataset.label_map,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=True,
    )
    evaluation_fn = ct.AccuracyFn(global_data.tokenizer, dataset.label_map, global_data.device,
                                  tokenize_labels=args.tokenize_labels)

    # Do not allow for initial trigger specification.
    trigger_ids = [global_data.tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=global_data.device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # Load datasets
    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=global_data.tokenizer.pad_token_id)
    try:
        train_dataset = load_trigger_dataset(dataset.train, templatizer)
    except KeyError as e:
        raise RuntimeError(
            'A field in your template is not present in the uploaded dataset. '
            f'Check that there is a column with the name: {e}'
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

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

        if (candidate_scores >= current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')

        # Skip eval
        best_trigger_ids = trigger_ids.clone()

    progress.progress(1.0)
    current_trigger = ','.join(global_data.tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0)))
    trigger_placeholder.markdown(f'**Current trigger**: {current_trigger}')

    best_trigger_tokens = global_data.tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))

    train_output = predict_test(map(lambda x: x['sentence'], dataset.train), dataset.label_map,
                                templatizer, best_trigger_ids, global_data.tokenizer, global_data.predictor, args)

    # Streamlit does not like accessing widgets across functions, which is
    # problematic for this "live updating" widget which we want to still
    # display even if the train output is cached. To get around this, we're
    # going to delete the widget and replace it with a very similar looking
    # widget outside the function...no one will ever notice ;)
    trigger_placeholder.empty()

    return (
        best_trigger_tokens,
        current_score/denom,
        dataset.label_map,
        templatizer,
        best_trigger_ids,
        global_data.tokenizer,
        global_data.predictor,
        args,
        train_output
    )


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


def manual_dataset(use_defaults):

    num_train_instances = st.slider("Number of Train Instances", 4, 32, 8)
    any_empty = False
    dataset = []
    data_col, label_col = st.beta_columns([3,1])
    for i in range(num_train_instances):
        default_data = DEFAULT_TRAIN[i]['sentence'] if use_defaults else ''
        default_label = DEFAULT_TRAIN[i]['label'] if use_defaults else ''
        with data_col:
            data = st.text_input("Train Instance " + str(i+1), default_data)
        with label_col:
            label = st.text_input("Train Label " + str(i+1), default_label, max_chars=20)
        if data == "" or label == "":
            any_empty = True
        dataset.append({'sentence': data, 'label': label})

    label_set = list(set(map(lambda x: x['label'], dataset)))
    label_idx = {x: i for i, x in enumerate(label_set)}
    label_map = dict(map(lambda x: (x, x), label_set))

    if any_empty:
        st.warning('Waiting for data to be added')
        st.stop()

    if len(label_set) < 2:
        st.warning('Not enough labels')
        st.stop()

    return Dataset(
        train=dataset,
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

    if train_csv is None:
        st.stop()

    with io.StringIO(train_csv.getvalue().decode('utf-8')) as f:
        reader = csv.DictReader(f)
        train_dataset = list(reader)
    if len(train_dataset) > 64:
        raise ValueError('Train dataset is too large. Please limit the number '
                         'of examples to 64 or less.')

    labels = set(x['label'] for x in train_dataset)
    label_map = {x: x for x in labels}

    return Dataset(
        train=train_dataset,
        label_map=label_map
    )


def run():
    #  css_hack()
    st.title('AutoPrompt Demo')
    st.markdown('''
    For many years, the predominant approach for training machine learning
    models to solve NLP tasks has been to use supervised training data to
    estimate model parameters using maximum likelihood estimation or some
    similar paradigm.  Whether fitting a logistic regression model over a
    bag-of-words, an LSTM over a sequence of GloVe embeddings, or finetuning a
    language model such as ELMo or BERT, the approach is essentially the same.
    However, as language models have become more and more capable of accurately
    generating plausible text a new possibility for solving classification
    tasks has emerged...

    ## Prompting

    Prompting is the method of converting classification tasks into
    *fill-in-the-blanks* problems that can be solved by a language model **without
    modifying the model's internals**. For example, to perform sentiment analysis,
    we may take the sentence we wish to classify and append the text "Overall, this
    movie was ____." and feed it into a language model like so:
    ''')
    st.image('assets/bert-mouth.png', use_column_width=True)
    st.markdown('''
    By measuring whether the language model assigns a higher probability to
    words that are associated with a **positive** sentiment ("good", "great",
    and "fantastic") vs. words that are associated with a **negative**
    sentiment ("bad", "terrible", or "awful") we can infer the
    predicted label for the given input. So in this example, because the word "good"
    has a higher probability than "bad", the predicted label is **positive**.

    ## AutoPrompt

    One issue that arises when using prompts is that it is not usually clear
    how to best pose a task as a fill-in-the-blanks problem in a way that gets
    the most performance from the language model. Even for a simple problem
    like sentiment analysis, we don't know whether it is better to ask whether
    a movie is good/bad, or whether you feel great/terrible about it, and for
    more abstract problems like natural language inference it is difficult to
    even know where to start.

    To cure this writer's block we introduce **AutoPrompt**, a data-driven
    approach for automatic prompt construction. The basic idea is
    straightfoward: instead of writing a prompt, a user need only write a
    **template** that specfies where the *task inputs* go along with placeholders for
    a number of *trigger tokens* that will automatically be learned by the
    model and the *predict token* that the model will fill in:
    ''')
    st.image('assets/template.png', use_column_width=True)
    st.markdown(
    '''
    In each iteration of the search process:
    1. The template is instantiated using a batch of training inputs.
    2. The loss of the model on each input is measured and used to identify a
    number of candidate replacements for the current trigger tokens.
    3. The performance of each candidate is measured on another batch of
    training data, and the best performing candidate is used in the next
    iteration.

    ### Demo

    To give a better sense of how AutoPrompt works, we have provided a simple
    interactive demo. You can generate a prompt using the training data we have
    pre-populated for you, or alternatively write your own training/evaluation
    instances or upload them using a csv below. In addition, you can vary
    some of the training parameters, as well as the template using the sidebar
    on the left.
    '''
    )
    args = Args.from_streamlit()
    dataset_mode = st.radio('How would you like to input your training data?',
                            options=['Example Data', 'Manual Input', 'From CSV'])

    if dataset_mode == 'Example Data':
        dataset = manual_dataset(use_defaults=True)
    elif dataset_mode == 'Manual Input':
        dataset = manual_dataset(use_defaults=False)
    else:
        dataset = csv_dataset()

    button = st.empty()
    clicked = button.button('Train')

    if clicked:
        trigger_tokens, eval_metric, label_map, templatizer, best_trigger_ids, tokenizer, predictor, args, train_output = run_autoprompt(args, dataset, cache_test=CacheTest(False))
    else:
        try:
            trigger_tokens, eval_metric, label_map, templatizer, best_trigger_ids, tokenizer, predictor, args, train_output = run_autoprompt(args, dataset, cache_test=CacheTest(True))
        except CacheMiss:
            st.stop()
        else:
            button.empty()


    st.markdown(f'**Final trigger**: {", ".join(trigger_tokens)}')
    st.dataframe(pd.DataFrame(train_output).style.highlight_min(axis=1, color=HIGHLIGHT_COLOR))
    logger.debug('Dev metric')
    st.write('Accuracy: ' + str(round(eval_metric.item()*100, 1)))
    st.write("""
    Et voila, you've now effectively finetuned a classifier using just a few
    kilobytes of parameters (the tokens in the prompt). If you like you can
    write down your "model" on the back of a napkin and take it with you.

    ### Try it out yourself!

    """)
    sentence = st.text_input("Sentence", 'Enter a test input here')
    pred_output = predict_test([sentence], label_map ,templatizer, best_trigger_ids, tokenizer, predictor, args)
    st.dataframe(pd.DataFrame(pred_output).style.highlight_min(axis=1, color=HIGHLIGHT_COLOR))

    st.markdown('''
    ## Where can I learn more?

    If you are interested in learning more about AutoPrompt we recommend
    [reading our paper](https://arxiv.org/abs/2010.15980) and [checking out our
    code](https://github.com/ucinlp/autoprompt), or if you'd like you can also
    watch our presentation at EMNLP 2020:
    ''')
    st.components.v1.iframe(
        src="https://www.youtube.com/embed/IBMT_oOCBbc",
        height=400,
    )
    st.markdown('Thanks!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout)
    run()

