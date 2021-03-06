import streamlit as st
import pandas as pd
from autoprompt import utils
from autoprompt.create_trigger import *

class Object(object):
    pass

def autoprompt_args():
    a = Object()
    a.template = st.sidebar.text_input("Template string", "[CLS] {sentence} [T] [T] [T] [P] . [SEP]")
    a.tokenize_labels = True
    a.filter = st.sidebar.checkbox("Filter", True)
    a.print_lama = False
    a.initial_trigger = None
    a.label_field = "label"

    a.bsz = 32
    a.eval_size = 1
    a.iters = int(st.sidebar.number_input("Iterations", 1, value=10))
    a.accumulation_steps = int(st.sidebar.number_input("Accumulation Steps", value=1))
    a.model_name = st.sidebar.selectbox("Model name", ['bert-base-cased', 'roberta-large'])
    a.seed = int(st.sidebar.number_input("seed", value=0))
    a.limit = None
    a.use_ctx = st.sidebar.checkbox("Use context sentences", False)
    a.perturbed = st.sidebar.checkbox("Perturbed", False)
    a.patience = int(st.sidebar.number_input("Patience", value=5))
    a.num_cand = int(st.sidebar.number_input("Num Candidates", value=10))
    a.sentence_size = int(st.sidebar.number_input("Sentence Size", value=50))
    return a

def load_trigger_dataset(dataset, templatizer, use_ctx, limit=None):
    instances = []

    for x in dataset:
        try:
            if use_ctx:
                # For relation extraction, skip facts that don't have context sentence
                if 'evidences' not in x:
                    logger.warning('Skipping RE sample because it lacks context sentences: {}'.format(x))
                    continue

                evidences = x['evidences']
                    
                # Randomly pick a context sentence
                obj_surface, masked_sent = random.choice([(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
                words = masked_sent.split()
                if len(words) > MAX_CONTEXT_LEN:
                    # If the masked sentence is too long, use the first X tokens. For training we want to keep as many samples as we can.
                    masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])
                
                # If truncated context sentence still has MASK, we need to replace it with object surface
                # We explicitly use [MASK] because all TREx fact's context sentences use it
                context = masked_sent.replace('[MASK]', obj_surface)
                x['context'] = context
                model_inputs, label_id = templatizer(x)
            else:
                # CHANGED
                model_inputs, label_id = templatizer(x)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            instances.append((model_inputs, label_id))
    if limit:
        return random.sample(instances, limit)
    else:
        return instances

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    global_data = Object()
    global_data.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading model, tokenizer, etc.')
    global_data.config, model, global_data.tokenizer = load_pretrained(model_name)
    global_data.model = model
    global_data.model.to(global_data.device)
    global_data.embeddings = get_embeddings(global_data.model, global_data.config)
    global_data.embedding_gradient = GradientStorage(global_data.embeddings)
    global_data.predictor = PredictWrapper(global_data.model)
    return global_data
    
@st.cache(persist=False, suppress_st_warning=True, max_entries=100, allow_output_mutation=True)
def run_autoprompt(global_data, args, dataset):
    model = global_data.model
    tokenizer = global_data.tokenizer
    config = global_data.config
    device = global_data.device
    embeddings = global_data.embeddings
    embedding_gradient = global_data.embedding_gradient
    predictor = global_data.predictor

    set_seed(args.seed)

    if args.label_map is not None:
        ## CHANGED
        label_map = args.label_map
        # label_map = json.loads(args.label_map)
        # logger.info(f"Label map: {label_map}")
    else:
        label_map = None

    templatizer = utils.TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        use_ctx=args.use_ctx
    )

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
    if label_map:
        evaluation_fn = AccuracyFn(tokenizer, label_map, device)
    else:
        evaluation_fn = lambda x, y: -get_loss(x, y)

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    if args.perturbed:
        train_dataset = utils.load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        ## CHANGED
        # train_dataset = utils.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
        train_dataset = load_trigger_dataset(dataset, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = utils.load_augmented_trigger_dataset(args.dev, templatizer)
    else:
        ## CHANGED
        # dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
        dev_dataset = load_trigger_dataset(dataset, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
    filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = utils.encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = -1e32
        else:
            for _, label_ids in train_dataset:
                filter[label_ids] = -1e32
        logger.info('Filtering special tokens and capitalized words.')
        for word, idx in tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= tokenizer.vocab_size:
                continue
            # Filter special tokens.
            if idx in tokenizer.all_special_ids:
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32
            # Filter capitalized words (lazy way to remove proper nouns).
            if isupper(idx, tokenizer):
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32

    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = -float('inf')
    # Measure elapsed time of trigger search
    start = time.time()
    progress = st.progress(0.0)
    for i in range(args.iters):

        logger.info(f'Iteration: {i}')
        # ADDED
        progress.progress(float(i)/args.iters)
        logger.info('Accumulating Gradient')
        model.zero_grad()

        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:

            # Shuttle inputs to GPU
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = embedding_gradient.get()
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

        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand,
                                    filter=filter)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
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
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
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
                    predict_logits = predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)

                candidate_scores[i] += eval_metric.sum()

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        if args.print_lama:
            if trigger_ids.eq(tokenizer.mask_token_id).any():
                current_score = float('-inf')

        if (candidate_scores > current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            continue

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        if args.print_lama:
            if best_trigger_ids.eq(tokenizer.mask_token_id).any():
                best_dev_metric = float('-inf')

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric
    progress.progress(1.0)

    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    if args.print_lama:
        # Templatize with [X] and [Y]
        if args.use_ctx:
            model_inputs, label_ids = templatizer({
                'sub_label': '[X]',
                'obj_label': tokenizer.lama_y,
                'context': ''
            })
        else:
            model_inputs, label_ids = templatizer({
                'sub_label': '[X]',
                'obj_label': tokenizer.lama_y,
            })
        lama_template = model_inputs['input_ids']
        # Instantiate trigger tokens
        lama_template.masked_scatter_(
            mask=model_inputs['trigger_mask'],
            source=best_trigger_ids.cpu())
        # Instantiate label token
        lama_template.masked_scatter_(
            mask=model_inputs['predict_mask'],
            source=label_ids)
        # Print LAMA JSON template
        relation = args.train.parent.stem

        # The following block of code is a bit hacky but whatever, it gets the job done
        if args.use_ctx:
            template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[SEP] ', '').replace('</s> ', '').replace('[ X ]', '[X]')
        else:
            template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[ X ]', '[X]')

        out = {
            'relation': args.train.parent.stem,
            'template': template
        }
        print(json.dumps(out))

    def predict(sentences):
        # Evaluate clean
        output = { 'sentences': [] }
        any_label = None
        for label in label_map.values():
            output[label] = []
            any_label = label
        output['prompt'] = []
        for sentence in sentences:
            model_inputs, _ = templatizer({'sentence': sentence, 'label': any_label})
            
            prompt_ids = replace_trigger_tokens(
                model_inputs, best_trigger_ids, model_inputs['trigger_mask'])
            # st.write(prompt_ids)
            # st.write(prompt_ids.shape)

            prompt = ' '.join(tokenizer.convert_ids_to_tokens(prompt_ids['input_ids'][0]))
            output['prompt'].append(prompt)

            predict_logits = predictor(model_inputs, best_trigger_ids)
            output['sentences'].append(sentence)
            for label in label_map.values():
                label_id = utils.encode_label(tokenizer=tokenizer, label=label, tokenize=args.tokenize_labels)
                label_loss = get_loss(predict_logits, label_id)
                # st.write(sentence, label, label_loss)
                output[label].append(label_loss.item())
        return output
    dev_output = predict(map(lambda x: x['sentence'], dataset))
    st.dataframe(pd.DataFrame(dev_output).style.highlight_min(axis=1))
    return best_trigger_tokens, best_dev_metric, predict

def run():
    st.title('AutoPrompt Demo')
    st.write("Give some examples, get a model!")
    st.markdown("See https://ucinlp.github.io/autoprompt/ for more details.")
    num_train_instances = st.slider("Number of Train Instances", 2, 50)

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

    args = autoprompt_args()
    label_set = set(map(lambda x: x['label'], dataset))
    args.label_map = dict(map(lambda x: (x, x), label_set))
    global_data = load_model(args.model_name)

    if any_empty:
        st.warning('Waiting for data to be added')
        st.stop()

    if len(label_set) < 2:
        st.warning('Not enough labels')
        st.stop()

    st.write('Training...')
    trigger_tokens, dev_metric, predict = run_autoprompt(global_data, args, dataset)
    st.write('Tokens: ' + ' ,'.join(trigger_tokens))
    st.write('Train accuracy: ' + str(round(dev_metric*100, 1)))
    st.write("### Test Predictions")
    eval_output = predict(map(lambda x: x['sentence'], eval_dataset))
    st.dataframe(pd.DataFrame(eval_output).style.highlight_min(axis=1))
    st.write("### Let's test it ourselves!")
    sentence = st.text_input("Sentence", dataset[1]['sentence'])
    pred_output = predict([sentence])
    st.dataframe(pd.DataFrame(pred_output).style.highlight_min(axis=1))

run()