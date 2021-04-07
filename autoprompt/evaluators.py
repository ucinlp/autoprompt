"""
Evaluators are used to perform MLM evaluation for different tasks.
"""
import json
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class MultipleChoiceEvaluator:
    """Used for multiple choice evaluation."""
    def __init__(
            self,
            model,
            tokenizer,
            decoding_strategy,
            **kwargs
    ):
        raise NotImplementedError('MultipleChoiceEvaluator not updated to use new metrics')
        self._model = model
        self._tokenizer = tokenizer
        self._decoding_strategy = decoding_strategy

    def __call__(self, model_inputs, labels, metric, train=True, **kwargs):
        if train:
            _, logits, *_ = self._model(model_inputs, labels, **kwargs)
            logits = logits.transpose(1, -1)  # Class probs need to be dim 1 for CE
            log_p = -F.cross_entropy(logits, labels, reduction='none')
            log_p = log_p.mean(dim=-1)
            loss = 1 - log_p[0] + log_p[1]
            loss[loss < 0] = 0.0
            # metrics = {'accuracy': log_p[0] > log_p[1]} # TODO: do we need a metric other than accuracy?
            predictions = None  # Too lazy to support ATM
        else:
            loss = torch.tensor(0.0, device=labels.device)
            prediction_idx = self._decode(model_inputs, labels, **kwargs)
            logger.debug(f'Prediction idx: {prediction_idx}')
            # metrics = {'accuracy': (prediction_idx == 0).sum()}
            predicted_instance = labels[prediction_idx]
            prediction_mask = model_inputs['predict_mask'][prediction_idx]
            predicted_label_ids = predicted_instance[prediction_mask]
            predictions = [self._tokenizer.decode(predicted_label_ids)]
        return loss, metrics, predictions

    def _decode(self, model_inputs, labels):
        predict_mask = model_inputs['predict_mask'].clone()

        if self._decoding_strategy == 'parallel':
            _, logits, *_ = self._model(model_inputs, labels)
            logits = logits.transpose(1, -1)
            log_p = -F.cross_entropy(logits, labels, reduction='none')
            log_p = log_p.sum(dim=-1)

        elif self._decoding_strategy == 'monotonic':
            idx0 = torch.arange(predict_mask.size(0), device=labels.device)
            log_p = torch.zeros_like(idx0, dtype=torch.float)
            input_ids = model_inputs['input_ids']
            iterations = predict_mask.sum(dim=-1).max().item()
            for _ in range(iterations):
                logits, *_ = self._model(model_inputs)
                logits = logits.transpose(1, -1)
                scores = -F.cross_entropy(logits, labels, reduction='none')
                scores = scores.transpose(1, -1)
                row_mask = predict_mask.any(dim=-1)
                idx1 = torch.argmax(predict_mask.long(), dim=-1)
                combined_mask = torch.zeros_like(predict_mask)
                combined_mask[idx0, idx1] = row_mask
                score = scores[combined_mask]
                input_ids[combined_mask] = labels[combined_mask]
                log_p[row_mask] += score
                predict_mask[combined_mask] = False

        elif self._decoding_strategy == 'iterative':
            idx0 = torch.arange(predict_mask.size(0), device=labels.device)
            log_p = torch.zeros_like(idx0, dtype=torch.float)
            input_ids = model_inputs['input_ids']
            iterations = predict_mask.sum().item()
            for _ in range(iterations):
                # NOTE: We're going to be lazy and make the search for the most
                # likely prediction easier by setting the logits for any tokens
                # other than the candidates to a huge negative number.
                logits, *_ = self._model(model_inputs)
                logits = logits.transpose(1, -1)
                scores = -F.cross_entropy(logits, labels, reduction='none')
                scores = scores.transpose(1, -1)
                scores[~predict_mask] = -1e32
                score, idx1 = torch.max(scores, dim=-1)
                row_mask = predict_mask.any(dim=-1)
                combined_mask = torch.zeros_like(predict_mask)
                combined_mask[idx0, idx1] = row_mask
                input_ids[combined_mask] = labels[combined_mask]
                log_p[row_mask] += score[row_mask]
                predict_mask[combined_mask] = False

        prediction_idx = log_p.argmax()
        return prediction_idx


class GenerativeEvaluator:
    """Used for generative evaluation."""
    def __init__(
            self,
            model,
            tokenizer,
            decoding_strategy,
            **kwargs
    ):
        raise NotImplementedError('GenerativeEvaluator not updated to use new metrics')
        self._model = model
        self._tokenizer = tokenizer
        self._decoding_strategy = decoding_strategy

    def __call__(self, model_inputs, labels, metric, train=True, **kwargs):
        predict_mask = model_inputs['predict_mask']
        if train:
            loss, logits, *_ = self._model(model_inputs, labels, **kwargs)
            prediction_ids = torch.full_like(model_inputs['input_ids'], -100)
            prediction_ids[predict_mask] = logits.argmax(dim=-1)[predict_mask]
        else:
            loss = torch.tensor(0.0, device=labels.device)
            prediction_ids = self._decode(model_inputs, **kwargs)
        # metrics = {'accuracy': (prediction_ids == labels).all(dim=-1).sum()} # TODO: do we need a metric other than accuracy?

        # Debug printing of predictions.
        predictions = []
        for label, pred, mask in zip(labels, prediction_ids, predict_mask):
            label_text = self._tokenizer.decode(label[mask])
            prediction_text = self._tokenizer.decode(pred[mask])
            predictions.append(json.dumps({
                'label': label_text,
                'prediction': prediction_text,
                'label_tokens': self._tokenizer.convert_ids_to_tokens(label[mask]),
                'prediction_tokens': self._tokenizer.convert_ids_to_tokens(pred[mask]),
            }))

        return loss, metrics, predictions

    def _decode(self, model_inputs, **kwargs):
        """
        Decode from model.

        Parameters
        ==========
        model : transformers.PretrainedModel
            The model to use for predictions.
        model_inputs : Dict[str, torch.LongTensor]
            The model inputs.
        decoding_strategy : str
            The decoding strategy. One of: parallel, monotonic, iterative.
            * parallel: all predictions made at the same time.
            * monotonic: predictions decoded from left to right.
            * iterative: predictions decoded in order of highest probability.
        """

        # initialize output to ignore label.
        output = torch.full_like(model_inputs['input_ids'], -100)
        predict_mask = model_inputs['predict_mask'].clone()

        if self._decoding_strategy == 'parallel':
            # Simple argmax over arguments.
            logits, *_ = self._model(model_inputs, **kwargs)
            preds = logits.argmax(dim=-1)
            output[predict_mask] = preds[predict_mask]

        elif self._decoding_strategy == 'monotonic':
            idx0 = torch.arange(predict_mask.size(0))
            input_ids = model_inputs['input_ids']
            iterations = predict_mask.sum(dim=-1).max().item()
            for _ in range(iterations):
                logits, *_ = self._model(model_inputs, **kwargs)
                row_mask = predict_mask.any(dim=-1)
                idx1 = torch.argmax(predict_mask.long(), dim=-1)
                combined_mask = torch.zeros_like(predict_mask)
                combined_mask[idx0, idx1] = row_mask
                pred = logits[combined_mask].argmax(dim=-1)
                input_ids[combined_mask] = pred
                output[combined_mask] = pred
                predict_mask[combined_mask] = False

        elif self._decoding_strategy == 'iterative':
            idx0 = torch.arange(predict_mask.size(0))
            input_ids = model_inputs['input_ids']
            iterations = predict_mask.sum().item()
            for _ in range(iterations):
                # NOTE: We're going to be lazy and make the search for the most
                # likely prediction easier by setting the logits for any tokens
                # other than the candidates to a huge negative number.
                logits, *_ = self._model(model_inputs, **kwargs)
                logits[~predict_mask] = -1e32
                top_scores, preds = torch.max(logits, dim=-1)
                row_mask = predict_mask.any(dim=-1)
                idx1 = torch.argmax(top_scores, dim=-1)
                combined_mask = torch.zeros_like(predict_mask)
                combined_mask[idx0, idx1] = row_mask
                pred = preds[combined_mask]
                input_ids[combined_mask] = pred
                output[combined_mask] = pred
                predict_mask[combined_mask] = False
        else:
            raise ValueError(
                'Something is really wrong with the control flow in this function'
            )

        return output


class ClassificationEvaluator:
    """Used for evaluating classifiers (e.g., tasks w/ fixed label pools)."""
    def __init__(
        self,
        model,
        tokenizer,
        label_map,
        **kwargs
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._label_map = label_map
        label_tokens = self._tokenizer(
            list(label_map.values()),
            add_special_tokens=False,
            return_tensors='pt',
        )['input_ids']
        if label_tokens.size(1) != 1:
            raise ValueError(
                'Multi-token labels not supported for classifier evaluation'
            )
        self._label_tokens = label_tokens.view(1, -1)
        self._label_keys = list(label_map.keys())

    def __call__(self, model_inputs, labels, metric, train=True, return_probs=False, **kwargs):
        # Ensure everything is on the same device
        label_tokens = self._label_tokens.to(labels.device)

        # Get predictions
        predict_mask = model_inputs['predict_mask']
        labels = labels[predict_mask].unsqueeze(-1)
        logits, *_ = self._model(model_inputs, **kwargs)
        # TODO, this might not work for LAMA
        predict_logits = torch.gather(
            logits[predict_mask],
            dim=-1,
            index=label_tokens.repeat(labels.size(0), 1)
        )

        # calibration, if enabled
        if self._model.calibration_layer is not None:
            predict_logits = self._model.calibration_layer(predict_logits) 

        preds = predict_logits.argmax(dim=-1, keepdims=True)

        # Convert label tokens to their indices in the label map.
        _, label_inds = torch.where(labels.eq(label_tokens))
        
        metric.update(label_inds, preds.squeeze(1))

        predictions = [self._label_keys[i] for i in preds.squeeze(1).tolist()]

        # Get loss
        probs = F.softmax(predict_logits, dim=-1)
        predict_logp = F.log_softmax(predict_logits, dim=-1)
        label_inds = label_inds.unsqueeze(-1)
        loss = -predict_logp.gather(-1, label_inds).mean()

        if return_probs:
            return loss, predictions, probs
        else:
            return loss, predictions


MLM_EVALUATORS = {
    'generative': GenerativeEvaluator,
    'classification': ClassificationEvaluator,
    'multiple-choice': MultipleChoiceEvaluator
}
