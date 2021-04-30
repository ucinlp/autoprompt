"""Transformer model wrappers."""
import warnings

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
)


def get_word_embeddings(model):
    """Gets the word embeddings from a transformer model."""
    for module in model.modules():
        if hasattr(module, 'word_embeddings'):
            return module.word_embeddings
    raise NotImplementedError(
        'Unable to retrieve word embeddings for model. You may need to add a special case to '
        '`get_word_embeddings`.'
    )


def get_lm_head(model):
    """Gets the lm head from a transformer model."""
    for module in model.modules():
        if hasattr(module, 'cls'):
            return module.cls
        if hasattr(module, 'lm_head'):
            return module.lm_head
        if hasattr(module, 'predictions'):
            return module.predictions
    raise NotImplementedError(
        'Unable to retrieve MLM head for model. You may need to add a special case to '
        '`get_lm_head`.'
    )


def get_clf_head(model):
    """Gets the clf head from a transformer model."""
    for module in model.modules():
        if hasattr(module, 'classifier'):
            return module.classifier
    raise NotImplementedError(
        'Unable to retrieve classifier head for model. You may need to add a special case '
        'to `get_clf_head`.'
    )


class ContinuousTriggerMLM(torch.nn.Module):
    """
    A masked language model w/ continuous triggers.

    Generic wrapper for HuggingFace transformers models that handles naming conventions and
    instantiating triggers during forward pass.

    Parameters
    ===
    base_model : transformers.XModelForMaskedLM
        Any HuggingFace masked language model.
    num_trigger_tokens : int
        The number of trigger tokens.
    initial_trigger_ids: Optional[torch.LongTensor]
        Token ids used to initialize the trigger embeddings.
    """
    def __init__(
            self,
            base_model,
            num_trigger_tokens,
            initial_trigger_ids=None
    ):
        super().__init__()
        self.base_model = base_model

        # To deal with inconsistent naming conventions, we give universal names to these modules.
        self.word_embeddings = get_word_embeddings(base_model)
        self.lm_head = get_lm_head(base_model)

        # Create trigger parameter.
        self.trigger_embeddings = torch.nn.Parameter(
            torch.randn(
                num_trigger_tokens,
                self.word_embeddings.weight.size(1),
            ),
        )
        if initial_trigger_ids is not None:
            self.trigger_embeddings.data.copy_(self.word_embeddings(initial_trigger_ids))

        self.calibration_layer = None

    def forward(self, model_inputs, labels=None):
        """
        Run model forward w/ preprocessing for continuous triggers.

        Parameters
        ==========
        model : transformers.PretrainedModel
            The model to use for predictions.
        model_inputs : Dict[str, torch.LongTensor]
            The model inputs.
        labels : torch.LongTensor
            (optional) Tensor of labels. Loss will be returned if provided.
        """
        # Ensure destructive pop operations are only limited to this function.
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        input_ids = model_inputs.pop('input_ids')
        del model_inputs['predict_mask']

        # Get embeddings of input sequence
        inputs_embeds = self.word_embeddings(input_ids)

        # Insert trigger embeddings into input embeddings.
        batch_size = input_ids.size(0)
        reps = trigger_mask.sum(dim=-1) // self.trigger_embeddings.size(0)
        for row, t_mask, rep in zip(inputs_embeds, trigger_mask, reps):
            row[t_mask] = self.trigger_embeddings.repeat((rep.item(), 1))
        model_inputs['inputs_embeds'] = inputs_embeds

        return self.base_model(**model_inputs, labels=labels)


class LinearComboMLM(torch.nn.Module):
    """
    A masked language model w/ continuous triggers.

    Generic wrapper for HuggingFace transformers models that handles naming conventions and
    instantiating triggers during forward pass.

    Parameters
    ===
    base_model : transformers.XModelForMaskedLM
        Any HuggingFace masked language model.
    num_trigger_tokens : int
        The number of trigger tokens.
    """
    def __init__(self, base_model, num_trigger_tokens, **kwargs):
        if 'initial_trigger_ids' in kwargs:
            warnings.warn('LinearComboMLM does not support initial triggers. Ignoring.')
        super().__init__()
        self.base_model = base_model
        self.word_embeddings = get_word_embeddings(base_model)
        self.lm_head = get_lm_head(base_model)
        # TODO(rloganiv): May need to flip
        self.trigger_projection = torch.nn.Parameter(
            torch.zeros(
                num_trigger_tokens,
                self.base_model.config.vocab_size,
            )
        )
        torch.nn.init.xavier_normal_(self.trigger_projection)

    def forward(self, model_inputs, labels=None, trigger_ids=None):
        # Ensure destructive pop operations are only limited to this function.
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        input_ids = model_inputs.pop('input_ids')
        del model_inputs['predict_mask']

        # Get embeddings of input sequence
        inputs_embeds = self.word_embeddings(input_ids)
        trigger_embeddings = torch.einsum(
            'tv,ve->te',
            self.trigger_projection,
            self.word_embeddings.weight,
        )

        # Insert trigger embeddings into input embeddings.
        batch_size = input_ids.size(0)
        inputs_embeds[trigger_mask] = trigger_embeddings.repeat((batch_size,1))
        model_inputs['inputs_embeds'] = inputs_embeds

        return self.base_model(**model_inputs, labels=labels)


class DiscreteTriggerMLM(torch.nn.Module):
    """
    A masked language model w/ discrete triggers.

    Generic wrapper for HuggingFace transformers models that handles naming conventions and
    instantiating triggers during forward pass.

    Parameters
    ===
    base_model : transformers.XModelForMaskedLM
        Any HuggingFace masked language model.
    initial_trigger_ids: torch.LongTensor
        Token ids used to initialize the trigger.
    """
    def __init__(
            self,
            base_model,
            initial_trigger_ids,
    ):
        super().__init__()
        self.base_model = base_model

        # To deal with inconsistent naming conventions, we give universal names to these modules.
        self.word_embeddings = get_word_embeddings(base_model)
        self.lm_head = get_lm_head(base_model)

        self.register_buffer('trigger_ids', initial_trigger_ids)

    def forward(self, model_inputs, labels=None, trigger_ids=None):
        """
        Run model forward w/ preprocessing for continuous triggers.

        Parameters
        ==========
        model_inputs : Dict[str, torch.LongTensor]
            The model inputs.
        labels : torch.LongTensor
            (optional) Tensor of labels. Loss will be returned if provided.
        trigger_ids : torch.LongTensor
            (optional) Tensor of trigger ids. Used to override existing ids during candidate
            evaluation.
        """
        # Ensure destructive pop operations are only limited to this function.
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        del model_inputs['predict_mask']

        # Update input ids in-place.
        input_ids = model_inputs['input_ids']
        batch_size = input_ids.size(0)
        if trigger_ids is None:
            trigger_ids = self.trigger_ids
        trigger_ids = trigger_ids.repeat((batch_size,))
        input_ids[trigger_mask] = trigger_ids

        return self.base_model(**model_inputs, labels=labels)


# TODO(rloganiv): Make consistent with other models.
class ContTriggerTransformer(PreTrainedModel):
    """
    Continuous trigger transformer for sequence classification.
    """
    def __init__(self, config, model_name, trigger_length):
        super().__init__(config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.embeds = get_word_embeddings(self.model)
        indices = np.random.randint(0, self.embeds.weight.shape[0], size=trigger_length)
        self.relation_embeds = torch.nn.Parameter(self.embeds.weight.detach()[indices], requires_grad=True)
        self.clf_head = get_clf_head(self.model)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Run model forward w/ preprocessing for continuous triggers.
        """
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        return output
