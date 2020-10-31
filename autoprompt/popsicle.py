"""
Frozen model with a linear topping...I'm really sleepy...
"""
import logging

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    BertConfig,
    BertForSequenceClassification,
    PretrainedConfig,
    RobertaConfig,
    RobertaForSequenceClassification
)


logger = logging.getLogger(__name__)



class Bertsicle(BertForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        with torch.no_grad():
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        pooled_output = outputs[1]  #by ROB
        pooled_output = outputs[0]
        pooled_output = pooled_output[:,1:,:] #eliminating CLS token
        pooled_output = torch.mean(pooled_output, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class Robertasicle(RobertaForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        with torch.no_grad():
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 1:, :]  # eliminating <s> token
        pooled_sequence_output = torch.mean(sequence_output, dim=1, keepdim=True)
        logits = self.classifier(pooled_sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


MODEL_MAPPING = {
        RobertaConfig: Robertasicle,
        BertConfig: Bertsicle
}


class AutoPopsicle:
    def __init__(self):
        raise EnvironmentError('You done goofed. Use `.from_pretrained()` or something.')

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError('We do not support this config.')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                logger.info(f'Config class: {config_class}')
                logger.info(f'Model class: {model_class}')
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError('We do not support "{pretrained_model_name_or_path}".')
