import copy
import logging
import os
from typing import Optional

import torch
import transformers


logger = logging.getLogger(__name__)


TRIGGER_WEIGHTS_NAME = 'triggers.bin'


class ContinuousTriggerConfig(transformers.PretrainedConfig):
    model_type = 'continuous-triggers'
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Instantiate model config.
        assert (
            'model' in kwargs and 'trigger_length' in kwargs
        )
        model_config = kwargs.pop('model')
        model_type = model_config.pop('model_type')
        self.model = transformers.AutoConfig(model_type, **model_config)

        # Add trigger-specific parameters.
        self.trigger_length = kwargs.pop('trigger_length')
        self.initial_trigger_token_ids = kwargs.pop('initial_trigger_token_ids', None)
        self.freeze_model_weights = kwargs.pop('freeze_model_weights', True)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['model'] = self.model.to_dict()
        output['model_type'] = self.__class__.model_type
        return output


class ContinuousTriggerBaseModel(transformers.PreTrainedModel):
    """
    Base class for a pretrained transformer model w/ learned continuous
    triggers. Essentially, just a wrapper around AutoModel that handles some of
    the tricky parts of model initialization and checkpoint restoration.
    """
    config_class = ContinuousTriggerConfig
    auto_model_cls = None

    def __init__(self, config):
        super().__init__(config)
        self.model = self.auto_model_cls.from_config(config.model)
        self.triggers = torch.nn.Embedding(
            config.trigger_length,
            config.model.hidden_size,
        )

    def forward(self):
        raise NotImplementedError

    @classmethod
    def from_pretrained_model(
            cls,
            pretrained_model_name_or_path,
            **kwargs
    ):
        config = kwargs.pop("config", None)
        if not isinstance(config, transformers.PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
        
        trigger_length = kwargs.pop('trigger_length', None)
        if cls.auto_model_cls is None:
            raise ValueError(
                f'Cannot restore checkpoint. No AutoModel associated to {cls}. '
                'Maybe you meant to use a subclass of ContinuousTriggerBaseModel?'
            )
        if os.path.isdir(pretrained_model_name_or_path):
            config = 
            model = cls.auto_model_cls.from_pretrained(pretrained_model_name_or_path)
            triggers = 
            trigger_path = os.path.join(pretrained_model_name_or_path, TRIGGER_WEIGHTS_NAME)
            if os.path.exists(trigger_path):
        else:
            model = 


class ContinuousTriggerForSequenceClassification(ContinuousTriggerBaseModel):
    auto_model_cls = ContinuousTriggerForSequenceClassification


class ContinuousTriggerForMaskedLM(ContinuousTriggerBaseModel):
    auto_model_cls = ContinuousTriggerForMaskedLM




