import torch

import autoprompt.utils as utils


# TODO: Discrete trigger model.


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
        self.word_embeddings = utils.get_word_embeddings(base_model)
        self.lm_head = utils.get_lm_head(base_model)

        # Create trigger parameter.
        self.trigger_embeddings = torch.nn.Parameter(
            torch.randn(
                num_trigger_tokens,
                self.word_embeddings.weight.size(1),
            ),
        )
        if initial_trigger_ids is not None:
            self.trigger_embeddings.data.copy_(self.word_embeddings(initial_trigger_ids))

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
        predict_mask = model_inputs.pop('predict_mask')
        input_ids = model_inputs.pop('input_ids')

        # Get embeddings of input sequence
        inputs_embeds = self.word_embeddings(input_ids)

        # Insert trigger embeddings into input embeddings.
        batch_size = input_ids.size(0)
        inputs_embeds[trigger_mask] = self.trigger_embeddings.repeat((batch_size, 1))
        model_inputs['inputs_embeds'] = inputs_embeds

        return self.base_model(**model_inputs, labels=labels)

