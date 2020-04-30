import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import BertTokenizer, BertForMaskedLM, BertModel
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
# from transformers.modeling_bert import BertOnlyMLMHead
class MyBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            a = (prediction_scores[:,:,1133]).unsqueeze(2) #todo change this from hard code.
            b = (prediction_scores[:,:,1105]).unsqueeze(2)
            c = torch.cat((a,b),2)
            masked_lm_labels[masked_lm_labels==1133]=0
            masked_lm_labels[masked_lm_labels==1105]=1
            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            masked_lm_loss = loss_fct(c.view(-1, 2), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
