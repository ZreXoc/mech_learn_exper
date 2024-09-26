import torch
from transformers import BertForTokenClassification
from src.constants import LABELS, MODEL_NAME

class BertModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        # self.bert = BertForTokenClassification.from_pretrained(
                       # '/home/xic/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f', 
                                     # num_labels=num_labels)
        # +1 for special label

    def forward(self, input_id, mask, label, token_type_ids):
        output = self.bert(input_ids=input_id,     attention_mask=mask,                        labels=label,
                           token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
        return output
