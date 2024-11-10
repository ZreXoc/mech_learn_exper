from math import inf
from typing import Optional, Tuple, Union
from huggingface_hub.hf_api import SPACES_SDK_TYPES
import torch
from torch import nn
from torch._dynamo.eval_frame import config
from transformers import BertForTokenClassification
from src.constants import ALL_LABELS, EOS_IDS, NUM_LABLES, PAD_LABEL, SOS_LABEL, EOS_LABEL, SOS_IDS,EOS_IDS,labels_to_ids
from torchcrf import CRF
from src.tokenizer2 import tokenizer
from src.config import config
from src.utils import BIO_to_inner

class NER_Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.bert = BertForTokenClassification.from_pretrained(
            config.pretrained, num_labels=NUM_LABLES, cache_dir='./cache')
        self.bert.resize_token_embeddings(len(tokenizer))

        self.CRF = CRF(NUM_LABLES, batch_first=True)
        # print(ALL_LABELS)
        # print(transitions)
        self.init_transitions()


    def init_transitions(self):
        start_transitions = torch.full((NUM_LABLES,), -100.)
        start_transitions[SOS_IDS] = 0

        transitions = torch.zeros(NUM_LABLES, NUM_LABLES)

        transitions[:, SOS_IDS] = -100
        transitions[EOS_IDS, :] = -100
        for label_from in ALL_LABELS:
            i = labels_to_ids[label_from]
            for label_to in ALL_LABELS:
                j = labels_to_ids[label_to]
                val = -100


                if (label_to == BIO_to_inner(label_from)):
                    val = 0
                if (label_to in [EOS_LABEL, 'O'] or label_to.startswith('B')):
                    val = 0

                # if(label_from == SOS_LABEL and label_to == EOS_LABEL): val = -100
                # if(label_from == EOS_LABEL or label_to == SOS_LABEL): val = -100
                # if(label_from in [SOS_LABEL, EOS_LABEL] or label_to in [SOS_LABEL,EOS_LABEL]): val = -100


                transitions[i, j] = val

        end_transitions = torch.full((NUM_LABLES,), -100.)
        end_transitions[EOS_IDS] = 0
        
        torch.set_printoptions(linewidth=1000)

        self.CRF.start_transitions = nn.Parameter(start_transitions)
        self.CRF.transitions = nn.Parameter(transitions)
        self.CRF.end_transitions = nn.Parameter(end_transitions)




    def forward(self, input_ids, mask, labels, token_type_ids):

        if(labels!=None):
            _, _logits = self.bert(input_ids=input_ids, attention_mask=mask, labels=labels,
                                     token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)
            # print(_,_logits)
            loss = self.CRF(emissions=_logits, tags=labels, mask=mask.bool())
            pred = self.CRF.decode(_logits,mask=mask.bool())
            return -loss, pred
        else:
            _logits = self.bert(input_ids=input_ids, attention_mask=mask, labels=labels, token_type_ids=token_type_ids, return_dict=False, output_hidden_states=False, output_attentions=False)[0]
            # print(_logits)
            pred = self.CRF.decode(_logits,mask=mask.bool())
            return pred
