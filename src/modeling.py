import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer,BertForMaskedLM,
                          AutoModelForSequenceClassification, BertModel, BertLayer,
                          PreTrainedModel)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.file_utils import ModelOutput
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

logger = logging.getLogger(__name__)
@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

@dataclass
class DistillOutput(ModelOutput):
    s2t_loss: Optional[Tensor] = None
    t2s_loss: Optional[Tensor] = None
    retriever_ce_loss: Optional[Tensor] = None
    reranker_ce_loss: Optional[Tensor] = None

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    accuracy: Optional[Tensor] = None
    logits: Optional[Tensor] = None

class Reranker(nn.Module):
    def __init__(self, args):
        super(Reranker, self).__init__()
        self.lm = AutoModelForSequenceClassification.from_pretrained(args['model_name_or_path'], num_labels=1, output_hidden_states=True)
        if args['gradient_checkpoint']:
            self.lm.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        ret = self.lm(**batch, return_dict=True)
        logits = ret.logits
        if self.training:
            scores = logits.view(-1, 2)
            target_label = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target_label)
            return RerankerOutput(loss=loss)
        else:
            # inference
            scores = logits.view(-1, 2)
            target_label = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target_label)
            accuracy = torch.sum(torch.argmax(scores, dim=1)==target_label) / scores.size(0)
            return RerankerOutput(loss=loss, accuracy=accuracy, logits=logits)

    def save_pretrained(self, save_directory):
        self.lm.save_pretrained(save_directory, safe_serialization=False)

class DualEncoder(nn.Module):
    def __init__(self, args):
        super(DualEncoder, self).__init__()
        try:
            self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
            # self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
        except:
            self.lm_q = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False, output_hidden_states=True)
            # self.lm_p = AutoModel.from_pretrained(args.retriever_model_name_or_path, use_cache=False, output_hidden_states=True)
        self.lm_p = self.lm_q
        # if args.gradient_checkpoint:
            # self.lm_q.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        # if self.args.negatives_x_device:
        #     if not dist.is_initialized():
        #         raise ValueError('Distributed training has not been initialized for representation all gather.')


    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def encode_query(self, query_inputs):
        qry_out = self.lm_q(**query_inputs, return_dict=True)
        q_hidden = qry_out.hidden_states[-1]
        q_reps = q_hidden[:, 0]
        return q_reps

    def encode_passage(self, passage_inputs):
        psg_out = self.lm_p(**passage_inputs, return_dict=True)
        p_hidden = psg_out.hidden_states[-1]  # 需要输入decoder
        p_reps = p_hidden[:, 0]
        return p_reps
    
    def forward(self, query_inputs=None, passage_inputs=None):
        self.process_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.training:
            q_reps = self.encode_query(query_inputs)
            p_reps = self.encode_passage(passage_inputs)

            if self.args.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if self.args.negatives_in_device:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
            else:
                p_reps = p_reps.view(-1, self.args.sample_num, 768)
                scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))
                scores = scores.squeeze(1)
                target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target)
            return EncoderOutput(loss=loss)
        else:
            if query_inputs is not None and passage_inputs is not None:
                q_reps = self.encode_query(query_inputs)
                p_reps = self.encode_passage(passage_inputs)
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                return EncoderOutput(scores=scores)
            elif passage_inputs is not None:
                p_reps = self.encode_passage(passage_inputs)
                return EncoderOutput(p_reps=p_reps)
            else:
                q_reps = self.encode_query(query_inputs)
                return EncoderOutput(q_reps=q_reps)
    
    def save_pretrained(self, save_directory):
        self.lm_q.save_pretrained(save_directory)

class DualEncoderMeanPooling(nn.Module):
    def __init__(self, args):
        super(DualEncoderMeanPooling, self).__init__()
        try:
            self.lm_q = AutoModel.from_pretrained(args.model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
            self.lm_p = AutoModel.from_pretrained(args.model_name_or_path, use_cache=False,  add_pooling_layer=False, output_hidden_states=True)
        except:
            self.lm_q = AutoModel.from_pretrained(args.model_name_or_path, use_cache=False, output_hidden_states=True)
            self.lm_p = AutoModel.from_pretrained(args.model_name_or_path, use_cache=False, output_hidden_states=True)
        
        if args.gradient_checkpoint:
            self.lm_q.gradient_checkpointing_enable()
            self.lm_p.gradient_checkpointing_enable()
        self.args = args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode_query(self, query_inputs):
        qry_out = self.lm_q(**query_inputs, return_dict=True)
        q_hidden = qry_out.hidden_states[-1]
        q_reps = self.mean_pooling(q_hidden, query_inputs['attention_mask'])
        # q_reps = q_hidden[:, 0]
        return q_reps

    def encode_passage(self, passage_inputs):
        psg_out = self.lm_p(**passage_inputs, return_dict=True)
        p_hidden = psg_out.hidden_states[-1]
        p_reps = self.mean_pooling(p_hidden, passage_inputs['attention_mask'])
        return p_reps

    def forward(self, query_inputs=None, passage_inputs=None):
        if self.training:
            q_reps = self.encode_query(query_inputs)
            p_reps = self.encode_passage(passage_inputs)

            if self.args.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            if self.args.negatives_in_device:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
            else:
                p_reps = p_reps.view(-1, self.args.sample_num, 768)
                scores = torch.matmul(q_reps.unsqueeze(1), p_reps.transpose(2,1))
                scores = scores.squeeze(1)
                target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = self.cross_entropy(scores, target)
            return loss
        else:
            if query_inputs is not None and passage_inputs is not None:
                q_reps = self.encode_query(query_inputs)
                p_reps = self.encode_passage(passage_inputs)
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                return scores
            elif passage_inputs is not None:
                return self.encode_passage(passage_inputs)
            else:
                return self.encode_query(query_inputs)




