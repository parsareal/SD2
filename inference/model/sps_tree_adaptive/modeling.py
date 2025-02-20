import torch

TOPK = 10
from .choices import *
from .utils_c import *
from .utils import *
import torch.nn.functional as F
import time
import math
from torch import nn

top_k=10

# def init_tree(self):
#     self.tree = mc_sim_7b_63
#     self.tree_buffer=generate_tree_buffers_draft(self.tree) #,self.embed_tokens.weight.device)

def set_params(self, total_tokens=63, depth=5, top_k=8, threshold=1.0):
    self.top_k = top_k
    self.total_tokens = total_tokens - 1
    self.depth = depth
    self.threshold = math.log(threshold)
    self.logsoftmax = nn.LogSoftmax(dim=-1)


def init_tree(self):
    self.tree_mask_init = torch.eye(self.top_k, device=self.model.embed_tokens.weight.device)[None, None]
    self.position_ids = torch.zeros(self.top_k, device=self.model.embed_tokens.weight.device, dtype=torch.long)
    self.tree_mask_init = self.tree_mask_init.to(self.model.embed_tokens.weight.device)

def reset(self):
    self.tree_mask = None

def customized_sample(self,logits, logits_processor,k=1, replacement=False):
    logits = logits_processor(None, logits)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    sampled_indices = torch.multinomial(probabilities, k, replacement=False)
    sampled_probs = torch.gather(probabilities, 1, sampled_indices)

    cumulative_sum = torch.cumsum(sampled_probs, dim=1)
    cumulative_sum = torch.cat(
        (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

    sampled_probs = sampled_probs / (1 - cumulative_sum)
    sampled_probs[torch.isinf(sampled_probs)] = -1
    sampled_probs[torch.isnan(sampled_probs)] = -1

    sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

    return sampled_indices, sampled_probs,probabilities
    
def reset_kv(self):
    self.stable_kv=None
    
@torch.no_grad()
def topk_generate(self, input_ids, logits_processor, max_length=4, use_cache=True):
    total_tokens = self.total_tokens
    depth = self.depth
    top_k = self.top_k

    sample_token = input_ids[:, -1]

    scores_list = []
    parents_list = []
    ss_token = []

    # input_ids = input_ids[:, 1:]

    self.reset()

    ss_token,ss_prob,ss_op = [],[],[]
    len_posi=input_ids.shape[1]
    
    # with Timer('topk generate'):
    if hasattr(self, "stable_kv") and self.stable_kv is not None:
        kv_len=self.stable_kv[0][0].shape[2]
        outputs = self(input_ids=input_ids[:,kv_len:], past_key_values=self.stable_kv,use_cache=True)
    else:
        outputs = self(input_ids=input_ids, use_cache=True)

    self.stable_kv = outputs['past_key_values']
    last_headout = outputs['logits'][:, -1]

    last_p = self.logsoftmax(last_headout)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    scores = topk_p[0]
    scores_list.append(scores[None])
    parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    ss_token.append(topk_index)
    input_ids = topk_index
    tree_mask = self.tree_mask_init
    topk_cs_index = torch.arange(top_k, device=self.model.embed_tokens.weight.device)

    for i in range(depth):
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        position_ids = position_ids.unsqueeze(0)
        
        # position_ids = position_ids.unsqueeze(0)
        attention_mask = torch.cat((torch.ones(self.tree_mask.shape[0], self.tree_mask.shape[1], self.tree_mask.shape[2], len_posi-i).to('cuda'), self.tree_mask), dim=-1) 
        outputs = self(input_ids=input_ids, past_key_values=outputs['past_key_values'], 
                        position_ids=position_ids, attention_mask=attention_mask, use_cache=True)

        len_posi += 1

        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k ** 2 * bias2 + bias1
        parents = (topk_cs_index + bias)
        parents_list.append(parents)

        last_headout = outputs['logits'][0].clone()
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        cu_scores = topk_p + scores[:, None]

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p

        out_ids = topk_cs_index // top_k
        input_ids = topk_index.view(-1)[topk_cs_index][None]

        ss_token.append(topk_index)
        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)


    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
    # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()
    # with Timer("mask"):
    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

    tree_position_ids = torch.sum(tree_mask, dim=1) - 1

    tree_mask = tree_mask.float()[None, None]
    draft_tokens = draft_tokens[None]

    del parents_list, scores_list, ss_token, ss_token_list, draft_parents

    # with Timer("retrieve"):

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()

    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    if logits_processor is not None:
        maxitem = total_tokens + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = sorted(retrieve_indices, key=custom_sort)

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
    tree_position_ids = tree_position_ids #.to(input_ids.device)

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids