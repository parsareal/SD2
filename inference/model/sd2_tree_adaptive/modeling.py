import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.cache_utils import Cache, DynamicCache
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .choices import *
from .utils_c import *
from .utils import *
import torch.nn.functional as F
import time
import math
from torch import nn

TOPK = 10
top_k=10

# def init_tree(self):
#     self.tree = mc_sim_7b_63
#     self.tree_buffer=generate_tree_buffers_draft(self.tree) #,self.embed_tokens.weight.device)

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

_CONFIG_FOR_DOC = "LlamaConfig"

@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def soft_llamaForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    draft_target: Optional[str] = 'draft',
    submodel_layer: Optional[int] = 20,
    adaptive_layer: Optional[int] = 20,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # print(f'Using Cache: {use_cache}')
    # if input_ids is not None:
    #     print(f'input_ids: {input_ids.shape}')
    # if inputs_embeds is not None:
    #     print(f'input_embeds: {inputs_embeds.shape}')

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.soft_llamaModel_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        draft_target=draft_target,
        submodel_layer=submodel_layer,
        adaptive_layer=adaptive_layer,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def soft_llamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    draft_target: Optional[str] = 'draft',
    submodel_layer: Optional[int] = 20,
    adaptive_layer: Optional[int] = 20,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    if draft_target == 'draft':
        intermediate_layers = self.layers[:submodel_layer]
    elif draft_target == 'target':
        intermediate_layers = self.layers[submodel_layer:]
    elif draft_target == 'adaptive':
        intermediate_layers = self.layers[adaptive_layer:submodel_layer]

    # for decoder_layer in self.layers:
    for idx, decoder_layer in enumerate(intermediate_layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            # print(f'past_key_values: {past_key_values[0][0].shape}')
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # add hidden states from the last decoder layer before norm
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # print(f'kv_seq_len before past_key_value.get_usable_length: {kv_seq_len}')
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    # print(f'kv_seq_len after past_key_value.get_usable_length: {kv_seq_len}')
    
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # print(f'layer idx: {self.layer_idx}')
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


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

# def adjust_submodels_hiddenstates(self):
#     self.all_layers_hidden_states_dict = {}
#     self.all_layers_number_of_exits = {}
#     for submodel in self.intermediate_layers:
#         self.all_layers_hidden_states_dict[submodel] = [0, None]
#         self.all_layers_number_of_exits[submodel] = 0

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
        # self.adjust_submodels_hiddenstates()
        # kv_len=self.stable_kv[0][0].shape[2]
        # outputs = self.soft_llamaForCausalLM_forward(input_ids=input_ids[:,kv_len:], past_key_values=self.stable_kv,use_cache=True)
        adaptive_input_cache = self.stable_kv
        for layer, threshold in zip(self.intermediate_layers, self.intermediate_thresholds):
            kv_len=self.stable_kv[layer-1][0].shape[2]
            if layer != self.intermediate_layers[0]:
                pre_layer = self.intermediate_layers[self.intermediate_layers.index(layer)-1]
            else:
                pre_layer = None

            # Calculate input hidden states 
            if layer != self.intermediate_layers[0] and self.layers_verified_hidden_states_dict[pre_layer] != None:
                # concat the hidden state with the current one
                input_hidden_states = torch.cat((self.layers_verified_hidden_states_dict[pre_layer], outputs.hidden_states[-2]), dim=-2)
                self.layers_verified_hidden_states_dict[pre_layer] = None
            else:
                input_hidden_states = outputs.hidden_states[-2] if layer != self.intermediate_layers[0] else None

            if layer != self.intermediate_layers[0]:
                    new_past = []
                    maximum_length = adaptive_input_cache[layer-1][0].shape[2]
                    # print(f'maximum_length: {maximum_length}')
                    for idx in range(len(adaptive_input_cache)):
                        new_past.append(
                            (
                                adaptive_input_cache[idx][0][:, :, :maximum_length, :],
                                adaptive_input_cache[idx][1][:, :, :maximum_length, :],
                            )
                        )
                    new_past = tuple(new_past)
            else:
                new_past = adaptive_input_cache

            
            # print('first call')
            # print(f'layer: {layer}')
            # print(f'input_ids: {input_ids.shape}')
            # print(f'new_past 2: {new_past[0][0].shape}')
            # print(f'new_past 4: {new_past[2][0].shape}')
            # print(f'new_past 6: {new_past[4][0].shape}')
            # print(f'adaptive_input_cache 2: {adaptive_input_cache[0][0].shape}')
            # print(f'adaptive_input_cache 4: {adaptive_input_cache[2][0].shape}')
            # print(f'adaptive_input_cache 6: {adaptive_input_cache[4][0].shape}')
            # if input_hidden_states is not None:
            #     print(f'input_hidden_states: {input_hidden_states.shape}')
            # for l in self.layers_verified_hidden_states_dict:
            #     if self.layers_verified_hidden_states_dict[l] is not None:
            #         print(f'layers_verified_hidden_states_dict {l}: {self.layers_verified_hidden_states_dict[l].shape}')

            # print('*****************************')
            

            # with Timer('forward call:'):
            outputs = self.soft_llamaForCausalLM_forward(
                    input_ids=input_ids[:,kv_len:] if layer==self.intermediate_layers[0] else None,
                    inputs_embeds=input_hidden_states if layer!=self.intermediate_layers[0] else None,
                    past_key_values=new_past,
                    return_dict=True,
                    use_cache=True,
                    output_hidden_states=True,
                    draft_target='draft' if layer == self.intermediate_layers[0] else 'adaptive',
                    submodel_layer=layer,
                    adaptive_layer=pre_layer,
                )
            
            if layer != self.intermediate_layers[0]:
                new_output_past = list(outputs['past_key_values'])
                new_output_past[:pre_layer] = list(adaptive_input_cache)[:pre_layer]
                adaptive_input_cache = tuple(new_output_past)
            else:
                adaptive_input_cache = outputs['past_key_values']

            next_token_logits = outputs.logits[0][-1]

            next_tokens_probs = next_token_logits.softmax(dim=-1)

            # print('initalized')
            # print(f'layer: {layer}, probs: {torch.max(next_tokens_probs, dim=-1).values}')
            # print(next_token_logits.shape)
            # print(next_tokens_probs.shape)
            if torch.max(next_tokens_probs, dim=-1).values > threshold or layer == self.intermediate_layers[-1]:
                if layer != self.intermediate_layers[-1]:
                    if self.layers_verified_hidden_states_dict[layer] != None:
                        self.layers_verified_hidden_states_dict[layer] = torch.cat((self.layers_verified_hidden_states_dict[layer], outputs.hidden_states[-2]), dim=-2)
                    else:
                        self.layers_verified_hidden_states_dict[layer] = outputs.hidden_states[-2]
                break
            else:
                continue
        self.stable_kv = adaptive_input_cache
    else:
        outputs = self.soft_llamaForCausalLM_forward(input_ids=input_ids, use_cache=True)
        # define global verified hidden states dict
        self.layers_verified_hidden_states_dict = {}
        for layer in self.intermediate_layers:
            self.layers_verified_hidden_states_dict[layer] = None

        self.stable_kv = outputs['past_key_values']
        adaptive_input_cache = outputs['past_key_values']
    
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
    
    all_layers_hidden_states_dict = {}
    all_attention_tree_dict = {}
    all_position_ids_dict = {}

    for layer in self.intermediate_layers:
        all_layers_hidden_states_dict[layer] = None
        all_attention_tree_dict[layer] = []
        all_position_ids_dict[layer] = []

    stop_generation = False
    for i in range(depth):
        # if stop_generation:
        #     break
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        position_ids = position_ids.unsqueeze(0)
        
        # position_ids = position_ids.unsqueeze(0)
        # attention_mask = torch.cat((torch.ones(self.tree_mask.shape[0], self.tree_mask.shape[1], self.tree_mask.shape[2], len_posi-i).to('cuda'), self.tree_mask), dim=-1) 
        
        # outputs = self.soft_llamaForCausalLM_forward(input_ids=input_ids, past_key_values=outputs['past_key_values'], 
                    # position_ids=position_ids, attention_mask=attention_mask, use_cache=True)
        
        # adaptive_input_cache = outputs['past_key_values']
        # print('**')
        # with Timer('topk generate'):
        for layer, threshold in zip(self.intermediate_layers, self.intermediate_thresholds):
            if layer != self.intermediate_layers[0]:
                pre_layer = self.intermediate_layers[self.intermediate_layers.index(layer)-1]
            else:
                pre_layer = None

            padded_attention_mask = None
            padded_position_ids = None
            input_hidden_states = None
            verified_tokens_num = 0

            # Calculate input hidden states 
            # with Timer('1st scope'):
            if layer != self.intermediate_layers[0] and all_layers_hidden_states_dict[pre_layer] != None:
                # concat the hidden state with the current one
                input_hidden_states = torch.cat((all_layers_hidden_states_dict[pre_layer], 
                                                outputs.hidden_states[-2]), 
                                                dim=-2)

                all_layers_hidden_states_dict[pre_layer] = None

                # handle attention mask of previous inputs
                max_attn_len = 10 * (i+1)
                all_attention_tree_dict[pre_layer].append(self.tree_mask)
                pre_layer_attn_masks = [F.pad(x, (0, max_attn_len - x.shape[3])) for x in all_attention_tree_dict[pre_layer]]
                self.tree_mask = torch.cat(pre_layer_attn_masks, dim=2)
                all_attention_tree_dict[pre_layer] = []
                
                # print(f'pre_layer_attn_masks: {pre_layer_attn_masks[0].shape}')
                # print(f'self.tree_mask: {self.tree_mask.shape}')

                # handle position ids of previous inputs
                all_position_ids_dict[pre_layer].append(position_ids)
                position_ids = torch.cat(all_position_ids_dict[pre_layer], dim=-1)
                all_position_ids_dict[pre_layer] = []

            else:
                input_hidden_states = outputs.hidden_states[-2] if layer != self.intermediate_layers[0] else None
            
            # with Timer('2nd scope'):
            # Check if there is hidden states in the verified tokens
            if layer != self.intermediate_layers[0] and self.layers_verified_hidden_states_dict[pre_layer] is not None:
                if input_hidden_states is not None:
                    input_hidden_states = torch.cat((self.layers_verified_hidden_states_dict[pre_layer], 
                                                        input_hidden_states),
                                                        dim=-2)
                else:
                    input_hidden_states = self.layers_verified_hidden_states_dict[pre_layer]

                verified_tokens_num = self.layers_verified_hidden_states_dict[pre_layer].shape[1]
                padded_attention_mask = torch.tril(torch.ones(self.tree_mask.shape[0],
                                                                self.tree_mask.shape[1],
                                                                verified_tokens_num, 
                                                                verified_tokens_num))
                padded_attention_mask = F.pad(padded_attention_mask, (0, self.tree_mask.shape[2]))
                padded_ones = torch.ones(self.tree_mask.shape[0],
                                                    self.tree_mask.shape[1],
                                                    padded_attention_mask.shape[2],
                                                    len_posi-i-verified_tokens_num)
                padded_attention_mask = torch.cat((padded_ones, padded_attention_mask), dim=-1)
                
                padded_position_ids = torch.arange(len_posi-i-verified_tokens_num, len_posi-i).unsqueeze(0).to(position_ids.device)
                position_ids = torch.cat((padded_position_ids, position_ids), dim=-1)

                self.layers_verified_hidden_states_dict[pre_layer] = None

            # with Timer('3rd scope'):
            if layer != self.intermediate_layers[0]:
                    new_past = []
                    maximum_length = adaptive_input_cache[layer-1][0].shape[2]
                    # print(f'maximum_length: {maximum_length}')
                    for idx in range(len(adaptive_input_cache)):
                        new_past.append(
                            (
                                adaptive_input_cache[idx][0][:, :, :maximum_length, :],
                                adaptive_input_cache[idx][1][:, :, :maximum_length, :],
                            )
                        )
                    new_past = tuple(new_past)
            else:
                new_past = adaptive_input_cache

            # with Timer('8th scope:'):
            # attention_mask = torch.cat((torch.ones(self.tree_mask.shape[0], self.tree_mask.shape[1], self.tree_mask.shape[2], len_posi-i).to('cuda'), self.tree_mask), dim=-1) 
            ones_shape = (self.tree_mask.shape[0], self.tree_mask.shape[1], self.tree_mask.shape[2], len_posi - i)
            ones_tensor = torch.ones(ones_shape, device=self.tree_mask.device)
            attention_mask = torch.cat((ones_tensor, self.tree_mask), dim=-1)

            if padded_attention_mask is not None:
                attention_mask = torch.cat((padded_attention_mask.to(attention_mask.device), attention_mask), dim=-2)

            # print(f'depth: {i}')
            # print(f'layer: {layer}')
            # print(f'input_ids: {input_ids.shape}')
            # print(f'new_past 2: {new_past[0][0].shape}')
            # print(f'new_past 4: {new_past[2][0].shape}')
            # print(f'new_past 6: {new_past[4][0].shape}')
            # print(f'adaptive_input_cache 2: {adaptive_input_cache[0][0].shape}')
            # print(f'adaptive_input_cache 4: {adaptive_input_cache[2][0].shape}')
            # print(f'adaptive_input_cache 6: {adaptive_input_cache[4][0].shape}')
            # print(f'attention_mask: {attention_mask.shape}')
            # print(f'position_ids: {position_ids.shape}')
            # if input_hidden_states is not None:
            #     print(f'input_hidden_states: {input_hidden_states.shape}')
            # for l in self.layers_verified_hidden_states_dict:
            #     if self.layers_verified_hidden_states_dict[l] is not None:
            #         print(f'layers_verified_hidden_states_dict {l}: {self.layers_verified_hidden_states_dict[l].shape}')

            # print('*****************************')
            
            # with Timer('forward call:'):
            outputs = self.soft_llamaForCausalLM_forward(
                    input_ids=input_ids if layer==self.intermediate_layers[0] else None,
                    inputs_embeds=input_hidden_states if layer!=self.intermediate_layers[0] else None,
                    past_key_values=new_past,
                    return_dict=True,
                    use_cache=True,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    draft_target='draft' if layer == self.intermediate_layers[0] else 'adaptive',
                    submodel_layer=layer,
                    adaptive_layer=pre_layer,
                )
            
            # update stable kv cache if there was verified num tokens
            # with Timer('4th scope'):
            if padded_attention_mask is not None:
                tmp_stable_kv = list(self.stable_kv)
                for l in range(pre_layer, layer):
                    tmp_layer_kv = list(tmp_stable_kv[l])
                    tmp_layer_kv[0] = torch.cat((tmp_layer_kv[0],
                                                    outputs.past_key_values[l][0][:, :, len_posi-i-verified_tokens_num:len_posi-i, :]),
                                                    dim=-2)
                    tmp_layer_kv[1] = torch.cat((tmp_layer_kv[1],
                                                    outputs.past_key_values[l][1][:, :, len_posi-i-verified_tokens_num:len_posi-i, :]),
                                                    dim=-2)
                    tmp_stable_kv[l] = tuple(tmp_layer_kv)
                self.stable_kv = tuple(tmp_stable_kv)

                # concat hidden states
                if layer != self.intermediate_layers[-1]:
                    if self.layers_verified_hidden_states_dict[layer] != None:
                        self.layers_verified_hidden_states_dict[layer] = torch.cat((self.layers_verified_hidden_states_dict[layer], outputs.hidden_states[-2][:, :verified_tokens_num, :]), dim=-2)
                    else:
                        self.layers_verified_hidden_states_dict[layer] = outputs.hidden_states[-2][:, :verified_tokens_num, :]
                
                # delete the verified tokens hidden states for the next round
                tmp_output_hs = list(outputs.hidden_states)
                tmp_output_hs[-2] = tmp_output_hs[-2][:, verified_tokens_num:, :]
                outputs.hidden_states = tuple(tmp_output_hs)
                position_ids = position_ids[:, verified_tokens_num:]
            
            # return to the original adaptive input cache before the intermidate layer forward call

            # with Timer('5th scope'):
            if layer != self.intermediate_layers[0]:
                new_output_past = list(outputs['past_key_values'])
                new_output_past[:pre_layer] = list(adaptive_input_cache)[:pre_layer]
                adaptive_input_cache = tuple(new_output_past)
            else:
                adaptive_input_cache = outputs['past_key_values']

            next_token_logits = outputs.logits[0][-top_k:, :]

            # pre-process distribution
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_tokens_probs = next_token_logits.softmax(dim=-1)


            # with Timer('6th scope:'):
            # if max(torch.max(next_tokens_probs, dim=-1).values) > threshold or layer == self.intermediate_layers[-1]:
            # print(f'layer: {layer}, probs: {torch.max(next_tokens_probs, dim=-1).values}')
            # print(next_token_logits.shape)
            # print(next_tokens_probs.shape)
            # print(f'mean prob: {torch.mean(torch.max(next_tokens_probs, dim=-1).values)}')
            if torch.max(next_tokens_probs) > threshold or layer == self.intermediate_layers[-1]:
            # if torch.mean(torch.max(next_tokens_probs, dim=-1).values) > threshold or layer == self.intermediate_layers[-1]:
            # if torch.mean(torch.max(next_tokens_probs, dim=-1).values) > threshold:
                # with Timer('7th scope:'):
                if layer != self.intermediate_layers[-1]:
                    if all_layers_hidden_states_dict[layer] != None:
                        all_layers_hidden_states_dict[layer] = torch.cat((all_layers_hidden_states_dict[layer], outputs.hidden_states[-2]), dim=-2)
                    else:
                        all_layers_hidden_states_dict[layer] = outputs.hidden_states[-2]
                    all_attention_tree_dict[layer].append(self.tree_mask)
                    all_position_ids_dict[layer].append(position_ids)
                break
            # elif layer == self.intermediate_layers[-1]:
            #     stop_generation = True
            #     break
            else:
                continue
        
        len_posi += 1

        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k ** 2 * bias2 + bias1
        parents = (topk_cs_index + bias)
        parents_list.append(parents)

        last_headout = outputs['logits'][0][-10:].clone()
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
        tree_mask = torch.cat((tree_mask[:, :, out_ids.to(tree_mask.device)], self.tree_mask_init), dim=3)

        # ss_token.append(topk_index)
        # ss_prob.append(topk_prob)
        # ss_op.append(op)
        # topk_index = topk_index.clone().view(-1)
        # select_index=topk_index[self.tree_buffer['tree_indices'][i]]
        # input_ids=select_index[None,:]
        # self.tree_mask = self.tree_buffer['attn_mask'][i]
        # position_ids = len_posi+self.tree_buffer["position_ids"][i]
        # position_ids = position_ids.unsqueeze(0)
        # attention_mask = torch.cat((torch.ones(self.tree_mask.shape[0], self.tree_mask.shape[1], self.tree_mask.shape[2], len_posi-i).to('cuda'), self.tree_mask), dim=-1) 
        # outputs = self(input_ids=input_ids, past_key_values=outputs['past_key_values'], 
        #                 position_ids=position_ids, attention_mask=attention_mask, use_cache=True)
        # len_posi += 1
        
        # last_headout = outputs['logits'][0].clone()


    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token.to(draft_tokens.device), draft_tokens), dim=0)

    parents_list = [x.to(top_scores_index.device) for x in parents_list]
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

    # with Timer("mask1"):
    #     tree_mask0 = [[False for _ in range(total_tokens + 1)] for _ in range(total_tokens + 1)]
    #     tree_mask0[0][0] = True
    #     for i in range(total_tokens):
    #         #tree_mask0[i + 1][0]=True
    #         tree_mask0[i + 1][i + 1] = True
    #         p=mask_index_list[i]
    #         tree_mask0[i + 1][p] = True
    #         while p:
    #             p=mask_index_list[p-1]
    #             tree_mask0[i + 1][p] = True
    #     tree_mask0 = torch.tensor(tree_mask0, dtype=torch.bool)
    #
    # print(tree_mask0.equal(tree_mask))
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