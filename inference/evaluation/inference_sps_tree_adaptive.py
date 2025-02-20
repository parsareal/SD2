"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype
from model.sps_tree_adaptive.utils import *
from model.sps_tree_adaptive.choices import *
from model.sps_tree_adaptive.modeling import *
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from model.sps.decoding import assisted_decoding
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from model.sps_tree_adaptive.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from model.sps_tree_adaptive.kv_cache import initialize_past_key_values
import torch

def sps_tree_adaptive_forward(inputs, model, tokenizer, max_new_tokens, tree_choices=None, logits_processor=None, temperature=0.0, drafter=None, max_steps=512):
    input_ids = inputs.input_ids
    input_ids = input_ids.clone()
    padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)

    drafter.reset_kv()
    # drafter.set_params()
    drafter.init_tree()
    accept_length_list = []
    model.generation_config.max_new_tokens = max_new_tokens

    ###########################
    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device #base_model.model.layers[-1].self_attn.q_proj.weight.device
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.lm_head.weight.device) #base_model.lm_head.weight.device)
        
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices
    
    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data
    
    reset_tree_mode(model)
    ###########################

    input_len = input_ids.shape[1]
    cur_length = input_len

    # past_key_values = None
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    if logits_processor is not None:
        prob = logits_processor(None, outputs.logits[:, -1])
        prob = torch.nn.functional.softmax(prob, dim=1)
        sample_token = torch.multinomial(prob, 1)
        sample_token = sample_token
    else:
        sample_token = torch.argmax(outputs.logits[:, -1])
        sample_token = sample_token[None, None]

    # past_key_values = outputs.past_key_values

    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = initialize_tree(
        torch.cat([input_ids, sample_token], dim=-1), drafter, logits_processor
    )
    
    new_token = 0
   

    for idx in range(max_steps):
        model.model.tree_mask = tree_mask

        # with Timer('tree_decoding'):

        logits, outputs = tree_decoding(
            model,
            draft_tokens,
            past_key_values,
            tree_position_ids,
            input_ids,
            tree_mask,
            retrieve_indices,
        )
        
        draft_tokens=torch.cat((draft_tokens,padding),dim=1)
        candidates=draft_tokens[0,retrieve_indices]
        
        # with Timer('evaluate_posterior'):
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, logits_processor
        )
        
        # input_ids, tree_logits, new_token, sample_token, past_key_values = update_inference_inputs(
        #     input_ids,
        #     candidates,
        #     best_candidate,
        #     accept_length,
        #     tree_buffers["retrieve_indices"],
        #     logits_processor,
        #     logits,
        #     tree_logits,
        #     new_token,
        #     past_key_values,
        #     drafter,
        #     sample_p
        # )

        # with Timer('update_inference_inputs'):
        # input_ids, draft_tokens, retrieve_indices,tree_mask, tree_position_ids, new_token, sample_token, past_key_values = update_inference_inputs(
        #         input_ids,
        #         candidates,
        #         best_candidate,
        #         accept_length,
        #         retrieve_indices,
        #         logits_processor,
        #         new_token,
        #         past_key_values,
        #         # current_length_data,
        #         drafter,
        #         sample_p
        #     )
        input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            retrieve_indices,
            logits_processor,
            new_token,
            past_key_values_data,
            current_length_data,
            drafter,
            sample_p
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        # input_ids = torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1)
        # print(f'text: {tokenizer.decode([x.item() for x in input_ids[0]])}')
        # from ipdb import set_trace; set_trace()
        # print(tokenizer.decode([x.item() for x in input_ids[0]]))

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            for i, id in enumerate(input_ids[0, input_len:]):
                if id == tokenizer.eos_token_id:
                    eos_token_ids_index = i
            invalid_len = len(input_ids[0, input_len:]) - eos_token_ids_index - 1
            if invalid_len > 0:
                accept_length_list[-1] -= invalid_len
                new_token -= invalid_len
            break
        if new_token > max_new_tokens:
            break
        if input_ids.shape[1] > 1960:
            break
    return input_ids, new_token, idx+1, accept_length_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drafter-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    args = parser.parse_args()

    GenerationMixin.assisted_decoding = assisted_decoding
    LlamaForCausalLM.init_tree = init_tree
    LlamaForCausalLM.topk_generate = topk_generate
    LlamaForCausalLM.reset_kv = reset_kv
    LlamaForCausalLM.customized_sample = customized_sample
    LlamaForCausalLM.set_params = set_params
    LlamaForCausalLM.reset = reset


    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"
    
    args.tree_choices = eval(args.tree_choices)

    print(f"Output to {answer_file}")

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=str_to_torch_dtype(args.dtype),
    #     low_cpu_mem_usage=True,
    #     device_map="auto"
    # )

    model = KVLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    drafter = AutoModelForCausalLM.from_pretrained(
        args.drafter_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    drafter.set_params(total_tokens=args.total_token, depth=args.depth, top_k=args.top_k)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model.eval()
    drafter.eval()

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=sps_tree_adaptive_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        drafter=drafter,
        tree_choices=args.tree_choices,
        temperature=args.temperature,
        logits_processor=logits_processor,
        max_steps=args.max_steps,
    )

    reorg_answer_file(answer_file)