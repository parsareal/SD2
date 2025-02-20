"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from fastchat.utils import str_to_torch_dtype
from model.sps_tree_adaptive.modeling import *

from evaluation.eval import run_eval, reorg_answer_file
from model.sps_tree_adaptive.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from model.sps_tree_adaptive.kv_cache import initialize_past_key_values
from model.sps_tree_adaptive.utils import *
from model.sps_tree_adaptive.choices import *


from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch

def baseline_forward(inputs, model, tokenizer, max_new_tokens, tree_choices, temperature=0.0, do_sample=False, logits_processor=None, ):
    input_ids = inputs.input_ids
    input_len = input_ids.shape[1]
    past_key_values = None

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

    # with Timer('target time'):
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    new_token = 0

    for idx in range(max_new_tokens):
        if logits_processor is not None:
            logits = outputs.logits[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            input_id = torch.multinomial(probabilities, 1)
        else:
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
        # with Timer('target time'):
        outputs = model(input_id, use_cache=True, past_key_values=past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        if input_ids.shape[1] > 1960:
            break

    new_token = len(input_ids[0][input_len:])
    step = new_token
    accept_length_list = [1] * new_token
    return input_ids, new_token, step, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
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
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    args.tree_choices = eval(args.tree_choices)
    print(f"Output to {answer_file}")

    model = KVLlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)


    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False
    
    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
        logits_processor=logits_processor,
        tree_choices=args.tree_choices,
    )

    reorg_answer_file(answer_file)