# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM, LlamaSdpaAttention
from model.sd2.decoding import assisted_decoding, adaptive_greedy_search, get_candidates, adaptive_generate, adjust_submodels_hiddenstates
from model.sd2.modeling import soft_llamaModel_forward, soft_llamaForCausalLM_forward, attention_forward
from transformers.generation.candidate_generator import AssistedCandidateGenerator
import torch

def sd2_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, temperature=0.0, drafter=None):
    input_ids = inputs.input_ids
    model.generation_config.max_new_tokens = max_new_tokens
    output_ids, idx, accept_length_list = model.generate(
        **inputs, generation_config=model.generation_config, assistant_model=drafter, do_sample=do_sample, temperature=temperature)
    new_token = len(output_ids[0][len(input_ids[0]):])
    return output_ids, new_token, idx+1, accept_length_list


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
        "--submodels",
        type=str,
        default=None,
        help="Determine which submodels are picked from the draft model.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Determine which thresholds are used during adaptive speculative decoding.",
    )
    args = parser.parse_args()

    submodels = [int(submodel) for submodel in args.submodels.split(',')]
    thresholds = [float(threshold) for threshold in args.thresholds.split(',')]

    GenerationMixin.assisted_decoding = assisted_decoding
    GenerationMixin.adaptive_greedy_search = adaptive_greedy_search
    GenerationMixin.adaptive_generate = adaptive_generate
    GenerationMixin.intermediate_layers = submodels
    GenerationMixin.intermediate_thresholds = thresholds
    GenerationMixin.adjust_submodels_hiddenstates = adjust_submodels_hiddenstates
    
    LlamaModel.soft_llamaModel_forward = soft_llamaModel_forward
    LlamaForCausalLM.soft_llamaForCausalLM_forward = soft_llamaForCausalLM_forward
    LlamaSdpaAttention.forward = attention_forward
    AssistedCandidateGenerator.get_candidates = get_candidates

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    model = AutoModelForCausalLM.from_pretrained(
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model.eval()
    drafter.eval()

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=sd2_forward,
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
        temperature=args.temperature,
        do_sample=do_sample,
    )

    reorg_answer_file(answer_file)