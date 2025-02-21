# Vicuna_PATH=/your_own_path/Vicuna-7b-v1.5
Vicuna_PATH=your_own_path/Vicuna-7b-v1.5
# Drafter_PATH=/your_own_path/vicuna-7b-12layers-soft-6,9,12/
Drafter_PATH=your_own_path/vicuna-7b-12layers-soft-6,9,12/
Submodels=6,9,12
Submodels=6,9,12
Thresholds=0.5,0.5,0
Total_Token=60
Depth=4
TopK=10


datastore_PATH=./model/rest/datastore/datastore_chat_large.idx
MODEL_NAME=Vicuna-7b-v1.5
TEMP=0.0 # 1.0
GPU_DEVICES=0,1 # 0,1,2,3,4,5,6,7


bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-vicuna7b-soft-12layers-6,9,12-epoch3-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sps_tree_adaptive --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sps-soft-12layers-layer9-${torch_dtype}-temp-${TEMP}-treeAttnAdaptive-depth${Depth}-totalToken${Total_Token}-topk${TopK} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --total-token $Total_Token --top-k $TopK --depth $Depth
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sd2 --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sd2-vicuna7b-soft-12layers-6,9,12-epoch3-${torch_dtype}-temp-${TEMP}-thr-${Thresholds} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --submodels $Submodels --thresholds $Thresholds
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_sd2_tree_adaptive --model-path $Vicuna_PATH --drafter-path $Drafter_PATH --model-id ${MODEL_NAME}-sd2AdaptAT-vicuna7b-soft-12layers-${Submodels}-epoch3-${torch_dtype}-temp-${TEMP}-thr-${Thresholds}-depth${Depth}-totalToken${Total_Token}-topk${TopK} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype --submodels $Submodels --thresholds $Thresholds --total-token $Total_Token --top-k $TopK --depth $Depth

