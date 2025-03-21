CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_xformers.py \
    --model_name_or_path YOUR_PATH/Vicuna-7b-v1.5/ \
    --data_path sharegpt_clean_lang_split.json \
    --fp16 True \
    --output_dir YOUR_PATH/vicuna-13b-soft-6,9,12 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to 'none' \
    --lazy_preprocess True \
    --soft True \
    --submodels 6,9,12 \

