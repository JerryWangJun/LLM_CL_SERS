
REPO_ROOT_DIR=/LLM_CL_SERS
SRC_DIR=$REPO_ROOT_DIR/src
MODEL_DIR=/model/llama2-7b
CATE=${1:-sum}
CUDA=${2:-0}

CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/LLAMA2-7B/lora/${CATE}/bs32x1x1-3ep-bf16
CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
    --stage ssr \
    --model_name_or_path $MODEL_DIR \
    --do_train True \
    --overwrite_cache True \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir $REPO_ROOT_DIR/data/ \
    --dataset ni_c012_${CATE}_train \
    --max_source_length 1024 \
    --max_target_length 512 \
    --learning_rate 2e-04 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_strategy no \
    --warmup_steps 0 \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --resume_lora_training True \
    --output_dir $CKPT_DIR \
    --plot_loss True \
    --bf16 True
    
CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
    --stage sftrp \
    --model_name_or_path $MODEL_DIR \
    --checkpoint_dir $CKPT_DIR \
    --overwrite_cache True \
    --predict_with_generate True \
    --finetuning_type lora \
    --template llama2 \
    --dataset_dir $REPO_ROOT_DIR/data/ \
    --dataset ni_c012_sum_eval \
    --max_source_length 1024 \
    --max_target_length 512 \
    --max_samples 100000 \
    --per_device_eval_batch_size 1 \
    --output_dir $CKPT_DIR/ni_c012_sum_eval \
    --do_predict True \
    --do_sample False \
    --bf16 True   
