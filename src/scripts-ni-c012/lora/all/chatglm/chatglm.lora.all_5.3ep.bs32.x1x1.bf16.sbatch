

REPO_ROOT_DIR=/SERS
SRC_DIR=$REPO_ROOT_DIR/src
MODEL_DIR=/model/chatglm/
# CATE=${1:-}
CUDA=${1:-0}
# PORT=${2:-9901}
CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/chatglm/lora/all_5/bs32x1x1-3ep-bf16

CATE_LIST=("qa" "qg" "sa" "sum" "trans")

DATASETS=""

for CUR_CATE in ${CATE_LIST[@]}; 
do
        if [[ $DATASETS == "" ]]
        then
            DATASETS="ni_c012_${CUR_CATE}_train"
        else
            DATASETS="$DATASETS,ni_c012_${CUR_CATE}_train"
        fi
done

echo dataset: $DATASETS

CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
    --stage ssr \
    --model_name_or_path $MODEL_DIR \
    --do_train True \
    --overwrite_cache True \
    --finetuning_type lora \
    --template vanilla \
    --dataset_dir $REPO_ROOT_DIR/data/ \
    --dataset $DATASETS \
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
    --lora_target query_key_value \
    --resume_lora_training True \
    --output_dir $CKPT_DIR \
    --plot_loss True \
    --bf16 True


for CUR_CATE in ${CATE_LIST[@]};
do
    echo $CUR_CATE
    CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
        --stage sftrp \
        --model_name_or_path $MODEL_DIR \
        --checkpoint_dir $CKPT_DIR \
        --overwrite_cache True \
        --predict_with_generate True \
        --finetuning_type lora \
        --template vanilla \
        --dataset_dir $REPO_ROOT_DIR/data/ \
        --dataset ni_c012_${CUR_CATE}_eval \
        --max_source_length 1024 \
        --max_target_length 512 \
        --max_samples 100000 \
        --per_device_eval_batch_size 1 \
        --output_dir $CKPT_DIR/ni_c012_${CUR_CATE}_eval \
        --do_predict True \
        --do_sample False \
        --bf16 True      
done
