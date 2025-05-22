REPO_ROOT_DIR=/LLM_CL_SERS/
SRC_DIR=$REPO_ROOT_DIR/src
MODEL_DIR=/model/chatglm
CUDA=${1:-0}
ORI_CKPT_DIR=""


CATE_LIST=("qa" "qg" "sa" "sum" "trans" "dsg" "expl" "para" "pe" "pos")
CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/chatglm/lora/${CATE_LIST[0]}/bs32x1x1-3ep-bf16

DATASETS=""

for CUR_CATE in ${CATE_LIST[@]}; 
do
    # echo CATE_LIST[0]: ${CATE_LIST[0]}
    if [[ ${CATE_LIST[0]} != $CUR_CATE ]]
    then
        ORI_CKPT_DIR=$CKPT_DIR
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/chatglm/lora/1shot-mask_se0.1_alpha0.6_lambda0.6-0.15_wth0.08_wsreg_cl2_queue_${CUR_CATE}/bs32x1x1-3ep-bf16
        echo $CKPT_DIR

        if [ -d $CKPT_DIR ]; then
            echo File $CKPT_DIR exists.
        else
            DATASETS="ni_c012_${CUR_CATE}_train"
            for PREV_CATE in ${CATE_LIST[@]};
            do
                if [[ $PREV_CATE == $CUR_CATE ]]
                then
                    break
                else
                    DATASETS="$DATASETS,ni_c012_1shot-mask_km20_se0.1_alpha0.6_cl2_queue_chatglm_${PREV_CATE}"
                fi
            done
            
            echo dataset: $DATASETS
            echo ORI_CKPT_DIR: $ORI_CKPT_DIR, CKPT_DIR: $CKPT_DIR
            CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
                --stage sft \
                --model_name_or_path $MODEL_DIR \
                --checkpoint_dir $ORI_CKPT_DIR \
                --do_train True \
                --overwrite_cache True \
                --finetuning_type lora \
                --template chatglm2 \
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
                --bf16 True  \
                --lambda_max 0.6 \
                --lambda_min 0.15 \
                --w_threshold 0.08 
            fi
    fi

    OUTPUT_PATH=$REPO_ROOT_DIR"/data/ni-cus0.12/genearated-masked-pesudo-kmeans20-self/chatglm/cl2_queue/se0.1_alpha0.6/$CUR_CATE.train.smp001.1shot-mask.retry3.json"
    if [ ! -f $OUTPUT_PATH ]; then
        echo File $OUTPUT_PATH does not exist.

        CUDA_VISIBLE_DEVICES=$CUDA python3 $REPO_ROOT_DIR/src/selearning_copy.py \
                --stage sftrp \
                --model_name_or_path $MODEL_DIR \
                --checkpoint_dir $CKPT_DIR \
                --overwrite_cache True \
                --predict_with_generate True \
                --finetuning_type lora \
                --output_dir $REPO_ROOT_DIR/data/ni-cus0.12/genearated-masked-pesudo-kmeans20-self/chatglm/cl2_queue/se0.1_alpha0.6/ \
                --output_file ${CUR_CATE}.train.smp001.1shot-mask.retry3.json \
                --predict_with_generate True \
                --dataset_dir $REPO_ROOT_DIR/data/ \
                --dataset ni_c012_1shot-mask_km20_ori_chatglm_${CUR_CATE} \
                --do_sample False \
                --template chatglm2 \
                --max_source_length 1024 \
                --max_target_length 512 \
                --max_samples 100000 \
                --per_device_eval_batch_size 1\
                --bf16 True \
                --do_predict True \
                --do_sample False \
                --se_ratio 10 \
                --se_alpha 0.6
    else
        echo File $OUTPUT_PATH exists.
    fi
    
done

for CUR_CATE in ${CATE_LIST[@]}; do
    echo "CUR_CATE: $CUR_CATE"

    cur_idx=-1
    for i in "${!CATE_LIST[@]}"; do
        if [[ "${CATE_LIST[$i]}" == "$CUR_CATE" ]]; then
            cur_idx=$i
            break
        fi
    done
    
    echo "Current evaluation index range up to: $cur_idx"

    if [[ ${CATE_LIST[0]} != "$CUR_CATE" ]]; then
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/chatglm/lora/1shot-mask_se0.1_alpha0.6_lambda0.6-0.15_wth0.08_wsreg_cl2_queue_${CUR_CATE}/bs32x1x1-3ep-bf16
    else
        CKPT_DIR=$REPO_ROOT_DIR/saves/ni-c012/chatglm/lora/${CUR_CATE}/bs32x1x1-3ep-bf16
    fi

    DATASETS="ni_c012_${CUR_CATE}_train"

    for (( idx=0; idx<=cur_idx; idx++ )); do
        CUR_CATE_Y=${CATE_LIST[idx]}
        echo "CUR_CATE_Y: $CUR_CATE_Y"

        if [ -d "$CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval" ]; then
            echo "File $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval exists."
            continue
        else
            echo "$CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval does not exist."
        fi

        CUDA_VISIBLE_DEVICES=$CUDA python3 $SRC_DIR/train_bash.py \
            --stage sftrp \
            --model_name_or_path $MODEL_DIR \
            --checkpoint_dir $CKPT_DIR \
            --overwrite_cache True \
            --predict_with_generate True \
            --finetuning_type lora \
            --template chatglm2 \
            --dataset_dir $REPO_ROOT_DIR/data/ \
            --dataset ni_c012_${CUR_CATE_Y}_eval \
            --max_source_length 1024 \
            --max_target_length 512 \
            --max_samples 100000 \
            --per_device_eval_batch_size 1 \
            --output_dir $CKPT_DIR/ni_c012_${CUR_CATE_Y}_eval \
            --do_predict True \
            --do_sample False \
            --bf16 True
    done
done