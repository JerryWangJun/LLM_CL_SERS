

REPO_ROOT_DIR=/LLM_CL_SERS
SRC_DIR=$REPO_ROOT_DIR/custom/icl_gen
MODEL_DIR=/model/chatglm/
CKPT_DIR=$MODEL_DIR
CVD=${1:-0}
GEN_CATE_LIST=${2:-"qa qg sa sum trans dsg expl para pe pos"}
N_SHOTS=${3:-2}
RETRIES=${4:-3}
RP=${5:-1.2}

echo do_sample_retries: $RETRIES

for CUR_CATE in $GEN_CATE_LIST;
do
   echo $CUR_CATE
   CUDA_VISIBLE_DEVICES=${CVD} python3 complete_param_nic010_cate.py \
         --model_name_or_path $CKPT_DIR \
         --input_path "${REPO_ROOT_DIR}/data/ni-cus0.12/split/$CUR_CATE.train.smp001.json" \
         --output_path "${REPO_ROOT_DIR}/data/ni-cus0.12/genearated-icl-naive/chatglm/ori/$CUR_CATE.train.smp001.${N_SHOTS}shot.smp${RETRIES}.rp${RP}.json" \
         --do_sample True \
         --do_sample_retries $RETRIES \
         --top_p 0.6 \
         --temperature 0.9 \
         --max_length 2048 \
         --num_beams 1 \
         --n_shots ${N_SHOTS} \
         --template vanilla \
         --resume True
done
