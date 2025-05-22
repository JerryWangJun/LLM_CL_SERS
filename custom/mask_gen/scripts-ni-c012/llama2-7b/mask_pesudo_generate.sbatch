
CATE_LIST=("qa" "qg" "sa" "sum" "trans" "dsg" "expl" "para" "pe" "pos")
for CUR_CATE in ${CATE_LIST[@]}; 
do
    echo $CUR_CATE
    CUDA_VISIBLE_DEVICES=0 python3 mask_pesudo_generate.py \
                    --generate_model_path /model/llama2-7b-hf/ \
                    --input_path /data/ni-cus0.12/split/$CUR_CATE.train.smp001.json \
                    --output_path /data/ni-cus0.12/generated-masked-pesudo/llama2-7b/$CUR_CATE.train.smp001.retry3.pesudo.json \
                    --retry 1
                    
done