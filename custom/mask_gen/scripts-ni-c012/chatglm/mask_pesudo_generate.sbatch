 CATE_LIST=("qa" "qg" "sa" "sum" "trans" "dsg" "expl" "para" "pe" "pos") 
for CUR_CATE in ${CATE_LIST[@]}; 
do
    echo $CUR_CATE
    CUDA_VISIBLE_DEVICES=0 python3 mask_pesudo_generate.py \
                    --generate_model_path  \
                    --input_path  \
                    --output_path \
                    --retry 3
                    
done