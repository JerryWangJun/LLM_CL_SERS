import numpy as np
import jsonlines
import random
import os
from sklearn.cluster import KMeans

sample_memory = 200
# n_cluster = 20

def largest_prefix(stra: str, strb: str):
    for i, c in enumerate(stra):
        if i == len(strb):
            return len(strb)
        if c != strb[i]:
            return i
    return len(stra)


for cur_cate in ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]:
    gen_data_dir = '/data/ni-cus0.12/genearated-icl-naive-parsed-filtered/llama2-7b/ori'
    json_suffix = '.train.smp001.2shot.smp3.rp1.2.json'
    pseudo_json_path = f"{gen_data_dir}/{cur_cate}{json_suffix}"
    if sample_memory == 200:
        tgt_json_dir = f"/data/ni-cus0.12/genearated-icl-naive-random-self/llama2-7b/ori"
    else:
        tgt_json_dir = f"/data/ni-cus0.12/genearated-icl-naive-random-self-smp{sample_memory}/llama2-7b/ori"
    tgt_json_path = f"{tgt_json_dir}/{cur_cate}{json_suffix}"

    if not os.path.exists(tgt_json_dir):
        os.makedirs(tgt_json_dir)

    np.random.seed(0)
    random.seed(0)


    with jsonlines.open(pseudo_json_path) as f:
        data = [l for l in f]


    sample_id_list = np.random.choice(
            len(data), sample_memory, replace=False
        )


    with jsonlines.open(tgt_json_path, 'w') as f:
        for id in sample_id_list:
            f.write(data[id])
