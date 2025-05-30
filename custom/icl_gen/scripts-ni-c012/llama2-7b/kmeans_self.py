import numpy as np
import jsonlines
import random
import os
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample_memory = 200
n_cluster = 20

def largest_prefix(stra: str, strb: str):
    for i, c in enumerate(stra):
        if i == len(strb):
            return len(strb)
        if c != strb[i]:
            return i
    return len(stra)


for cur_cate in ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]:
    gen_data_dir = '/data/ni-cus0.12/genearated-icl-naive-parsed-filted/T5-large'
    json_suffix = '.train.smp001.2shot.smp3.rp1.2.json'
    npy_suffix = '.train.smp001.2shot.smp3.rp1.2.npy'
    pseudo_json_path = f"{gen_data_dir}/{cur_cate}{json_suffix}"
    pseudo_emb_path = f"{gen_data_dir}/{cur_cate}{npy_suffix}"
    tgt_suffix = '.train.smp001.2shot.smp3.rp1.2.json'
    
    if sample_memory == 200:
        tgt_json_dir = f"/data/ni-cus0.12/genearated-masked-pesudo-kmeans{n_cluster}-self/T5-large/ori-van"
    else:
        tgt_json_dir = f"/data/ni-cus0.12/genearated-masked-pesudo-kmeans{n_cluster}-self-smp{sample_memory}/T5-large/ori-van"
    tgt_json_path = f"{tgt_json_dir}/{cur_cate}{tgt_suffix}"

    print(pseudo_json_path)
    print(pseudo_emb_path)
    print(tgt_json_path)

    if not os.path.exists(tgt_json_dir):
        os.makedirs(tgt_json_dir)

    pseudo_emb_list = np.load(pseudo_emb_path)
    n_pseudo_emb, n_dim = pseudo_emb_list.shape
    pseudo_emb_list = pseudo_emb_list / np.linalg.norm(pseudo_emb_list, axis=-1).repeat(n_dim).reshape(
        n_pseudo_emb, n_dim
    )

    np.random.seed(0)
    random.seed(0)

    kmeans = KMeans(n_clusters=n_cluster, n_init='auto')
    
    pseudo_labels = kmeans.fit_predict(pseudo_emb_list)

    centric_distances = np.array([np.linalg.norm(e-kmeans.cluster_centers_[pseudo_labels[i]]) for i, e in enumerate(pseudo_emb_list)])

    n_cluster_instances = [0]* n_cluster
    uniq_idx, uniq_cnt = np.unique(pseudo_labels, return_counts=True)
    for i, idx in enumerate(uniq_idx):
        n_cluster_instances[idx] = uniq_cnt[i]
    print(n_cluster_instances)
    clu_sample_num = [round(sample_memory*n/n_pseudo_emb) for n in n_cluster_instances]
    print(len(clu_sample_num), clu_sample_num)

    with jsonlines.open(pseudo_json_path) as f:
        data = [l for l in f]

    sampled_data = []

    for clu_idx in range(n_cluster):
        cur_clu_idx_list = np.where(pseudo_labels==clu_idx)[0]
        cur_clu_dis_list = centric_distances[cur_clu_idx_list]
        easys = np.argsort(cur_clu_dis_list)[:clu_sample_num[clu_idx]]

        for samp_idx in easys:
            sampled_data.append(data[cur_clu_idx_list[samp_idx]])

    print("len(sampled_data):", len(sampled_data))
    with jsonlines.open(tgt_json_path, 'w') as f:
        f.write_all(sampled_data)
