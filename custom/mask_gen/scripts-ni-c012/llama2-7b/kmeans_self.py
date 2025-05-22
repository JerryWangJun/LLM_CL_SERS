import numpy as np
import jsonlines
import random
import os
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample_memory = 400
n_cluster = 100

# largest_prefix 函数用于找到两个字符串之间的最大公共前缀长度
def largest_prefix(stra: str, strb: str):
    for i, c in enumerate(stra):
        if i == len(strb):
            return len(strb)
        if c != strb[i]:
            return i
    return len(stra)

#  "dsg", "expl", "para", "pe", "pos"
for cur_cate in ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]:
    gen_data_dir = '/data/ni-cus0.12/generated-masked-pesudo-filted/llama2-7b/ori-van'
    # data_path = f"{data_dir}/{cur_cate}.train.json"
    # emb_path = f"{data_dir}/{cur_cate}.train.npy"
    json_suffix = '.train.smp005.1shot-mask.retry3.json'
    npy_suffix = '.train.smp005.1shot-mask.retry3.npy'
    # json_suffix = '.train.smp005.retry30.pesudo.json'
    # npy_suffix = '.train.smp005.retry30.pesudo.npy'
    pseudo_json_path = f"{gen_data_dir}/{cur_cate}{json_suffix}"
    pseudo_emb_path = f"{gen_data_dir}/{cur_cate}{npy_suffix}"
    tgt_suffix = '.train.smp001.1shot-mask.retry3.json'
    
    if sample_memory == 200:
        tgt_json_dir = f"/data/ni-cus0.12/genearated-masked-pesudo-kmeans{n_cluster}-self/llama2-7b/ori-van"
    else:
        tgt_json_dir = f"/data/ni-cus0.12/genearated-masked-pesudo-kmeans{n_cluster}-self-smp{sample_memory}/llama2-7b/ori-van"
    tgt_json_path = f"{tgt_json_dir}/{cur_cate}{tgt_suffix}"

    # print(data_path)
    # print(emb_path)
    # print(json_suffix)
    print(pseudo_json_path)
    print(pseudo_emb_path)
    print(tgt_json_path)

    if not os.path.exists(tgt_json_dir):
        os.makedirs(tgt_json_dir)

    # emb_list = np.load(emb_path)
    # n_emb, n_dim = emb_list.shape
    # emb_list = emb_list / np.linalg.norm(emb_list, axis=-1).repeat(n_dim).reshape(
    #     n_emb, n_dim
    # )

    pseudo_emb_list = np.load(pseudo_emb_path)
    n_pseudo_emb, n_dim = pseudo_emb_list.shape
    pseudo_emb_list = pseudo_emb_list / np.linalg.norm(pseudo_emb_list, axis=-1).repeat(n_dim).reshape(
        n_pseudo_emb, n_dim
    )

    np.random.seed(0)
    random.seed(0)

    kmeans = KMeans(n_clusters=n_cluster, n_init='auto')
    # labels = kmeans.fit_predict(emb_list)
    
    pseudo_labels = kmeans.fit_predict(pseudo_emb_list)

    centric_distances = np.array([np.linalg.norm(e-kmeans.cluster_centers_[pseudo_labels[i]]) for i, e in enumerate(pseudo_emb_list)])

    n_cluster_instances = [0]* n_cluster
    uniq_idx, uniq_cnt = np.unique(pseudo_labels, return_counts=True)
    for i, idx in enumerate(uniq_idx):
        n_cluster_instances[idx] = uniq_cnt[i]
    # print(np.unique(pseudo_labels, return_counts=True)[0])
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
