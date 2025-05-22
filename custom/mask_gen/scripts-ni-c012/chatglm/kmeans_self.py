import numpy as np
import jsonlines
import random
import os
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sample_memory = 200
n_cluster = 20
samples_per_cluster = sample_memory // n_cluster 

for cur_cate in ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]:
    gen_data_dir = ''
    json_suffix = '.train.smp001.1shot-mask.retry3.json'
    npy_suffix = '.train.smp001.1shot-mask.retry3.npy'
    
    pseudo_json_path = f"{gen_data_dir}/{cur_cate}{json_suffix}"
    pseudo_emb_path = f"{gen_data_dir}/{cur_cate}{npy_suffix}"
    
    if sample_memory == 200:
        tgt_json_dir = f""
    else:
        tgt_json_dir = f""

    tgt_json_path = f"{tgt_json_dir}/{cur_cate}{json_suffix}"

    print(pseudo_json_path)
    print(pseudo_emb_path)
    print(tgt_json_path)

    if not os.path.exists(tgt_json_dir):
        os.makedirs(tgt_json_dir)

    # Read pseudo sample embeddings
    pseudo_emb_list = np.load(pseudo_emb_path)
    n_pseudo_emb, n_dim = pseudo_emb_list.shape
    pseudo_emb_list = pseudo_emb_list / np.linalg.norm(pseudo_emb_list, axis=-1).repeat(n_dim).reshape(n_pseudo_emb, n_dim)

    np.random.seed(0)
    random.seed(0)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_cluster, n_init='auto')
    pseudo_labels = kmeans.fit_predict(pseudo_emb_list)

    # load json data
    with jsonlines.open(pseudo_json_path) as f:
        data = [l for l in f]

    sampled_data = []

    for clu_idx in range(n_cluster):
        cur_clu_idx_list = np.where(pseudo_labels == clu_idx)[0]
        num_available = len(cur_clu_idx_list)
        
        replace = num_available < samples_per_cluster
        selected_indices = np.random.choice(
            cur_clu_idx_list, 
            min(samples_per_cluster, num_available), 
            replace=replace
        )
        
        for samp_idx in selected_indices:
            sampled_data.append(data[samp_idx])

    if len(sampled_data) > sample_memory:
        sampled_data = random.sample(sampled_data, sample_memory)
    elif len(sampled_data) < sample_memory:
        remaining = sample_memory - len(sampled_data)
        extra_samples = random.sample(data, remaining)
        sampled_data.extend(extra_samples)
    
    print(f"Final sampled count: {len(sampled_data)} (target: {sample_memory})")
    
    with jsonlines.open(tgt_json_path, 'w') as f:
        f.write_all(sampled_data)