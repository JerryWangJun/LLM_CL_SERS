import os
import torch
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = "/model/sup-simcse-roberta-base/" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

src_dir = "/data/ni-cus0.12/generated-masked-pesudo-filted/llama2-7b/ori-van/"

bsz = 32

for task in ["qa", "qg", "sa", "sum", "trans"]:
    print(task, os.path.join(src_dir, task+'.train.smp001.1shot-mask.retry3.json'))
    with jsonlines.open(os.path.join(src_dir, task+'.train.smp001.1shot-mask.retry3.json')) as f:
        data = [l for l in f]

    emb_list = None

    for i in tqdm(range(0, len(data), bsz)):
        texts = [l['inputs'] for l in data[i:i+bsz]]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.to("cpu")
        if emb_list is None:
            emb_list = embeddings
        else: emb_list = torch.cat([emb_list, embeddings], dim=0)
    
    print(f"emb_list.shape:{emb_list.shape}")
    n_emb, n_dim = emb_list.shape

    emb_list = emb_list.numpy()

    np.save(os.path.join(src_dir, task+".train.smp001.1shot-mask.retry3.npy"), emb_list)
