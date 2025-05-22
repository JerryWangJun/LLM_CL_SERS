# Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting with Task Similarity in LLMs

## Introduction
This work aims to alleviate the catastrophic forgetting problem encountered in the continuous learning process of large language models through pseudo sample rehearsal and task similarity regularization.

## Environment Requirements
Our environment is given in requirements.txt
All experiments were run on an A100 GPU.

## How to Run
1. Run "/LLM_CL_SERS/custom/mask_gen/scripts-ni-c012/{model}/mask_pesudo_generate.sbatch" to generate pseudo inputs from a small number of real samples
2. Run "/LLM_CL_SERS/custom/mask_gen/scripts-ni-c012/{model}/pseudo_filter.sbatch" to filter pseudo inputs
3. Run "/LLM_CL_SERS/custom/mask_gen/scripts-ni-c012/{model}/txt2emb.sbatch" and "/LLM_CL_SERS/custom/mask_gen/scripts-ni-c012/{model}/kmeans_self.sbatch" to get pseudo inputs after clustering
3. Run "/LLM_CL_SERS/src/scripts-ni-c012/lora/sing/{model}/single.sbatch" to train a model fine-tuned on the first task
4. Add the storage location of the initial pseudo samples and the pseudo samples after evolution in "/LLM_CL_SERS/data/dataset_info.json" as following
    "ni_c012_icl_km20_ori_{model}_qa": {
    "file_name": "ni-cus0.12/genearated-icl-naive-kmeans20-self/{model}/ori-van/{task_name}.train.smp001.2shot.smp3.rp1.2.json",
    "columns": {
      "prompt": "inputs",
      "query": "",
      "response": "outputs",
      "history": ""
    }
  },

  "ni_c012_1shot-mask_km20_se{k}_alpha{alpha}_{cl\cl2\cl3}_queue_{model}_para": {
    "file_name": "ni-cus0.12/genearated-masked-pesudo-kmeans20-self/{model}/{cl\cl2\cl3}_queue/se{k}_alpha{alpha}/{task_name}.train.smp001.1shot-mask.retry3.json",
    "columns": {
      "prompt": "inputs",
      "query": "",
      "response": "targets",
      "history": ""
    }
  },	
5. Run "/LLM_CL_SERS/src/scripts-ni-c012/lora/{cl\cl2\cl3}/{model}/{model}_1shot-mask_se_wsreg.sbatch" to save the model of each training stage in the save path

## Acknowledgement
We would like to thank the authors of SSR for their open-sourced code.

