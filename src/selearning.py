import os
import sys
from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from llmtuner.extras.callbacks import LogCallback
import torch
import copy
import numpy as np
import gc
import json
import torch.nn.functional as F

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.tuner.core import get_train_args, load_model_and_tokenizer
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import CustomSeq2SeqTrainer
if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments, GeneralArguements

def main():
    run_se_learning()

def run_se_learning(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks
    
    self_evolution(model_args, data_args, training_args, finetuning_args, generating_args, general_args, callbacks)


def self_evolution(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    general_args: "GeneralArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
    threshold = 30.0,
    config_path = "/project/wangjun4/LLM_CL/SSR-main-copy/data/dataset_info.json"
):
    ###############先导入初始模型###################
    model_args_llm0 = copy.deepcopy(model_args)
    model_args_llm0.checkpoint_dir = None
    dataset_llm0 = get_dataset(model_args_llm0, data_args)
    model_llm0, tokenizer_llm0 = load_model_and_tokenizer(model_args_llm0, finetuning_args, training_args.do_train, stage="sft")
    dataset_llm0 = preprocess_dataset(dataset_llm0, tokenizer_llm0, data_args, training_args, stage="sft")

    if training_args.predict_with_generate:
        tokenizer_llm0.padding_side = "left" # use left-padding in generation

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer_llm0,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer_llm0.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    # 更新训练参数
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.max_target_length,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    # 初始化自定义训练器
    trainer_llm0 = CustomSeq2SeqTrainer(
        model=model_llm0,
        args=training_args,
        tokenizer=tokenizer_llm0,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer_llm0) if training_args.predict_with_generate else None,
        **split_dataset(dataset_llm0, data_args, training_args)
    )

    # Keyword arguments for `model.generate`
    # 设置生成参数
    gen_kwargs_llm0 = generating_args.to_dict()
    gen_kwargs_llm0["eos_token_id"] = [tokenizer_llm0.eos_token_id] + tokenizer_llm0.additional_special_tokens_ids
    gen_kwargs_llm0["pad_token_id"] = tokenizer_llm0.pad_token_id
    gen_kwargs_llm0["logits_processor"] = get_logits_processor()
    if training_args.do_eval or training_args.do_predict:
        gen_kwargs_llm0["use_cache"] = True

    trainer_llm0.model.gradient_checkpointing_disable()
    trainer_llm0.model.config.use_cache = True

    #############################导入当前模型###########################################
    dataset = get_dataset(model_args, data_args) 
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft") # input_ids, attention mask, labels

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    if training_args.do_eval or training_args.do_predict:
        gen_kwargs["use_cache"] = True

    trainer.model.gradient_checkpointing_disable()
    trainer.model.config.use_cache = True

    #################评估困难样本#####################
    low_rouge_idx, low_rouge_data = self_question(trainer, tokenizer, dataset, gen_kwargs, threshold = general_args.se_ratio)
    
    evolution_data = trainer.label_evolution(low_rouge_data, trainer_llm0.model, trainer.model, tokenizer, se_alpha=general_args.se_alpha)
    
    all_indices = list(range(len(dataset)))
    non_low_rouge_idx = [idx for idx in all_indices if idx not in low_rouge_idx]
    non_low_rouge_data = [dataset[idx] for idx in non_low_rouge_idx]
    non_low_rouge_data_pred_results = trainer.predict(non_low_rouge_data, metric_key_prefix="predict", **gen_kwargs)
    non_low_rouge_data_pred, non_low_rouge_data_label = trainer.get_predictions(non_low_rouge_data_pred_results)

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_data = json.load(config_file)  
    dataset_file_path = config_data[data_args.dataset]["file_name"]
    dataset_file_path = os.path.join(data_args.dataset_dir, dataset_file_path)
    with open(dataset_file_path, 'r', encoding='utf-8') as target_file:
        data_lines = target_file.readlines()

##################对于生成样本的保存##############################    
    for idx, new_output in zip(low_rouge_idx, evolution_data):
        data_modify = json.loads(data_lines[idx])  # 解析当前行的 JSON 数据
        data_modify["targets"] = new_output
        del data_modify['outputs'] 
        data_lines[idx] = json.dumps(data_modify, ensure_ascii=False) + "\n"

    for idx, output in zip(non_low_rouge_idx, non_low_rouge_data_pred):
        data_modify = json.loads(data_lines[idx])  # 解析当前行的 JSON 数据
        data_modify["targets"] = output
        del data_modify['outputs']
        data_lines[idx] = json.dumps(data_modify, ensure_ascii=False) + "\n"

#####################对于训练样本的保存(有id等字段需要删除，输入字段是input而不是inputs)################################
#     for idx, new_output in zip(low_rouge_idx, evolution_data):
#         data_modify = json.loads(data_lines[idx])  # 解析当前行的 JSON 数据
#         data_modify["output"] = new_output  # 设置新的 targets         
#         data_lines[idx] = json.dumps(data_modify, ensure_ascii=False) + "\n"

#     for idx, output in zip(non_low_rouge_idx, non_low_rouge_data_pred):
#         data_modify = json.loads(data_lines[idx])  # 解析当前行的 JSON 数据
#         data_modify["output"] = output  # 设置新的 targets
#         data_lines[idx] = json.dumps(data_modify, ensure_ascii=False) + "\n"
        

    output_path = os.path.join(training_args.output_dir, general_args.output_file)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(data_lines)



# def self_question(trainer, tokenizer, dataset, gen_kwargs, threshold = 30.0):

#     predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
#     metrics = ComputeMetrics(tokenizer)
#     Rouge_L = metrics.compute_rouge_l_per_sample((predict_results.predictions, predict_results.label_ids))
#     low_rouge_idx = [
#        idx for idx, score in enumerate(Rouge_L["rouge-l"]) if score <= threshold
#     ]
#     low_rouge_data = [dataset[idx] for idx in low_rouge_idx]
# 
#     return low_rouge_idx, low_rouge_data

def self_question(trainer, tokenizer, dataset, gen_kwargs, threshold = 30.0):
    predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
    metrics = ComputeMetrics(tokenizer)
    Rouge_L = metrics.compute_rouge_l_per_sample((predict_results.predictions, predict_results.label_ids))
    rouge_l_scores = Rouge_L["rouge-l"]
    sorted_idx = sorted(range(len(rouge_l_scores)), key=lambda idx: rouge_l_scores[idx])
    threshold_idx = int(len(rouge_l_scores) * threshold / 100)
    low_rouge_idx = sorted_idx[:threshold_idx]
    low_rouge_data = [dataset[idx] for idx in low_rouge_idx]
    
    return low_rouge_idx, low_rouge_data





if __name__ == "__main__":
    main()
