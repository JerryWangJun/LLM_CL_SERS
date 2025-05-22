# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py
import os
from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
import random
import torch

from torch.nn.utils.rnn import pad_sequence
from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import CustomSeq2SeqTrainer, CustomDataCollator

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


import torch
import random
from torch.nn.utils.rnn import pad_sequence

import torch
import random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import torch
import random
from torch.nn.utils.rnn import pad_sequence

def compute_ws(model, dataset, pad_token_id, num_projection=32, device="cuda"):

    new_data = [ex for ex in dataset if ex["is_replay"] == 0]
    old_data = [ex for ex in dataset if ex["is_replay"] == 1]

    sample_size = min(500, len(new_data), len(old_data))
    print("check sample_size: ", sample_size)
    new_samples = random.sample(new_data, sample_size)
    old_samples = random.sample(old_data, sample_size)
    
    def batch_process(samples, batch_size=16):
        batches = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            input_ids = pad_sequence(
                [torch.tensor(s["input_ids"]) for s in batch],
                batch_first=True,
                padding_value=pad_token_id
            ).to(device)
            attention_mask = pad_sequence(
                [torch.tensor(s["attention_mask"], dtype=torch.bfloat16) for s in batch],
                batch_first=True,
                padding_value=pad_token_id,
            ).to(device)
            batches.append((input_ids, attention_mask))
        return batches

    model = model.to(device).bfloat16()
    model.eval() 
    new_embs, old_embs = [], []

    with torch.no_grad():
        hidden_size = model.config.hidden_size
        proj_matrix = torch.linalg.qr(
            torch.randn(hidden_size, num_projection, device=device)
        )[0].to(torch.bfloat16)
        print("check proj_size", proj_matrix.shape)
        
        if sample_size <=400:
            batch_size = 2
        elif sample_size <=600:
            batch_size = 2
        elif sample_size <=800:
            batch_size = 2            
        else:
            batch_size = 2
        print("check batch_size:", batch_size)

        for data_type, samples in [("new", new_samples), ("old", old_samples)]:
            emb_list = []
            for batch in batch_process(samples, batch_size):
                input_ids, attention_mask = batch
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    h = outputs.hidden_states[-2].mean(dim=1)  # [batch, dim]
                del outputs
                
                projected = F.normalize(h.to(torch.bfloat16) @ proj_matrix, dim=-1)
                emb_list.append(projected.cpu())
                
                del h, projected
                torch.cuda.empty_cache()
            if data_type == "new":
                new_embs = torch.cat(emb_list, dim=0)
            else:
                old_embs = torch.cat(emb_list, dim=0)

    quantiles = torch.linspace(0.05, 0.95, 20, device=new_embs.device)
    ws_dists = []
    for dim in range(num_projection):
        q_new = torch.quantile(new_embs[:, dim].float(), quantiles)
        q_old = torch.quantile(old_embs[:, dim].float(), quantiles)
        ws_dists.append(torch.mean(torch.abs(q_new - q_old)))
    ws_dist = torch.mean(torch.stack(ws_dists))
    print(f"Robust WS Distance: {ws_dist:.4f}")
    return ws_dist.item()



def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)     
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
    
    for i in range(len(dataset)):
      assert "is_replay" in dataset[i], f"Sample {i} missing 'is_replay' field"

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation

    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    
    if training_args.do_train:
        ws_distance = compute_ws(
            model=model, 
            dataset=dataset,
            pad_token_id=tokenizer.pad_token_id,
            device=training_args.device
        )
    
    
    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.max_target_length,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        ws_distance = ws_distance, 
        lora_target = finetuning_args.lora_target,
        lambda_min = finetuning_args.lambda_min,
        lambda_max = finetuning_args.lambda_max,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    if training_args.do_eval or training_args.do_predict:
        gen_kwargs["use_cache"] = True

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        trainer.model.gradient_checkpointing_disable()
        trainer.model.config.use_cache = True
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        trainer.model.gradient_checkpointing_disable()
        trainer.model.config.use_cache = True
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
        

