import os
import json
import torch
from collections import deque
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        # ��� is_replay �ֶε� batch ��
        for idx, f in enumerate(features):
            if "is_replay" not in f:
                print(f"ERROR: Missing 'is_replay' in feature {idx}")
                print(f"Feature keys: {f.keys()}")
                raise KeyError(f"'is_replay' missing in feature {idx}")
        batch["is_replay"] = torch.tensor([f["is_replay"] for f in features])
        return batch

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """ 
    def __init__(self, *args, ws_distance=None, lora_target=None, lambda_max=0.6, lambda_min=0.15, w_threshold=0.08, **kwargs):
        Seq2SeqTrainer.__init__(self, *args, **kwargs)
    
        self.args.remove_unused_columns = False
        print("lora target:", lora_target)
        self.lora_target = lora_target
        if self.args.do_train:
            self.ws_dis = ws_distance
            self.lambda_reg = lambda_min + (lambda_max - lambda_min) * (1 - np.exp(-self.ws_dis/w_threshold))
            self.lambda_reg = np.clip(self.lambda_reg, lambda_min, lambda_max)
            self._lora_params = self._get_lora_parameters()
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.w_threshold = w_threshold
        
        
                           
    def _get_lora_parameters(self):
        lora_params = []
        print("lora_target:", self.lora_target)
        for n, p in self.model.named_parameters():
            if "lora" in n and any(t in n for t in self.lora_target):
                lora_params.append(n)
    
        return lora_params
        
    def compute_loss(self, model, inputs, return_outputs=False):
        is_replay = inputs.pop("is_replay")
        outputs = model(**inputs)
        loss = outputs.loss
        
        replay_ratio = is_replay.float().mean()
        
        if self.lambda_reg > 0.15:  # ���˹�С��lambdaֵ
            reg_loss = 0.5 * sum(
                torch.mean(p.pow(2)) 
                for n, p in model.named_parameters() 
                if n in self._lora_params
            )
            effective_lambda = self.lambda_reg * (1 - replay_ratio)    # �طű���Խ�ߣ�����Խ��
            print(f"Reg loss: {reg_loss.item()}, Lambda: {effective_lambda}")  # �������
            loss += effective_lambda * reg_loss
        
        return (loss, outputs) if return_outputs else loss
   
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = self._pad_tensors_to_target_len(
                        inputs["attention_mask"], inputs["labels"], pad_token_id=0
                    )
                if "position_ids" in inputs:
                    inputs["position_ids"] = self._pad_tensors_to_target_len(
                        inputs["position_ids"], inputs["labels"], pad_token_id=0
                    )

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
    
        if generated_tokens is not None and self.args.predict_with_generate:

            generated_tokens[:, :max(prompt_len, label_len)] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


    
    def get_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return decoded_preds, decoded_labels
    

    def label_evolution(self, dataset, model_1, model_2, tokenizer, se_alpha = 0.5, max_output_length=512, ):
        self.args.per_device_eval_batch_size=1
        dataloader = self.get_eval_dataloader(dataset)
        
        generated_ids_list = [] 
        output_ids = []
        
        for batch in tqdm(dataloader, desc="Generating outputs", unit="batch"):

            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            assert self.tokenizer.pad_token_id is not None, "Pad token is required."
            input_ids = batch["input_ids"]  
            eos_token_id = tokenizer.eos_token_id
            current_length = input_ids.size(-1)
            generated_ids = input_ids
            input_lenth = len(input_ids)
            max_total_length = input_lenth + max_output_length 
            while current_length < max_total_length:
                with torch.no_grad():
                    logits_1 = model_1(input_ids=generated_ids).logits[:, -1, :]
                    logits_2 = model_2(input_ids=generated_ids).logits[:, -1, :]

                new_logits = torch.log(se_alpha * torch.exp(logits_1) + (1-se_alpha) * torch.exp(logits_2))

                probs = F.softmax(new_logits, dim=-1)
                next_token_id = torch.argmax(probs, dim=-1)
                
                generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)
                if next_token_id.item() == eos_token_id:
                  break

                current_length += 1
            generated_ids_list.append(generated_ids[:, input_ids.size(-1):]) 
            
        output_seqs = []
        for generated_id in generated_ids_list:
            generated_id_np = generated_id.cpu().numpy()
            generated_id_np = np.where(generated_id_np != IGNORE_INDEX, generated_id_np, self.tokenizer.pad_token_id)
            decoded_seq = self.tokenizer.batch_decode(generated_id_np, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for item in decoded_seq:
              output_seqs.append(item)
                  
        return output_seqs
    