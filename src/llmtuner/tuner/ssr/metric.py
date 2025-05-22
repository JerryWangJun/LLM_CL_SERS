import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer


@dataclass
### 这个代码定义了一个 ComputeMetrics 类，用于计算文本生成模型的评估指标，包括 ROUGE 和 BLEU 分数。
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}
    

    ##############################新增函数实现输出每一个样本的ROUGE-L分数################
    def compute_rouge_l_per_sample(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, list[float]]:
        """
        This function computes the ROUGE-L F-scores for each individual sample.
        """
        preds, labels = eval_preds
        rouge_l_scores = []

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                rouge_l_scores.append(0.0)
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                rouge_l_scores.append(round(scores[0]["rouge-l"]["f"] * 100, 4))  # 转换为百分制

        return {"rouge-l": rouge_l_scores}
