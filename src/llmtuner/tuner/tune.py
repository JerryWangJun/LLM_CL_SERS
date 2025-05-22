from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core import get_train_args, load_model_and_tokenizer
from llmtuner.tuner.pt import run_pt
from llmtuner.tuner.sft import run_sft
from llmtuner.tuner.sftrp import run_sftrp
from llmtuner.tuner.ssr import run_ssr

if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if general_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "ssr":
        run_ssr(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "sftrp":
        run_sftrp(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, _, training_args, finetuning_args, _, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
