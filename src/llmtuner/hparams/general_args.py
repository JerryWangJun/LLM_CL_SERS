from typing import Literal, Optional
from dataclasses import dataclass, field
from typing import List

@dataclass
class GeneralArguments:
    r"""
    Arguments pertaining to which stage we are going to perform.
    """
    stage: Optional[Literal["sft", "ssr"]] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."}
    )
    reg_cl_method: Optional[Literal["ewc", "l2"]]= field(
        default="ewc",
        metadata={"help": ""}
    )
    reg_p: float = field(
        default=0.1
    )
    result_before: str = field(
        default=None
    )
    output_file: str = field(
        default=None
    )
    se_ratio: float = field(
        default=20.0
    )
    se_alpha: float = field(
        default=0.5
    )