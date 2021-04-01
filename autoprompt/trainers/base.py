from typing import Any, Dict, Tuple, Union

import transformers
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import autoprompt.templatizers as templatizers
import autoprompt.utils as utils


class Trainer:
    def __init__(
        self,
        args: Dict[str, Any],
        config: transformers.PretrainedConfig,
        tokenizer: transformers.PreTrainedTokenizer,
        templatizer: templatizers.MultiTokenTemplatizer,
        label_map: Dict[str, str],
        distributed_config: utils.DistributedConfig,
        writer: Union[SummaryWriter, utils.NullWriter],
    ) -> None:
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.templatizer = templatizer
        self.label_map = label_map
        self.distributed_config = distributed_config
        self.writer = writer

    def train(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader,
    ) -> Tuple[torch.nn.Module, float]:
        raise NotImplementedError

    def test(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
    ) -> float:
        raise NotImplementedError
