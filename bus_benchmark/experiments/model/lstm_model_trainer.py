import logging
from typing import Optional

import torch
import torch.nn as nn

from bus_benchmark.experiments.fixed_interval_dataset import FixedIntervalDataset
from bus_benchmark.experiments.model.model_trainer_base import ModelTrainerBase
from .lstm_model import LSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModelTrainer(ModelTrainerBase):
    def __init__(
        self,
        base_dataset: FixedIntervalDataset,
        n_epochs: int = 30,
        seq_len: int = 8,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 10,
        factor: float = 0.5,
        weight_decay: float = 1e-5,
        data_set_name: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            model_name="LSTM",
            base_dataset=base_dataset,
            n_epochs=n_epochs,
            seq_len=seq_len,
            horizons=horizons,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            factor=factor,
            weight_decay=weight_decay,
            data_set_name=data_set_name,
            verbose=verbose,
        )

    def _get_config_dict(self):
        return {
            **super()._get_config_dict(),
            "model_class": "LSTM",
            "optimizer": "SmoothL1Loss",
            "loss_fn": "Adam",
            "scheduler": "ReduceLROnPlateau",
        }

    def _get_model(
        self, n_input_features: int, n_output_features: int, n_output_timesteps: int
    ) -> nn.Module:
        model = LSTMModel(
            n_input_features=n_input_features,
            n_output_features=n_output_features,
            n_output_timesteps=n_output_timesteps,
        )
        return model.to(self.device)

    def _run_model(
        self, model: nn.Module, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return model(x)

    def _get_loss_fn(self):
        return torch.nn.SmoothL1Loss()

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _get_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.factor, patience=self.patience
        )
