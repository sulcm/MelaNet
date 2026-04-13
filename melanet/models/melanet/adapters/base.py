import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Union, Any
from abc import ABC, abstractmethod
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from transformers import Trainer, TrainingArguments, set_seed

from .config import FeatureAdapterConfig
from ..utils import resolve_device


class BaseAdapter(ABC):
    _is_fitted: bool = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseAdapter":
        ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, np.ndarray]:
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_path: str) -> "BaseAdapter":
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: FeatureAdapterConfig) -> "BaseAdapter":
        ...

    @abstractmethod
    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        ...


class LearnableAdapter(nn.Module, BaseAdapter):
    activation_str2fn: dict[str, nn.Module] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }

    def fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor],
        loss_fn: nn.Module,
        optim_name: str = TrainingArguments.default_optim,
        optim_kwargs: dict[str, Any] = {},
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 30,
        logging_steps: int = 100,
        num_workers: int = 4,
        shuffle: bool = True,
        device: str = "cuda",
        seed: Optional[int] = 42
    ) -> "LearnableAdapter":
        if seed is not None:
            set_seed(seed)
        device = resolve_device(device)

        def log(loss, epoch, max_epochs, step, max_steps):
            print(f"[Epoch: {epoch}/{max_epochs}; Step: {step}/{max_steps}]\tLoss: {loss:.5f}")

        training_args = TrainingArguments(
            do_train=True,
            learning_rate=learning_rate,
            optim=optim_name,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            dataloader_num_workers=num_workers,
            seed=seed
        )

        optim_cls, base_optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args, self)
        optim_kwargs = {
            **base_optim_kwargs,
            **optim_kwargs
        }
        optim = optim_cls(
            self.parameters(),
            **optim_kwargs
        )

        if y is None:
            def collate_fn(batch) -> tuple[torch.Tensor, None]:
                return torch.stack(batch), None

            dataset = X
        else:
            def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
                inputs, targets = zip(*batch)
                return torch.stack(inputs), torch.stack(targets)

            dataset = TensorDataset(X, y)

        data_loader = DataLoader(
            dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=shuffle,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collate_fn
        )

        self.train().to(device)

        losses = []
        steps = 0
        max_steps = len(data_loader) * epochs
        tqdm_bar = tqdm(desc="Step", total=max_steps)
        for epoch in range(epochs):
            print("=" * 80)
            for inputs, target in data_loader:
                inputs = inputs.to(device)
                if target is not None:
                    target = target.to(device)

                optim.zero_grad()

                output = self(inputs)

                loss = loss_fn(output, target)
                loss.backward()
                optim.step()

                steps += 1
                losses.append(loss.item())
                tqdm_bar.update(1)
                if steps % training_args.logging_steps == 0:
                    log(losses[-1], epoch+1, epochs, steps, max_steps)
            log(losses[-1], epoch+1, epochs, steps, max_steps)

        self._is_fitted = True
        self.eval()
        return self

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LearnableAdapter":
        checkpoint = torch.load(pretrained_path)

        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model._is_fitted = True

        return model