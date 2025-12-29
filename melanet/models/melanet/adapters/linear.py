import os

import torch
import torch.nn as nn

from typing import Any, Optional

from torch.utils.data import DataLoader, TensorDataset
from transformers import Trainer, TrainingArguments, set_seed

from ..embeddings.utils import normalize_embeddings
from ..utils import resolve_device


class LinearAdapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        normalize: bool = False
    ):
        super(LinearAdapter, self).__init__()

        self.projection = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.normalize = normalize
        self._is_fitted = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert self.training or self._is_fitted, "LinearAdapter is not fitted. Call `fit()` first."

        proj_features = self.projection(features)
        if self.normalize:
            proj_features = normalize_embeddings(proj_features)
        return proj_features

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
    ) -> None:
        if seed is not None:
            set_seed(seed)
        device = resolve_device(device)

        self.train().to(device)
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

        def log(loss, epoch, max_epochs, step, max_steps):
            print(f"[Epoch: {epoch}/{max_epochs}; Step: {step}/{max_steps}]\tLoss: {loss:.5f}")

        losses = []
        steps = 0
        max_steps = len(data_loader) * epochs
        for epoch in range(epochs):
            print("="*80)
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
                if steps % training_args.logging_steps == 0:
                    log(losses[-1], epoch+1, epochs, steps, max_steps)
            log(losses[-1], epoch+1, epochs, steps, max_steps)

        self._is_fitted = True
        self.eval()

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LinearAdapter":
        checkpoint = torch.load(pretrained_path)

        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model._is_fitted = True

        return model

    def save_as_pretrained(self, save_path: str, allow_overwrite: bool = True) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=allow_overwrite)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "in_features": self.projection.in_features,
                    "out_features": self.projection.out_features,
                    "bias": self.projection.bias is not None,
                    "normalize": self.normalize
                }
            },
            save_path
        )