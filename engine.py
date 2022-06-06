from typing import Optional
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.nn import CrossEntropyLoss
from model import MyModel

class MNISTEngine(LightningModule):
    def __init__(self, lr: float = 0.001, lr_scheduler_init_kwargs: dict = {"step_size": 1, "gamma": 0.7}) -> None:
        super().__init__()
        self.model = MyModel()
        self.lr_scheduler_init_kwargs = lr_scheduler_init_kwargs
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int, **kwargs) -> STEP_OUTPUT:
        images, labels = batch
        logits = self.model(images)
        return CrossEntropyLoss()(logits, labels)
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int, **kwargs) -> Optional[STEP_OUTPUT]:
        images, labels = batch
        logits = self.model(images)
        return CrossEntropyLoss()(logits, labels)
    
    def test_step(self, batch: torch.Tensor, batch_idx: int, **kwargs) -> Optional[STEP_OUTPUT]:
        images, labels = batch
        logits = self.model(images)
        return CrossEntropyLoss()(logits, labels)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, **self.lr_scheduler_init_kwargs)
        return [optimizer], [lr_scheduler]
