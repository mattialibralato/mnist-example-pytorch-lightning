from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def prepare_data(self) -> None:
        MNIST(self.data_dir, train = True, transform = ToTensor(), download = True)
        MNIST(self.data_dir, train = False, transform = ToTensor(), download = True)

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = MNIST(self.data_dir, train = False, transform = ToTensor(), download = False)
        mnist_full = MNIST(self.data_dir, train = True, transform = ToTensor(), download = False)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size, shuffle = True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size = self.batch_size, shuffle = True)