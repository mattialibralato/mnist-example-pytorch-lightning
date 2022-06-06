from datamodule import MNISTDataModule
from engine import MNISTEngine
from pytorch_lightning import Trainer

def main(
    data_path: str = "data",
    batch_size: int = 128,
    max_epochs: int = 2,
    lr: float = 0.001,
    lr_scheduler_step_size: int = 1,
    lr_scheduler_gamma: float = 0.7
) -> None:
    datamodule = MNISTDataModule(data_path, batch_size)
    model = MNISTEngine(lr = lr, lr_scheduler_init_kwargs = {"step_size": lr_scheduler_step_size, "gamma": lr_scheduler_gamma})

    #Training
    trainer = Trainer(max_epochs = max_epochs)
    trainer.fit(model = model, datamodule = datamodule)

    #Testin
    trainer.test(model = model, datamodule = datamodule)

if __name__ == "__main__":
    main()