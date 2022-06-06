from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)   # along axis 0 we expect the items of the batch

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 2, (3, 3), padding = 1),   # 1 x 28 x 28 to 2 x 28 x 28
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 2, (3, 3), padding = 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                   # 2 x 28 x 28 to 2 x 14 x 14
            nn.Conv2d(2, 4, (3, 3), padding = 1),   # 2 x 14 x 14 to 4 x 14 x 14
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, (3, 3), padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),                   # 4 x 14 x 14 to 4 x 7 x 7
            Flatten(),                              # 4 x 7 x 7 to 1 x 196
            nn.Linear(196, 10),                     # 1 x 196 to 1 x 10
        )
    
    def forward(self, x):
        return self.model(x)