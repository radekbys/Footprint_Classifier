import torch, torchvision
from torch import nn


class BinaryFootprintClassifierModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers1 = nn.Sequential(
            # convolutional layers
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=2
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=3
            ),
            nn.LeakyReLU(),
            # pooling and dropout layers
            nn.MaxPool2d(kernel_size=5, stride=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout2d(p=0.1),
        )

        # deep layers
        self.layers2 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=4896, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x):

        return self.layers2(self.layers1(x))
