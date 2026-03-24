import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input = 224
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels =  32, kernel_size = 2),
            # output size = input - (kernel - 1) -> 224 - (2 - 1) = 223
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            # output size = input / 2 = 223 / 2 = 111
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels =  32, kernel_size = 2),
            # 111 - (2 - 1) = 110
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
            # 110 / 2 = 55
        )
        self.layer3 = nn.Flatten()
        # 32 feature maps (channels) * 55 (height) * 55 (width) = 96,800
        self.layer4 = nn.Sequential(
            nn.Linear(in_features = 96800, out_features = 256),
            nn.ReLU()
        )
        self.layer5 = nn.Linear(in_features= 256, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)