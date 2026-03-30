import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, (3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels, dropout)
        self.max_pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.conv(x)
        return self.max_pool(x), x

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, (2,2), stride=2)
        # concat skip here
        self.conv = DoubleConv(out_channels * 2, out_channels, dropout)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.encoder1 = Encoder(in_channels, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)
        self.bottleneck = DoubleConv(512, 1024, dropout=0.5)
        self.decoder1 = Decoder(1024, 512, dropout=0.5)
        self.decoder2 = Decoder(512, 256, dropout=0.3)
        self.decoder3 = Decoder(256, 128, dropout=0.3)
        self.decoder4 = Decoder(128, 64, dropout=0.0)
        self.output = nn.Conv2d(64, out_channels, 1)


    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        x = self.bottleneck(x)
        x = self.decoder1(x, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)
        return self.output(x)


class DiceScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Generates the dice score between tumor pixels in the mask
        :param prediction: prediction mask
        :param target: target mask
        :return:
        """
        intersection = torch.sum(prediction * target)
        sum_pred = torch.sum(prediction)
        sum_target = torch.sum(target)
        return (2 * intersection) / ( sum_pred + sum_target + 1e-6)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Generates the dice loss between tumor pixels in the mask
        :param prediction: prediction mask
        :param target: target mask
        :return:
        """
        intersection = torch.sum(prediction * target)
        sum_pred = torch.sum(prediction)
        sum_target = torch.sum(target)
        return 1 - (2 * intersection) / ( sum_pred + sum_target + 1e-6)



# SANITY CHECK
"""
model = UNetModel(in_channels=3, out_channels=1)
x = torch.randn(1, 3, 256, 256)
print(model(x).shape)  # should be (1, 1, 256, 256)
"""