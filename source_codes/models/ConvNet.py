import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super(ConvNet, self).__init__()
        conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        # bn1 = nn.BatchNorm2d(16)
        non_linearity1 = nn.ReLU()
        max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # bn2 = nn.BatchNorm2d(32)
        non_linearity2 = nn.ReLU()
        max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.features = [conv1, non_linearity1,
                         max_pool1, conv2, non_linearity2, max_pool2, avgpool]
        self.feature_extractor = nn.Sequential(*self.features)

        self.fc = [
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
            ]
        self.classifier = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
