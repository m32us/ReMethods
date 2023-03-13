import torch.nn as nn


def conv_layer_b(channel_input, channel_output):
    return [
        nn.Conv2d(channel_input, channel_output, kernel_size=3, padding=1),
        nn.BatchNorm2d(channel_output),
        nn.ReLU()
    ]


def conv_layer(channel_input, channel_output):
    return [
        nn.Conv2d(channel_input, channel_output, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    ]


def vgg_conv_block_b(input_list, output_list):
    layers = []
    for i in range(len(input_list)):
        layers += conv_layer_b(input_list[i], output_list[i])
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return layers


def vgg_conv_block(input_list, output_list):
    layers = []
    for i in range(len(input_list)):
        layers += conv_layer(input_list[i], output_list[i])
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return layers


def vgg_full_connected_layer(size_input, size_output):
    return [
        nn.Linear(size_input, size_output),
        nn.Dropout(p=0.5, inplace=True),
        nn.ReLU(inplace=True),
    ]


class VGG11(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG11, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 1 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        layer1 = vgg_conv_block(input_list=[3], output_list=[64])

        # Block 02
        # Contain 1 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        layer2 = vgg_conv_block(input_list=[64], output_list=[128])

        # Block 03
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 1x1, activation function ReLU
        layer3 = vgg_conv_block(input_list=[128, 256], output_list=[256, 256])

        # Block 04
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        layer4 = vgg_conv_block(input_list=[256, 512], output_list=[512, 512])

        # Block 05
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        layer5 = vgg_conv_block(input_list=[512, 512], output_list=[512, 512])

        avgpool = [nn.AdaptiveAvgPool2d(output_size=(1, 1))]

        self.features = layer1 + layer2 + layer3 + layer4 + layer5 + avgpool
        self.feature_extractor = nn.Sequential(*self.features)

        # Full-Connected Layer
        layer6 = vgg_full_connected_layer(size_input=512, size_output=4096)
        layer7 = vgg_full_connected_layer(size_input=4096, size_output=4096)
        layer8 = [nn.Linear(4096, n_classes)]

        # Final layer
        self.fc = layer6 + layer7 + layer8
        self.classifier = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG13(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG13, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        conv1 = vgg_conv_block(input_list=[3, 64], output_list=[64, 64])

        # Block 02
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        conv2 = vgg_conv_block(
            input_list=[64, 128], output_list=[128, 128])

        # Block 03
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 1x1, activation function ReLU
        conv3 = vgg_conv_block(
            input_list=[128, 256], output_list=[256, 256])

        # Block 04
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        conv4 = vgg_conv_block(
            input_list=[256, 512], output_list=[512, 512])

        # Block 05
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        conv5 = vgg_conv_block(
            input_list=[512, 512], output_list=[512, 512])

        self.features = conv1 + conv2 + conv3 + conv4 + conv5
        self.feature_extractor = nn.Sequential(*self.features)

        # Full-Connected Layer
        layer6 = vgg_full_connected_layer(size_input=512, size_output=4096)
        layer7 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        layer8 = nn.Linear(4096, n_classes)

        self.fc = layer6 + layer7 + layer8
        self.classifier = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        conv1 = vgg_conv_block(input_list=[3, 64], output_list=[64, 64])

        # Block 02
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        conv2 = vgg_conv_block(
            input_list=[64, 128], output_list=[128, 128])

        # Block 03
        # Contain 3 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 3x3, activation function ReLU
        conv3 = vgg_conv_block(
            input_list=[128, 256, 256], output_list=[256, 256, 256])

        # Block 04
        # Contain 3 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        conv4 = vgg_conv_block(
            input_list=[256, 512, 512], output_list=[512, 512, 512])

        # Block 05
        # Contain 3 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        conv5 = vgg_conv_block(
            input_list=[512, 512, 512], output_list=[512, 512, 512])

        self.features = conv1 + conv2 + conv3 + conv4 + conv5
        self.feature_extractor = nn.Sequential(*self.features)

        # Full-Connected Layer
        layer6 = vgg_full_connected_layer(size_input=512, size_output=4096)
        layer7 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        layer8 = nn.Linear(4096, n_classes)

        self.fc = layer6 + layer7 + layer8
        self.classifier = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG19(nn.Module):

    def __init__(self, n_classes=1000):
        super(VGG19, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        conv1 = vgg_conv_block(input_list=[3, 64], output_list=[64, 64])

        # Block 02
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        conv2 = vgg_conv_block(
            input_list=[64, 128], output_list=[128, 128])

        # Block 03
        # Contain 4 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 3x3, activation function ReLU
        conv3 = vgg_conv_block(
            input_list=[128, 256, 256, 256], output_list=[256, 256, 256, 256])

        # Block 04
        # Contain 4 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        conv4 = vgg_conv_block(
            input_list=[256, 512, 512, 512], output_list=[512, 512, 512, 512])

        # Block 05
        # Contain 4 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        conv5 = vgg_conv_block(
            input_list=[512, 512, 512], output_list=[512, 512, 512])

        self.features = conv1 + conv2 + conv3 + conv4 + conv5
        self.feature_extractor = nn.Sequential(*self.features)

        # Full-Connected Layer
        layer6 = vgg_full_connected_layer(size_input=512, size_output=4096)
        layer7 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        layer8 = nn.Linear(4096, n_classes)

        self.fc = layer6 + layer7 + layer8
        self.classifier = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
