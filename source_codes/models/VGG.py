import torch.nn as nn


def conv_layer_b(channel_input, channel_output):
    layer = nn.Sequential(
        nn.Conv2d(channel_input, channel_output, kernel_size=3, padding=1),
        nn.BatchNorm2d(channel_output),
        nn.ReLU()
    )
    return layer


def conv_layer(channel_input, channel_output):
    layer = nn.Sequential(
        nn.Conv2d(channel_input, channel_output, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return layer


def vgg_conv_block_b(input_list, output_list):
    layers = [conv_layer_b(input_list[i], output_list[i])
              for i in range(len(input_list))]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def vgg_conv_block(input_list, output_list):
    layers = [conv_layer(input_list[i], output_list[i])
              for i in range(len(input_list))]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


def vgg_full_connected_layer(size_input, size_output):
    layer = nn.Sequential(
        nn.Linear(size_input, size_output),
        nn.BatchNorm1d(size_output),
        nn.ReLU(),
    )
    return layer


class VGG11(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG11, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 1 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        self.conv1 = vgg_conv_block(input_list=[3], output_list=[64])

        # Block 02
        # Contain 1 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        self.conv2 = vgg_conv_block(input_list=[64], output_list=[128])

        # Block 03
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 1x1, activation function ReLU
        self.conv3 = vgg_conv_block(
            input_list=[128, 256], output_list=[256, 256])

        # Block 04
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv4 = vgg_conv_block(
            input_list=[256, 512], output_list=[512, 512])

        # Block 05
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv5 = vgg_conv_block(
            input_list=[512, 512], output_list=[512, 512])

        # Full-Connected Layer
        self.fc1 = vgg_full_connected_layer(size_input=512, size_output=4096)
        self.fc2 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        vgg11_features = self.conv5(x)
        x = vgg11_features.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.classifier(x)
        return x


class VGG13(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG13, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        self.conv1 = vgg_conv_block(input_list=[3, 64], output_list=[64, 64])

        # Block 02
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        self.conv2 = vgg_conv_block(
            input_list=[64, 128], output_list=[128, 128])

        # Block 03
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 1x1, activation function ReLU
        self.conv3 = vgg_conv_block(
            input_list=[128, 256], output_list=[256, 256])

        # Block 04
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv4 = vgg_conv_block(
            input_list=[256, 512], output_list=[512, 512])

        # Block 05
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv5 = vgg_conv_block(
            input_list=[512, 512], output_list=[512, 512])

        # Full-Connected Layer
        self.fc1 = vgg_full_connected_layer(size_input=512, size_output=4096)
        self.fc2 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        vgg13_features = self.conv5(x)
        x = vgg13_features.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.classifier(x)

        return x


class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        self.conv1 = vgg_conv_block(input_list=[3, 64], output_list=[64, 64])

        # Block 02
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        self.conv2 = vgg_conv_block(
            input_list=[64, 128], output_list=[128, 128])

        # Block 03
        # Contain 3 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 3x3, activation function ReLU
        self.conv3 = vgg_conv_block(
            input_list=[128, 256, 256], output_list=[256, 256, 256])

        # Block 04
        # Contain 3 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv4 = vgg_conv_block(
            input_list=[256, 512, 512], output_list=[512, 512, 512])

        # Block 05
        # Contain 3 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv5 = vgg_conv_block(
            input_list=[512, 512, 512], output_list=[512, 512, 512])

        # Full-Connected Layer
        self.fc1 = vgg_full_connected_layer(size_input=512, size_output=4096)
        self.fc2 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        vgg16_features = self.conv5(x)
        x = vgg16_features.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.classifier(x)
        return x


class VGG19(nn.Module):

    def __init__(self, n_classes=1000):
        super(VGG19, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)

        # Block 01
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 64 filters, kernel size 3x3, activation function ReLU
        self.conv1 = vgg_conv_block(input_list=[3, 64], output_list=[64, 64])

        # Block 02
        # Contain 2 convolution layers and 1 maxpool layer
        # Convolution layer has 128 filters, kernel size 3x3, activation function ReLU
        self.conv2 = vgg_conv_block(
            input_list=[64, 128], output_list=[128, 128])

        # Block 03
        # Contain 4 convolution layers and 1 maxpool layer
        # Convolution layer has 256 filters, kernel size 3x3, activation function ReLU
        self.conv3 = vgg_conv_block(
            input_list=[128, 256, 256, 256], output_list=[256, 256, 256, 256])

        # Block 04
        # Contain 4 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv4 = vgg_conv_block(
            input_list=[256, 512, 512, 512], output_list=[512, 512, 512, 512])

        # Block 05
        # Contain 4 convolution layers and 1 maxpool layer
        # Convolution layer has 512 filters, kernel size 3x3, activation function ReLU
        self.conv5 = vgg_conv_block(
            input_list=[512, 512, 512], output_list=[512, 512, 512])

        # Full-Connected Layer
        self.fc1 = vgg_full_connected_layer(size_input=512, size_output=4096)
        self.fc2 = vgg_full_connected_layer(size_input=4096, size_output=4096)

        # Final layer
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        vgg19_features = self.conv5(x)
        x = vgg19_features.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.classifier(x)
        return x
