import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.activation = nn.ReLU() if activation == 'relu' else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class InceptionSepConvBlock(nn.Module):
    def __init__(self, in_channels, f1, f2, f3, activation='relu'):
        super(InceptionSepConvBlock, self).__init__()
        # Branch 1: Depth-wise separable convolution (3x3)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),  # Depth-wise
            nn.Conv2d(in_channels, f1, kernel_size=1, padding=0, bias=False),  # Point-wise
            nn.ReLU() if activation == 'relu' else nn.Identity(),
            nn.Conv2d(f1, f1, kernel_size=3, padding=1, groups=f1, bias=False),  # Depth-wise
            nn.Conv2d(f1, f1, kernel_size=1, padding=0, bias=False),  # Point-wise
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )
        # Branch 2: Depth-wise separable convolution (5x5)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),  # Depth-wise
            nn.Conv2d(in_channels, f2, kernel_size=1, padding=0, bias=False),  # Point-wise
            nn.ReLU() if activation == 'relu' else nn.Identity(),
            nn.Conv2d(f2, f2, kernel_size=5, padding=2, groups=f2, bias=False),  # Depth-wise
            nn.Conv2d(f2, f2, kernel_size=1, padding=0, bias=False),  # Point-wise
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )
        # Branch 3: MaxPooling + 1x1 Conv
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, f3, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat([branch1, branch2, branch3], dim=1)

class DepthWiseUNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=1, dropout_rate=0.0):
        super(DepthWiseUNet, self).__init__()
        self.dropout_rate = dropout_rate

        # Encoder
        self.conv1 = ConvBlock(in_channels, 64, 3)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = InceptionSepConvBlock(64, f1=64, f2=48, f3=16)  # 128 output channels (64+48+16)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = InceptionSepConvBlock(128, f1=128, f2=100, f3=28)  # 256 output channels (128+100+28)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = InceptionSepConvBlock(256, f1=256, f2=210, f3=46)  # 512 output channels (256+210+46)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = InceptionSepConvBlock(512, f1=512, f2=420, f3=92)  # 1024 output channels (512+420+92)

        # Decoder
        self.u6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_4 = InceptionSepConvBlock(1024, f1=256, f2=210, f3=46)  # 512 output channels (256+210+46)
        self.u7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_3 = InceptionSepConvBlock(512, f1=128, f2=100, f3=28)  # 256 output channels (128+100+28)
        self.u8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_2 = InceptionSepConvBlock(256, f1=64, f2=48, f3=16)  # 128 output channels (64+48+16)
        self.u9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_1 = ConvBlock(128, 64, 3)
        self.output = nn.Conv2d(64, out_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid() if out_classes == 1 else nn.Identity()

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        max1 = self.max1(conv1)
        conv2 = self.conv2(max1)
        max2 = self.max2(conv2)
        conv3 = self.conv3(max2)
        max3 = self.max3(conv3)
        conv4 = self.conv4(max3)
        max4 = self.max4(conv4)
        conv5 = self.conv5(max4)

        # Decoder
        u6 = self.u6(conv5)
        conct1 = torch.cat([u6, conv4], dim=1)
        conv_4 = self.conv_4(conct1)
        u7 = self.u7(conv_4)
        conct2 = torch.cat([u7, conv3], dim=1)
        conv_3 = self.conv_3(conct2)
        u8 = self.u8(conv_3)
        conct3 = torch.cat([u8, conv2], dim=1)
        conv_2 = self.conv_2(conct3)
        u9 = self.u9(conv_2)
        conct4 = torch.cat([u9, conv1], dim=1)
        conv_1 = self.conv_1(conct4)
        output = self.output(conv_1)
        output = self.sigmoid(output)
        return output
# Instantiate the model
model = DepthWiseUNet(in_channels=3, out_classes=1)

# Create a dummy input tensor (batch_size=1, channels=3, height=256, width=256)
dummy_input = torch.randn(1, 3, 256, 256)

# Forward pass
try:
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: torch.Size([1, 1, 256, 256])
    print("Model runs successfully!")
except Exception as e:
    print("Error during forward pass:", str(e))