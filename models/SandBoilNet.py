import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Set seed for reproducibility
torch.manual_seed(42)

class PCALayer(nn.Module):
    def __init__(self, n_components):
        super(PCALayer, self).__init__()
        self.n_components = n_components
        self.kernel = None

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        input_dim = channels
        flattened = x.view(batch_size, input_dim, -1).transpose(1, 2)  # [batch, pixels, channels]

        # Compute mean and center the data
        mean = flattened.mean(dim=1, keepdim=True)
        centered = flattened - mean

        # Compute covariance matrix
        cov = torch.bmm(centered.transpose(1, 2), centered) / (flattened.size(1) - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='U')

        # Sort eigenvectors by eigenvalues in descending order
        _, indices = eigenvalues.sort(dim=-1, descending=True)
        top_eigenvectors = torch.gather(
            eigenvectors, dim=-1, 
            index=indices[:, :, :self.n_components].expand(batch_size, eigenvectors.size(1), self.n_components)
        )

        # Project centered data onto top principal components
        projected = torch.bmm(centered, top_eigenvectors)

        # Reshape to match input spatial dimensions
        output = projected.view(batch_size, height, width, self.n_components).permute(0, 3, 1, 2)
        return output

    def _initialize_weights(self, input_dim):
        self.kernel = nn.Parameter(
            torch.empty(input_dim, self.n_components).normal_(std=0.02),
            requires_grad=False
        )

class SpatialPoolingBlock(nn.Module):
    def __init__(self, ratio=4):
        super(SpatialPoolingBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        filters = x.size(1)
        se_shape = (filters, 1, 1)

        # Spatial pyramid pooling branches
        spp_1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        spp_1 = nn.AdaptiveMaxPool2d(1)(spp_1).view(-1, filters, 1, 1)
        spp_1 = nn.Conv2d(filters, filters, kernel_size=1, bias=True)(spp_1)
        spp_1 = F.relu(spp_1)

        spp_2 = F.max_pool2d(x, kernel_size=4, stride=4, padding=2)
        spp_2 = nn.AdaptiveMaxPool2d(1)(spp_2).view(-1, filters, 1, 1)
        spp_2 = nn.Conv2d(filters, filters, kernel_size=1, bias=True)(spp_2)
        spp_2 = F.relu(spp_2)

        spp_3 = F.max_pool2d(x, kernel_size=8, stride=8, padding=4)
        spp_3 = nn.AdaptiveMaxPool2d(1)(spp_3).view(-1, filters, 1, 1)
        spp_3 = nn.Conv2d(filters, filters, kernel_size=1, bias=True)(spp_3)
        spp_3 = F.relu(spp_3)

        # Combine features
        feature = spp_1 + spp_2 + spp_3
        feature = torch.sigmoid(feature)
        return x * feature

class AttentionBlock(nn.Module):
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.spatial_pooling = SpatialPoolingBlock()

    def forward(self, x):
        channel_attention = self.spatial_pooling(x)
        spatial_attention = nn.Conv2d(x.size(1), 1, kernel_size=1, padding=0, bias=True)(channel_attention)
        spatial_attention = torch.sigmoid(spatial_attention)
        channel_attention = channel_attention * spatial_attention
        return x + channel_attention

class Conv2DBN(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, padding='same'):
        super(Conv2DBN, self).__init__()
        padding_value = kernel_size // 2 if padding == 'same' else 0
        self.conv = nn.Conv2d(in_channels, filters, kernel_size=kernel_size, padding=padding_value, bias=True)
        self.gn = nn.GroupNorm(num_groups=filters, num_channels=filters)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x

class IterLBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IterLBlock, self).__init__()
        filters_1x1 = filters // 8
        filters_3x3 = filters // 2
        filters_5x5 = filters // 4
        filters_pool_proj = filters // 8

        self.conv1x1 = Conv2DBN(in_channels, filters_1x1, 1)
        self.conv3x3 = nn.Sequential(
            Conv2DBN(in_channels, filters_3x3, 3),
            Conv2DBN(filters_3x3, filters_3x3, 3)
        )
        self.conv5x5 = nn.Sequential(
            Conv2DBN(in_channels, filters_5x5, 5),
            Conv2DBN(filters_5x5, filters_5x5, 5)
        )
        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2DBN(in_channels, filters_pool_proj, 1)
        )
        self.gn = nn.GroupNorm(num_groups=filters, num_channels=filters)
        self.act = nn.LeakyReLU(0.02)

    def forward(self, x):
        conv_1x1 = self.conv1x1(x)
        conv_3x3 = self.conv3x3(x)
        conv_5x5 = self.conv5x5(x)
        pool_proj = self.pool_proj(x)
        output = torch.cat([conv_1x1, conv_3x3, conv_5x5, pool_proj], dim=1)
        output = self.gn(output)
        output = self.act(output)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.iterL = IterLBlock(in_channels, filters)
        self.attention = AttentionBlock()

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.iterL(x)
        x = self.attention(x)
        return x

class SandboilNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, input_filters=32, height=256, width=256):
        super(SandboilNet, self).__init__()
        self.filters = input_filters
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Load pretrained ResNet50
        self.base_model = models.resnet50(weights='IMAGENET1K_V2')
        self.base_model.eval()

        # Freeze layers and set BatchNorm to eval mode
        for param in self.base_model.parameters():
            param.requires_grad = False
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        # Unfreeze last layers (approximating the last 48 layers in TensorFlow)
        unfreeze_layers = ['layer4', 'layer3']
        for name, param in self.base_model.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True

        # Encoder
        self.initial_conv = Conv2DBN(64, self.filters, 3)  # Accepts 64 channels from ResNet conv1
        self.iterL_s11 = IterLBlock(self.filters, self.filters)
        self.attention_s11 = AttentionBlock()

        self.pca2 = PCALayer(32)
        self.attention_s21 = AttentionBlock()
        self.iterL_s21 = IterLBlock(256, self.filters)  # ResNet layer1 outputs 256 channels

        self.pca3 = PCALayer(32)
        self.attention_s31 = AttentionBlock()
        self.iterL_s31 = IterLBlock(512, self.filters)  # ResNet layer2 outputs 512 channels

        self.pca4 = PCALayer(64)
        self.attention_s41 = AttentionBlock()
        self.iterL_s41 = IterLBlock(1024, self.filters * 2)  # ResNet layer3 outputs 1024 channels

        # Bridge
        self.pca_b11 = PCALayer(128)
        self.attention_b11 = AttentionBlock()
        self.iterL_b11 = IterLBlock(2048, self.filters * 4)  # ResNet layer4 outputs 2048 channels

        # Decoder
        self.decoder1 = DecoderBlock(self.filters * 4 + 1024, self.filters * 4)
        self.decoder2 = DecoderBlock(self.filters * 4 + 512, self.filters * 2)
        self.decoder3 = DecoderBlock(self.filters * 2 + 256, self.filters)
        self.decoder4 = DecoderBlock(self.filters + self.filters, self.filters // 2)

        # Output
        self.output_conv = nn.Conv2d(self.filters // 2, self.n_classes, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        s11 = self.base_model.conv1(x)
        s11 = self.base_model.bn1(s11)
        s11 = self.base_model.relu(s11)
        s11 = self.initial_conv(s11)
        s11 = self.iterL_s11(s11)
        s11 = self.attention_s11(s11)

        s21 = self.base_model.maxpool(s11)
        s21 = self.base_model.layer1(s21)
        pca2 = self.pca2(s21)
        s21 = self.attention_s21(s21)
        s21 = self.iterL_s21(s21)
        s21 = s21 + pca2

        s31 = self.base_model.layer2(s21)
        pca3 = self.pca3(s31)
        s31 = self.attention_s31(s31)
        s31 = self.iterL_s31(s31)
        s31 = s31 + pca3

        s41 = self.base_model.layer3(s31)
        pca4 = self.pca4(s41)
        s41 = self.attention_s41(s41)
        s41 = self.iterL_s41(s41)
        s41 = s41 + pca4

        # Bridge
        b11 = self.base_model.layer4(s41)
        pcb11 = self.pca_b11(b11)
        b11 = self.attention_b11(b11)
        b11 = self.iterL_b11(b11)
        b11 = b11 + pcb11

        # Decoder
        d11 = self.decoder1(b11, s41)
        d21 = self.decoder2(d11, s31)
        d31 = self.decoder3(d21, s21)
        d41 = self.decoder4(d31, s11)

        # Output
        outputs = self.output_conv(d41)
        outputs = self.sigmoid(outputs)
        return outputs