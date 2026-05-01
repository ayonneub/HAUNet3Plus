import torch
import torch.nn as nn
import torch.nn.functional as F



# Utilities


def init_weights(m, init_type='kaiming'):
    if init_type == 'kaiming':
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# Basic double conv block used in encoder
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size) if is_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size) if is_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        if dropout_p and dropout_p > 0:
            layers.insert(3, nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



# Attention Modules


class ChannelAttention(nn.Module):
    """
    Global Average Pool -> FC -> ReLU -> FC -> Sigmoid, then channel reweight.
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        hidden = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc2(self.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """
    AvgPool(channel) + MaxPool(channel) -> concat -> 7x7 conv -> Sigmoid -> spatial reweight.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class UncertaintyEstimator(nn.Module):
    """
    Lightweight module to estimate per-pixel uncertainty using 1x1 conv.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))  # Output: [B, 1, H, W]


class HybridAttention(nn.Module):
    """
    Uncertainty-Guided Hybrid Attention: Channel + Spatial attention modulated by uncertainty map.
    """
    def __init__(self, in_channels, ratio=16, spatial_kernel=7, unc_alpha_init=0.1, unc_beta_init=0.5):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)
        self.unc_estimator = UncertaintyEstimator(in_channels)
        self.alpha = nn.Parameter(torch.tensor(unc_alpha_init))  # Learnable modulation for CA
        self.beta = nn.Parameter(torch.tensor(unc_beta_init))   # Learnable modulation for SA

    def forward(self, x):
        # Compute uncertainty map
        u = self.unc_estimator(x)  # [B, 1, H, W]

        # Channel Attention with uncertainty modulation
        x_ca = self.ca(x)
        u_avg = torch.mean(u, dim=[2, 3], keepdim=True)  # Global avg pool of uncertainty [B, 1, 1, 1]
        x_ca = x_ca + self.alpha * u_avg.expand_as(x_ca)  # Modulate CA weights

        # Spatial Attention with uncertainty modulation
        x_sa = self.sa(x_ca)
        x_sa = x_sa * (1 + self.beta * u)  # Element-wise modulation

        return x_sa, u  # Return attended features and uncertainty map



# UNet 3+ with Deep Supervision + Uncertainty-Guided Hybrid Attention


class UNet_3Plus_DeepSup_AU(nn.Module):
    """
    UNet 3+ backbone with deep supervision heads (d1..d5).
    Uncertainty-Guided HybridAttention is applied to every incoming skip feature at each decoder stage
    BEFORE concatenation. Returns segmentation outputs and uncertainty maps.
    """
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        is_batchnorm=True,
        dropout_p=0.0,
        att_ratio=16,
        feature_scale=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.dropout_p = dropout_p
        self.att_ratio = att_ratio

        # Encoder filters (classic UNet: 64,128,256,512,1024)
        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.conv1 = unetConv2(in_channels, filters[0], is_batchnorm, dropout_p=dropout_p)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], is_batchnorm, dropout_p=dropout_p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], is_batchnorm, dropout_p=dropout_p)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], is_batchnorm, dropout_p=dropout_p)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], is_batchnorm, dropout_p=dropout_p)

        # UNet 3+ Multi-scale Decoder Setup 
        self.CatChannels = filters[0]     # after projecting every input to same #channels
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks  # 64 * 5 = 320

        # stage 4d (target spatial 1/8)
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        self.ha4 = nn.ModuleList([HybridAttention(self.CatChannels, ratio=att_ratio, unc_alpha_init=0.1, unc_beta_init=0.5) for _ in range(5)])

        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        # stage 3d (target 1/4)
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        self.ha3 = nn.ModuleList([HybridAttention(self.CatChannels, ratio=att_ratio, unc_alpha_init=0.1, unc_beta_init=0.5) for _ in range(5)])

        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # stage 2d (target 1/2)
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        self.ha2 = nn.ModuleList([HybridAttention(self.CatChannels, ratio=att_ratio, unc_alpha_init=0.1, unc_beta_init=0.5) for _ in range(5)])

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # stage 1d (target 1/1)
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        self.ha1 = nn.ModuleList([HybridAttention(self.CatChannels, ratio=att_ratio, unc_alpha_init=0.1, unc_beta_init=0.5) for _ in range(5)])

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # Deep Supervision Heads 
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

        # Upsampling for deep supervision to full scale (x1)
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # init
        self.apply(init_weights)

    def forward(self, x):
        # Encoder 
        h1 = self.conv1(x)                   # 1/1
        h2 = self.conv2(self.maxpool1(h1))   # 1/2
        h3 = self.conv3(self.maxpool2(h2))   # 1/4
        h4 = self.conv4(self.maxpool3(h3))   # 1/8
        hd5 = self.conv5(self.maxpool4(h4))  # 1/16

        # Decoder Stage 4d (1/8) 
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))

        # Apply uncertainty-guided attention, collect uncertainty maps
        h1_PT_hd4, u1_4 = self.ha4[0](h1_PT_hd4)
        h2_PT_hd4, u2_4 = self.ha4[1](h2_PT_hd4)
        h3_PT_hd4, u3_4 = self.ha4[2](h3_PT_hd4)
        h4_Cat_hd4, u4_4 = self.ha4[3](h4_Cat_hd4)
        hd5_UT_hd4, u5_4 = self.ha4[4](hd5_UT_hd4)

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), dim=1)
        )))

        # Decoder Stage 3d (1/4) 
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h3_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))

        h1_PT_hd3, u1_3 = self.ha3[0](h1_PT_hd3)
        h2_PT_hd3, u2_3 = self.ha3[1](h2_PT_hd3)
        h3_Cat_hd3, u3_3 = self.ha3[2](h3_Cat_hd3)
        hd4_UT_hd3, u4_3 = self.ha3[3](hd4_UT_hd3)
        hd5_UT_hd3, u5_3 = self.ha3[4](hd5_UT_hd3)

        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), dim=1)
        )))

        #Decoder Stage 2d (1/2)
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))

        h1_PT_hd2, u1_2 = self.ha2[0](h1_PT_hd2)
        h2_Cat_hd2, u2_2 = self.ha2[1](h2_Cat_hd2)
        hd3_UT_hd2, u3_2 = self.ha2[2](hd3_UT_hd2)
        hd4_UT_hd2, u4_2 = self.ha2[3](hd4_UT_hd2)
        hd5_UT_hd2, u5_2 = self.ha2[4](hd5_UT_hd2)

        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), dim=1)
        )))

        #  Decoder Stage 1d (1/1)
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))

        h1_Cat_hd1, u1_1 = self.ha1[0](h1_Cat_hd1)
        hd2_UT_hd1, u2_1 = self.ha1[1](hd2_UT_hd1)
        hd3_UT_hd1, u3_1 = self.ha1[2](hd3_UT_hd1)
        hd4_UT_hd1, u4_1 = self.ha1[3](hd4_UT_hd1)
        hd5_UT_hd1, u5_1 = self.ha1[4](hd5_UT_hd1)

        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), dim=1)
        )))

        # Deep Supervision logits -> upsample to full res 
        d5 = self.upscore5(self.outconv5(hd5))
        d4 = self.upscore4(self.outconv4(hd4))
        d3 = self.upscore3(self.outconv3(hd3))
        d2 = self.upscore2(self.outconv2(hd2))
        d1 = self.outconv1(hd1)

        # Collect uncertainty maps from all stages
        uncertainties = [
            [u1_4, u2_4, u3_4, u4_4, u5_4],  # Stage 4d
            [u1_3, u2_3, u3_3, u4_3, u5_3],  # Stage 3d
            [u1_2, u2_2, u3_2, u4_2, u5_2],  # Stage 2d
            [u1_1, u2_1, u3_1, u4_1, u5_1],  # Stage 1d
        ]

        # Return probabilities (sigmoid) for binary and uncertainty maps
        return (
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
        ), uncertainties



# Loss: CE + Dice with Uncertainty Regularization


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    L = alpha * BCE + (1 - alpha) * Dice
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + (1.0 - self.alpha) * self.dice(pred, target)


class DeepSupervisionLossUncertainty(nn.Module):
    """
    Applies CombinedLoss to all deep supervision outputs with weights.
    Adds uncertainty regularization to prevent trivial uncertainty maps.
    """
    def __init__(self, alpha=0.5, ds_weights=(1.0, 0.8, 0.6, 0.4, 0.2), unc_reg_weight=0.01):
        super().__init__()
        self.base = CombinedLoss(alpha=alpha)
        self.weights = ds_weights
        self.unc_reg_weight = unc_reg_weight

    def forward(self, outputs, target, uncertainties=None):
        assert len(outputs) == 5, "Expecting 5 deep supervision outputs"
        total = 0.0
        wsum = 0.0
        for w, out in zip(self.weights, outputs):
            total = total + w * self.base(out, target)
            wsum += w

        # Add uncertainty regularization if provided
        if uncertainties is not None:
            unc_loss = 0.0
            for stage_uncs in uncertainties:  # Each stage has 5 uncertainty maps
                for u in stage_uncs:
                    unc_loss += torch.mean(u)  # Encourage non-trivial uncertainty
            unc_loss = unc_loss / (len(uncertainties) * 5)  # Average over all maps
            total += self.unc_reg_weight * unc_loss

        return total / wsum