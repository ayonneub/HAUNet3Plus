import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A convolutional block with two Conv2d-BatchNorm-ReLU sequences."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    """Encoder block with a ConvBlock followed by max pooling."""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)  # Skip connection for decoder
        down = self.pool(skip)  # Downsampled feature map
        return skip, down

class AttentionGate(nn.Module):
    """Attention gate to focus on relevant features from skip connections."""
    def __init__(self, g_channels, s_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        g1 = self.Wg(g)  # Process gating signal
        s1 = self.Ws(s)  # Process skip connection
        out = F.relu(g1 + s1)  # Element-wise addition and activation
        psi = self.psi(out)  # Attention coefficients
        return s * psi  # Scale skip connection

class DecoderBlock(nn.Module):
    """Decoder block with upsampling, attention gate, and ConvBlock."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.att = AttentionGate(in_channels, skip_channels, out_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # Upsample
        skip = self.att(x, skip)  # Apply attention to skip connection
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return self.conv(x)

class AttentionUNet(nn.Module):
    """Attention U-Net architecture for image segmentation."""
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet, self).__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
 
        # Decoder
        self.dec1 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        s1, p1 = self.enc1(x)  # Skip connection and pooled output
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        
        # Bottleneck
        b1 = self.bottleneck(p3)
        
        # Decoder path
        d1 = self.dec1(b1, s3)
        d2 = self.dec2(d1, s2)
        d3 = self.dec3(d2, s1)

        # Final output
        return torch.sigmoid(self.final_conv(d3))

