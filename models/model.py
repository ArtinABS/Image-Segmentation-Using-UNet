import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group", gn_groups: int = 8, p_drop: float = 0.0):
        super().__init__()
        Norm = {
            "batch": lambda c: nn.BatchNorm2d(c),
            "instance": lambda c: nn.InstanceNorm2d(c, affine=True),
            "group": lambda c: nn.GroupNorm(num_groups=min(gn_groups, c), num_channels=c),
        }[norm]
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.n1 = Norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.n2 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.n1(self.conv1(x)))
        # x = self.drop(x)
        x = self.relu(self.n2(self.conv2(x)))
        return x


class Down(nn.Module):
    """Downscale with MaxPool then a ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)


    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Up(nn.Module):
    """Upscale, then concat with skip, then a ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int, use_transpose: bool = True):
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            # Bilinear upsample + 1×1 conv to reduce channels
            self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3)
            )
        self.conv = ConvBlock(in_channels, out_channels)


    def forward(self, x, skip):
        x = self.up(x)
        # Handle possible misalignments due to odd sizes
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 9, base_c: int = 64, use_transpose: bool = True):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_c) # 64
        self.enc2 = Down(base_c, base_c * 2) # 128
        self.enc3 = Down(base_c * 2, base_c * 4) # 256
        self.enc4 = Down(base_c * 4, base_c * 8) # 512

        # Bottleneck
        self.bottleneck = Down(base_c * 8, base_c * 16) # 1024

        # Decoder
        self.up4 = Up(base_c * 16, base_c * 8, use_transpose) # 1024→512
        self.up3 = Up(base_c * 8, base_c * 4, use_transpose) # 512→256
        self.up2 = Up(base_c * 4, base_c * 2, use_transpose) # 256→128
        self.up1 = Up(base_c * 2, base_c, use_transpose) # 128→64
        
        # Head
        self.head = nn.Conv2d(base_c, num_classes, kernel_size=1)


        self.apply(self._init_weights)


    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        # Decoder with skip connections
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        logits = self.head(d1) # (N, 9, H, W)
        return logits
    

if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=9)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output:", y.shape) # Expect: (2, 9, 256, 256)
    total_params = sum(p.numel() for p in model.parameters())
    print("Params:", total_params)