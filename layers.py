from torchvision.models import vgg19_bn, vgg19
import torch.nn as nn
import torch
import torch.nn.functional as F

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 dilation=1, bias=False, activate=True):
        super().__init__()
        self.activate = activate
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.activate else x


class ASPP(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ASPP, self).__init__()

    self.mean = nn.AdaptiveAvgPool2d((1,1))
    self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
    self.atrous_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
    self.atrous_6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6) # rate = 6
    self.atrous_12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12) # rate = 12
    self.atrous_18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18) # rate = 18
    self.conv_1x1 = nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)


  def forward(self, x):
    image_features = self.mean(x)
    image_features = self.conv(image_features)
    image_features = F.interpolate(image_features, size=x.size()[2:], mode="bilinear", align_corners=True)
    atrous_1 = self.atrous_1(x)
    atrous_2 = self.atrous_6(x)
    atrous_3 = self.atrous_12(x)
    atrous_4 = self.atrous_18(x)
    return self.conv_1x1(torch.cat([image_features, atrous_1, atrous_2,
                                       atrous_3, atrous_4], axis=1))

class SELayer(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SELayer, self).__init__()
        self.sequeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(in_channel, in_channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reduction, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.sequeeze(x).view(batch_size, channel_size)
        y = self.excite(y).view(batch_size, channel_size, 1, 1)
        return x*y.expand_as(x)


class ConvBlock(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    self.doubleconv = nn.Sequential(
        Conv2D(in_channel, out_channel, kernel_size=3, padding=1, activate=True, bias=False),
        Conv2D(out_channel, out_channel, kernel_size=3, padding=1, activate=True, bias=False),
        SELayer(out_channel)
    )

  def forward(self, x):
    return self.doubleconv(x)


class Stage1Encoder(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    vgg = vgg19_bn(pretrained=True)
    self.x1 = vgg.features[0:6] # channels: in -> 3 & out -> 64
    self.x2 = vgg.features[6:13] # channels: in -> 64 & out -> 128
    self.x3 = vgg.features[13:26] # channels: in -> 128 & out -> 256
    self.x4 = vgg.features[26:39] # channels: in -> 256 & out -> 512
    self.x5 = vgg.features[39:52] # channels: in -> 512 & out -> 512
    self.skip_channels = [512, 256, 128, 64]
  def forward(self, x):
    x1 = self.x1(x)
    x2 = self.x2(x1)
    x3 = self.x3(x2)
    x4 = self.x4(x3)
    x5 = self.x5(x4)
    return x5, [x4, x3, x2, x1]

class Stage1Decoder(nn.Module):
  def __init__(self, in_channel, out_channel, skip_channels):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    self.c1 = ConvBlock(in_channel + skip_channels[0], 256) #skip_channel[0]= 512
    self.c2 = ConvBlock(256 + skip_channels[1], 128) #skip_channel[1] = 256
    self.c3 = ConvBlock(128 + skip_channels[2], 64) # skip channel[2] = 128
    self.c4 = ConvBlock(64 + skip_channels[3], out_channel) # skip_channel[3] = 64
    self.convs = [self.c1, self.c2, self.c3, self.c4]

  def forward(self, x, skip):
    try:
      for i in range(4):
        x = self.up(x)
        x = torch.cat([x, skip[i]], axis=1)
        x = self.convs[i](x)
    except Exception as e:
      print(e)
      print(f"input: {x.shape}, skip:{skip[i].shape}")
      exit()
    return x


class Stage2Encoder(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    self.pool = nn.MaxPool2d((2,2))
    self.e1 = ConvBlock(in_channel, 32)
    self.e2 = ConvBlock(32, 64)
    self.e3 = ConvBlock(64, 128)
    self.e4 = ConvBlock(128, out_channel)
    self.skip_channels = [out_channel, 128, 64, 32]
  def forward(self, x):
    x1 = self.e1(x)
    p1 = self.pool(x1)

    x2 = self.e2(p1)
    p2 = self.pool(x2)

    x3 = self.e3(p2)
    p3 = self.pool(x3)

    x4 = self.e4(p3)
    p4 = self.pool(x4)
    return p4, [x4, x3, x2, x1]

class Stage2Decoder(nn.Module):
  def __init__(self, in_channel, out_channel, stage1_skip_channels, stage2_skip_channels):
    super().__init__()
    assert len(stage1_skip_channels) == len(stage2_skip_channels)
    combined_channel = [stage1_skip_channels[i] + stage2_skip_channels[i] for i in range(len(stage1_skip_channels))]
    self.combined_channel = combined_channel
    self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    self.c1 = ConvBlock(in_channel + combined_channel[0], 256) #combined_channel[0]= 512 + 256
    self.c2 = ConvBlock(256 + combined_channel[1], 128) #combined_channel[1] = 256 + 128
    self.c3 = ConvBlock(128 + combined_channel[2], 64) # combined_channel[2] = 128 + 64
    self.c4 = ConvBlock(64 + combined_channel[3], out_channel) # combined_channel[3] = 64 + 32
    self.convs = [self.c1, self.c2, self.c3, self.c4]

  def forward(self, x, stage1_skip, stage2_skip):
    try:
      for i in range(4):
        x = self.up(x)
        x = torch.cat([x, stage1_skip[i], stage2_skip[i]], axis=1)
        x = self.convs[i](x)
    except Exception as e:
      print(e)
      print(f"input: {x.shape}, skip1:{stage1_skip[i].shape}, skip2:{stage2_skip[i].shape}, actual:{self.combined_channel[i]}")
      exit()
    return x
