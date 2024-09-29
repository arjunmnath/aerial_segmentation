import torch.nn as nn
from layers import Stage1Encoder, ASPP, Stage1Decoder, Stage2Encoder, Stage2Decoder

class DoubleUnet(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    self.decoder_output_channels = 32
    self.aspp_output_channels = 64
    self.stage1_encoder_channels = 512
    self.stage2_encoder_channels = 256
    self.stage1_encoder = Stage1Encoder(in_channel, self.stage1_encoder_channels)
    self.stage1_aspp = ASPP(self.stage1_encoder_channels, self.aspp_output_channels)
    self.stage1_decoder = Stage1Decoder(self.aspp_output_channels,
                                        self.decoder_output_channels,
                                        self.stage1_encoder.skip_channels)
    self.y1 = nn.Conv2d(self.decoder_output_channels, out_channel, kernel_size=1, padding=0)
    self.sigmoid = nn.Sigmoid()
    self.channel_correction = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    self.stage2_encoder = Stage2Encoder(out_channel, self.stage2_encoder_channels)
    self.stage2_aspp = ASPP(self.stage2_encoder_channels, self.aspp_output_channels)
    self.stage2_decoder = Stage2Decoder(self.aspp_output_channels, self.decoder_output_channels,
                                      self.stage1_encoder.skip_channels,
                                        self.stage2_encoder.skip_channels)
    self.y2 = nn.Conv2d(self.decoder_output_channels, out_channel, kernel_size=1, padding=0)

  def forward(self, x):
    x0 = x
    x , stage1_skip = self.stage1_encoder(x)
    x = self.stage1_aspp(x)
    x = self.stage1_decoder(x, stage1_skip)
    y1 = self.y1(x)
    
    x0 = self.channel_correction(x0) 
    stage2_x = x0 * self.sigmoid(y1)
    x, stage2_skip = self.stage2_encoder(stage2_x)
    x = self.stage2_aspp(x)
    x = self.stage2_decoder(x, stage1_skip, stage2_skip)

    y2 = self.y2(x)
    return y1, y2
