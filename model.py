import torch
import torch.nn as nn
from torch.autograd import Variable


class CustomUNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    #3 downsampling blocks
    self.conv1 = self.downsample_block(in_channels, 32, 7, 3)
    self.conv2 = self.downsample_block(32, 64, 3, 1)
    self.conv3 = self.downsample_block(64, 128, 3, 1)

    #3 upsampling blocks
    self.upconv3 = self.upsample_block(128, 64, 3, 1)
    self.upconv2 = self.upsample_block(64*2, 32, 3, 1)
    self.upconv1 = self.upsample_block(32*2, out_channels, 3, 1)
  
  def __call__(self, x):
    
    # downsampling part
    conv1 = self.conv1(x)
    conv2 = self.conv2(conv1)
    conv3 = self.conv3(conv2)

    #upsampling part
    upconv3 = self.upconv3(conv3)
    upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
    upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
    
    return upconv1

  def downsample_block(self, in_channels, out_channels, kernel_size, padding):
    downsample_block = nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    return downsample_block
  
  def upsample_block(self, in_channels, out_channels, kernel_size, padding):
    upsample_block = nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    )
    return upsample_block