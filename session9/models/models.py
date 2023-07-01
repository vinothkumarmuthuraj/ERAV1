import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1,groups=1,bias=True,dilation=1,padding_mode='replicate'):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                        stride=stride,padding=padding,groups=groups,bias=bias,padding_mode=padding_mode,
                                        dilation=dilation)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class PointwiseConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,bias=True):
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1, 1),
                                        stride=1,padding=0,bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SeparableConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True,padding_mode='replicate'):
        super().__init__()

        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels,out_channels=in_channels, kernel_size=kernel_size,
                                              stride=stride,padding=padding,groups=in_channels,
                                              bias=bias,padding_mode=padding_mode, dilation=dilation)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels,out_channels=out_channels,bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class norm_conv_fn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,bias):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class atrous_conv(nn.Module):
    def __init__(self,in_channels,dilation_out_channels,kernel_size,dilation_stride,dilation,padding,bias):
        super().__init__()

        self.dilated_conv = nn.Conv2d(in_channels=in_channels, out_channels=dilation_out_channels,
                                      kernel_size=kernel_size, stride=dilation_stride, dilation=dilation, bias=bias,
                                      padding=padding, padding_mode="replicate")
        self.batch_norm = nn.BatchNorm2d(dilation_out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dilated_conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class Conv_block(nn.Module):
    def __init__(self, in_channels, in_channel3, out_channel1, out_channel2, out_channel3, kernel_size, stride=1, padding=1,
                            dilation1=1, dilation2=1, dilation3=1,bias=True, block_name="block1"):
        super().__init__()
        self.block_name = block_name
        self.normal_conv = norm_conv_fn(in_channels=in_channels, out_channels=out_channel1, kernel_size=kernel_size,
                                        stride=stride,padding=padding,bias=bias)

        self.depthwise_sep_conv1 = SeparableConv2D(in_channels, out_channel1, kernel_size, stride=stride,
                                                      padding=padding, bias=True, dilation=dilation1)

        self.depthwise_sep_conv2 = SeparableConv2D(out_channel1, out_channel2, kernel_size, stride=stride,
                                                   padding=padding, bias=True, dilation=dilation2)
        self.depthwise_sep_conv3 = SeparableConv2D(in_channel3, out_channel3, kernel_size, stride=stride,
                                                   padding=padding, bias=True, dilation=dilation3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_name == "block1":
            x = self.normal_conv(x)
        else:
            x = self.depthwise_sep_conv1(x)
        x = torch.cat([x,self.depthwise_sep_conv2(x)],1)
        x = self.depthwise_sep_conv3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = Conv_block(in_channels=3, in_channel3=96, out_channel1=32, out_channel2=64, out_channel3=128,kernel_size=3,
                                      stride=1, padding=1, dilation1=2, dilation2=1,dilation3=3,bias=False,
                                      block_name="block1")
        self.conv_block2 = Conv_block(in_channels=128, in_channel3=96,out_channel1=32, out_channel2=64,out_channel3=128, kernel_size=3,
                                      stride=1, padding=1, dilation1=2,dilation2=1,dilation3=3, bias=False,
                                      block_name="block2")
        self.conv_block3 = Conv_block(in_channels=128, in_channel3=96, out_channel1=32,out_channel2=64,out_channel3=128, kernel_size=3,
                                      stride=1, padding=1, dilation1=2, dilation2=1,dilation3=3,bias=False,
                                      block_name="block3")
        self.conv_block4 = Conv_block(in_channels=128, in_channel3=96, out_channel1=32,out_channel2=64,out_channel3=64, kernel_size=3,
                                      stride=1, padding=1, dilation1=2,dilation2=1,dilation3=3, bias=False,
                                      block_name="block4")
        self.avg_pool = nn.AvgPool2d(kernel_size=10, stride=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output








