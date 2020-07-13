import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model.jpu import SeparableConv2d

def conv_bn(inp, oup, stride = 1, leaky = 0):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		#nn.GroupNorm(32,oup),
		#nn.ReLU(oup)
		nn.BatchNorm2d(oup),
		nn.LeakyReLU(negative_slope=leaky, inplace=False)
		#nn.ReLU(oup)
	)

def conv_bn_no_relu(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		#nn.GroupNorm(32,oup),
		nn.BatchNorm2d(oup),
	)

def conv_bn1X1(inp, oup, stride=1, leaky=0):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
		#nn.GroupNorm(32,oup),
		nn.BatchNorm2d(oup),
		#nn.ReLU(oup)
		nn.LeakyReLU(negative_slope=leaky, inplace=False)
	)

class SSH(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(SSH, self).__init__()
		assert out_channel % 4 == 0
		leaky = 0
		if (out_channel <= 64):
			leaky = 0.1
		self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

		self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
		self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

		self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
		self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

	def forward(self, input):
		conv3X3 = self.conv3X3(input)

		conv5X5_1 = self.conv5X5_1(input)
		conv5X5 = self.conv5X5_2(conv5X5_1)

		conv7X7_2 = self.conv7X7_2(conv5X5_1)
		conv7X7 = self.conv7x7_3(conv7X7_2)

		out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
		out = F.relu(out)
		return out

class FPNBlock(nn.Module):
	def __init__(self, pyramid_channels, skip_channels): #sk是输入
		super().__init__()
		self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

	def forward(self, x, skip=None):
		x = F.interpolate(x,size=[skip.size(2),skip.size(3)], mode="nearest")
		skip = self.skip_conv(skip)
		x = x + skip
		return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.sharedMLP = nn.Sequential(
			nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
			nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avgout = self.sharedMLP(self.avg_pool(x))
		maxout = self.sharedMLP(self.max_pool(x))
		return self.sigmoid(avgout + maxout)	

class PSPBlock(nn.Module):

	def __init__(self, in_channels, out_channels, pool_size, use_batchnorm=True):
		super().__init__()
		if pool_size == 1:
			use_batchnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
		self.pool = nn.Sequential(
			nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
			conv_bn(in_channels, out_channels, stride=1, leaky = 0)
		)

	def forward(self, x):
		h, w = x.size(2), x.size(3)
		x = self.pool(x)
		x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
		return x


class PSPModule(nn.Module):
	def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
		super().__init__()

		self.blocks = nn.ModuleList([
			PSPBlock(in_channels, in_channels // len(sizes), size, use_batchnorm=use_batchnorm) for size in sizes
		])

	def forward(self, x):
		xs = [block(x) for block in self.blocks] + [x]
		x = torch.cat(xs, dim=1)
		return x


class PSPHead(nn.Module):

	def __init__(
			self,
			in_channels,
			use_batchnorm=True,
			out_channels=512,
			dropout=0.2,
	):
		super().__init__()

		self.psp = PSPModule(
			in_channels=in_channels,
			sizes=(1, 2, 3, 6),
			use_batchnorm=use_batchnorm,
		)

		self.conv = conv_bn(in_channels*2, out_channels, stride=1, leaky = 0)
		if dropout!=None:
			self.dropout = nn.Dropout2d(p=dropout)
		else:
			self.dropout=None

	def forward(self, *features):
		x = features[-1]
		x = self.psp(x)
		x = self.conv(x)
		if self.dropout!=None:
			x = self.dropout(x)

		return x

class Conv3x3GNReLU(nn.Module):
	def __init__(self, in_channels, out_channels, upsample=False):
		super().__init__()
		self.upsample = upsample
		self.block = nn.Sequential(
			nn.Conv2d(
				in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
			),
			nn.GroupNorm(32, out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x,skip=None):
		x = self.block(x)
		if self.upsample:
			x = F.interpolate(x,size=(skip.size(2),skip.size(3)), mode="bilinear",align_corners=True)
		return x

class SegmentationBlock(nn.Module):
	def __init__(self, in_channels, out_channels, n_upsamples=0):
		super().__init__()

		blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

		if n_upsamples > 1:
			for _ in range(1, n_upsamples):
				blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

		self.block = nn.Sequential(*blocks)

	def forward(self, x):
		return self.block(x)

class GlobalAvgPooling(nn.Module):
	def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
		super(GlobalAvgPooling, self).__init__()
		self.gap = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			norm_layer(out_channels),
			nn.ReLU(True)
		)
	def forward(self, x):
		size = x.size()[2:]
		pool = self.gap(x)
		out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
		return out

class ASPP(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		kernel_sizes = [1, 3, 3, 1]
		dilations = [1, 3, 6, 1]
		paddings = [0, 3, 6, 0]
		self.aspp = torch.nn.ModuleList()
		for aspp_idx in range(len(kernel_sizes)):
			conv = torch.nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=kernel_sizes[aspp_idx],
				stride=1,
				dilation=dilations[aspp_idx],
				padding=paddings[aspp_idx],
				bias=True)
			self.aspp.append(conv)
		self.gap = torch.nn.AdaptiveAvgPool2d(1)
		self.aspp_num = len(kernel_sizes)
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.fill_(0)

	def forward(self, x):
		avg_x = self.gap(x)
		out = []
		for aspp_idx in range(self.aspp_num):
			inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
			out.append(F.relu_(self.aspp[aspp_idx](inp)))
		out[-1] = out[-1].expand_as(out[-2])
		out = torch.cat(out, dim=1)
		return out
