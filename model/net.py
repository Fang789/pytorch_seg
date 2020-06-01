import torch
import torch.nn as nn
import torch.nn.functional as F

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

def conv_bn1X1(inp, oup, stride, leaky=0):
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
	def __init__(self, pyramid_channels, skip_channels):
		super().__init__()
		self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

	def forward(self, x, skip=None):
		x = F.interpolate(x,size=[skip.size(2),skip.size(3)], mode="nearest")
		skip = self.skip_conv(skip)
		x = x + skip
		return x

class PSPBlock(nn.Module):

	def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
		super().__init__()
		if pool_size == 1:
			use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
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
			PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
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
			use_bathcnorm=use_batchnorm,
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

	def forward(self, x):
		x = self.block(x)
		if self.upsample:
			x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
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
