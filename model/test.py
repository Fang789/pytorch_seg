import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.net import FPNBlock,PSPHead,Conv3x3GNReLU,SSH,SpatialAttention
from model.jpu import JPU, SeparableConv2d

class FCB(nn.Module):
	def __init__(self,in_channel,out_channel,dropprob,dilate_rate):
		super().__init__()

		self.conv3x1 = nn.Conv2d(in_channel//2, out_channel//2, (3, 1), stride=1, padding=(0,1), bias=False)
		self.conv1x3 = nn.Conv2d(in_channel//2, out_channel//2, (1, 3), stride=1, padding=(1,0), bias=False)
		self.conv1x1 = nn.Conv2d(out_channel, out_channel,1, stride=1, padding=0, bias=False)
		self.bn = nn.BatchNorm2d(out_channel//2, eps=1e-03, momentum=0.01)
		self.relu = nn.ReLU(inplace=True)

		self.dilation1 = nn.Sequential(SeparableConv2d(out_channel,out_channel, kernel_size=3, padding=dilate_rate, dilation=dilate_rate, bias=False),
										nn.BatchNorm2d(out_channel, eps=1e-03, momentum=0.01, affine=True,
							                                 track_running_stats=True),
										nn.ReLU(inplace=True))
		self.dropout = nn.Dropout2d(dropprob)       

	def split(self,x):
		c = int(x.size()[1])
		c1 = round(c * 0.5)
		x1 = x[:, :c1, :, :].contiguous()
		x2 = x[:, c1:, :, :].contiguous()
		return x1, x2
		
	def channel_shuffle(self,x,groups=2):
		batchsize, num_channels, height, width = x.data.size()
		channels_per_group = num_channels // groups
		# reshape
		x = x.view(batchsize,groups,channels_per_group,height,width)
		x = torch.transpose(x,1,2).contiguous()
		# flatten
		x = x.view(batchsize,-1,height,width)
		return x

	def forward(self,inputs):

		residual = inputs
		x1,x2 = self.split(inputs)
		x1 = self.conv3x1(x1)
		x1 = self.bn(x1)
		x1 = self.conv1x3(x1)
		x1 = self.bn(x1)
		#x1 = self.relu(x1)

		x2 =self.conv1x3(x2)
		x2 = self.bn(x2)
		x2 =self.conv3x1(x2)
		x2 = self.bn(x2)
		#x2 = self.relu(x2)

		if (self.dropout.p != 0):
			x1 = self.dropout(x1)
			x2 = self.dropout(x2)

		x = torch.cat([x1,x2],dim=1)
		x = self.dilation1(x)
		x = self.conv1x1(x)
		
		x = self.relu(x+residual)
		
		return self.channel_shuffle(x)

class DownsamplerBlock (nn.Module):
	def __init__(self, in_channel, out_channel):
		super(DownsamplerBlock,self).__init__()

		self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=True)
		self.pool = nn.MaxPool2d(2, stride=2)
		self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, input):
		x1 = self.pool(input)
		x2 = self.conv(input)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		output = torch.cat([x2, x1], 1)
		output = self.bn(output)
		output = self.relu(output)
		return output

class Encoder(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.initial_block = DownsamplerBlock(3,32)
		self.layers = nn.ModuleList()
		for x in range(0, 3):
			self.layers.append(FCB(32,32, 0.03, 1))
		self.layers.append(DownsamplerBlock(32,64))
		for x in range(0, 2):
			self.layers.append(FCB(64,64, 0.03, 1))
		self.layers.append(DownsamplerBlock(64,128))
		for x in range(0, 1):    
			self.layers.append(FCB(128,128, 0.3, 1))
			self.layers.append(FCB(128,128, 0.3, 2))
			self.layers.append(FCB(128,128, 0.3, 5))
			self.layers.append(FCB(128,128, 0.3, 9))

		for x in range(0, 1):    
			self.layers.append(FCB(128,128, 0.3, 2))
			self.layers.append(FCB(128,128, 0.3, 5))
			self.layers.append(FCB(128,128, 0.3, 9))
			self.layers.append(FCB(128,128, 0.3, 17))

		#decoder
		self.p5 = nn.Conv2d(128,128, kernel_size=1)
		self.p4 = FPNBlock(128,128)
		self.p3 = FPNBlock(128,64)

		self.cls1 = nn.Conv2d(128,num_classes, kernel_size=1, stride=1, padding=0, bias=True)	

	def forward(self, input):
		feature=[]
		output = self.initial_block(input)
		for idx,layer in enumerate(self.layers):
			output = layer(output)
			if idx == 3 or idx == 6 :
				feature.append(output)

		p5 = self.p5(output) 
		p4 = self.p4(p5,feature[1]) #->1/4
		p3 = self.p3(p4,feature[0])  #->1/2

		x = F.interpolate(p3,scale_factor=4, mode="bilinear", align_corners=True)
		x = self.cls1(x)

		return x
