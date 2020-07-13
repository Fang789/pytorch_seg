import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
from model.net import FPNBlock,PSPHead,Conv3x3GNReLU,SSH,SpatialAttention
from model.jpu import JPU, SeparableConv2d

def split(x):
	c = int(x.size()[1])
	c1 = round(c * 0.5)
	x1 = x[:, :c1, :, :].contiguous()
	x2 = x[:, c1:, :, :].contiguous()

	return x1, x2

def channel_shuffle(x,groups):
	batchsize, num_channels, height, width = x.data.size()
	channels_per_group = num_channels // groups
	# reshape
	x = x.view(batchsize,groups,channels_per_group,height,width)
	x = torch.transpose(x,1,2).contiguous()
	# flatten
	x = x.view(batchsize,-1,height,width)
	return x

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

class SS_nbt_module(nn.Module):
	def __init__(self, chann, dropprob, dilated):        
		super().__init__()
		oup_inc = chann//2
		# dw
		self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
		self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
		self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
		self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
		self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
		self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
		# dw
		self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)
		self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)
		self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)
		self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))
		self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))
		self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)       
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout2d(dropprob)       

	def forward(self, input):
		# x1 = input[:,:(input.shape[1]//2),:,:]
		# x2 = input[:,(input.shape[1]//2):,:,:]
		residual = input
		x1, x2 = split(input)
		output1 = self.conv3x1_1_l(x1)
		output1 = self.relu(output1)
		output1 = self.conv1x3_1_l(output1)
		output1 = self.bn1_l(output1)
		output1 = self.relu(output1)

		output1 = self.conv3x1_2_l(output1)
		output1 = self.relu(output1)
		output1 = self.conv1x3_2_l(output1)
		output1 = self.bn2_l(output1)

		output2 = self.conv1x3_1_r(x2)
		output2 = self.relu(output2)
		output2 = self.conv3x1_1_r(output2)
		output2 = self.bn1_r(output2)
		output2 = self.relu(output2)

		output2 = self.conv1x3_2_r(output2)
		output2 = self.relu(output2)
		output2 = self.conv3x1_2_r(output2)
		output2 = self.bn2_r(output2)

		if (self.dropout.p != 0):
			output1 = self.dropout(output1)
			output2 = self.dropout(output2)

		out = torch.cat([output1,output2],dim=1)
		out = F.relu(residual + out, inplace=True)
		return channel_shuffle(out,2)

class Encoder(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.initial_block = DownsamplerBlock(3,32)
		self.layers = nn.ModuleList()
		for x in range(0, 3):
			self.layers.append(SS_nbt_module(32, 0.03, 1))
		self.layers.append(DownsamplerBlock(32,64))
		for x in range(0, 2):
			self.layers.append(SS_nbt_module(64, 0.03, 1))
		self.layers.append(DownsamplerBlock(64,128))
		for x in range(0, 1):    
			self.layers.append(SS_nbt_module(128, 0.3, 1))
			self.layers.append(SS_nbt_module(128, 0.3, 2))
			self.layers.append(SS_nbt_module(128, 0.3, 5))
			self.layers.append(SS_nbt_module(128, 0.3, 9))
		for x in range(0, 1):    
			self.layers.append(SS_nbt_module(128, 0.3, 2))
			self.layers.append(SS_nbt_module(128, 0.3, 5))
			self.layers.append(SS_nbt_module(128, 0.3, 9))
			self.layers.append(SS_nbt_module(128, 0.3, 17))

	def forward(self, input, predict=False):
		output = self.initial_block(input)
		feature=[]
		for idx,layer in enumerate(self.layers):
			output = layer(output)
			#print(idx,output.shape)
			if idx==5 or idx ==10 :
				feature.append(output)
		feature.append(output)
		return output,feature

class TestNet(nn.Module):
	def __init__(self, classes):  
		super().__init__()

		self.encoder = Encoder(classes)
		#decoder
		out_channels=128
		in_channels_list=[32,64,128,128]
		norm_layer=nn.BatchNorm2d

		self.p5 = nn.Conv2d(out_channels,out_channels, kernel_size=1)
		self.p4 = FPNBlock(out_channels,in_channels_list[2])
		self.p3 = FPNBlock(out_channels,in_channels_list[1])

		self.p5_up = FPNBlock(out_channels,in_channels_list[3])
		self.p4_up = FPNBlock(out_channels,in_channels_list[2])
		self.p3_up = FPNBlock(out_channels,in_channels_list[1])

		self.cbr = Conv3x3GNReLU(out_channels,out_channels,upsample= False)

		self.dilation1 = nn.Sequential(SeparableConv2d(out_channels*2,out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
										norm_layer(out_channels),
										nn.ReLU(inplace=True))

		self.dilation2 = nn.Sequential(SeparableConv2d(out_channels*2,out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
										norm_layer(out_channels),
										nn.ReLU(inplace=True))

		self.dilation3 = nn.Sequential(SeparableConv2d(out_channels*2,out_channels, kernel_size=3,padding=4, dilation=4, bias=False),
										norm_layer(out_channels),
										nn.ReLU(inplace=True))
	
		self.cls1 = nn.Conv2d(384,classes, kernel_size=1, stride=1, padding=0, bias=True)	

	def forward(self,input):
		x,out = self.encoder(input)
		
		p5 = self.p5(out[2]) 
		p4 = self.p4(p5, out[1])
		p3 = self.p3(p4, out[0])

		p5_cat = self.p5_up(p5,out[2])
		p4_cat = self.p4_up(p4,out[1])
		p3_cat = self.p3_up(p3,out[0])

		p3_2 = self.cbr(p3,p3_cat)

		#test
		feat = torch.cat((p3_cat,p3_2),dim=1)
		out = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
		
		x = F.interpolate(out,scale_factor=4, mode="bilinear", align_corners=True)
		x = self.cls1(x)
		
		return x

