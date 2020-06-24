import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
import numpy as np
from model.efficientnet import EfficientNet2 
from model.net import FPNBlock,PSPHead,Conv3x3GNReLU,SSH,SpatialAttention
from model.jpu import JPU, SeparableConv2d
from model.aspp import ASPP

class RetinaSeg(nn.Module):
	def __init__(self,encoder_name,classes):
		"""
		encoder_name: backbone name
		classes: network classes
		"""
		super(RetinaSeg,self).__init__()
		
		backbone=None
		if encoder_name == 'Resnet50':
			import torchvision.models as models
			backbone = models.resnet50(pretrained=True)
		elif  encoder_name == 'Resnest50':
			from resnest.torch import resnest50
			backbone= resnest50(pretrained=True)
		elif encoder_name =="efficient":
			backbone = EfficientNet2()

		#print(backbone)
		return_layers= {'layer2': 0, 'layer3': 1, 'layer4': 2}
		if encoder_name!="efficient":
			self.body = _utils.IntermediateLayerGetter(backbone, return_layers)
		else:
			self.body=backbone

		if encoder_name=='Resnet50' or encoder_name=='Resnest50':
			in_channels_list = [512,1024,2048]
		elif encoder_name=="efficient":
			#in_channels_list =[56,160,1792] #b4
			in_channels_list =[24,40,112,1280] #b0

		out_channels = 256

		self.p5 = nn.Conv2d(in_channels_list[3],out_channels, kernel_size=1)
		self.p4 = FPNBlock(out_channels,in_channels_list[2])
		self.p3 = FPNBlock(out_channels,in_channels_list[1])

		self.p5_up = FPNBlock(out_channels,in_channels_list[2])
		self.p4_up = FPNBlock(out_channels,in_channels_list[1])
		self.p3_up = FPNBlock(out_channels,in_channels_list[0])

		self.cbr = Conv3x3GNReLU(out_channels,out_channels,upsample=True)
		#self.split_dilated = Split_DilatedConv(out_channels*2,out_channels)
		norm_layer=nn.BatchNorm2d

		self.dilation1 = nn.Sequential(SeparableConv2d(out_channels*2,out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
										norm_layer(out_channels),
										nn.ReLU(inplace=True))

		self.dilation2 = nn.Sequential(SeparableConv2d(out_channels*2,out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
										norm_layer(out_channels),
										nn.ReLU(inplace=True))

		self.dilation3 = nn.Sequential(SeparableConv2d(out_channels*2,out_channels, kernel_size=3,padding=4, dilation=4, bias=False),
										norm_layer(out_channels),
										nn.ReLU(inplace=True))

		#self.dilation4 = nn.Sequential(SeparableConv2d(2*out_channels,out_channels, kernel_size=3, padding=8, dilation=8, bias=False),
		#									norm_layer(out_channels),
		#									nn.ReLU(inplace=True))

		#self.aspp = nn.Sequential(
		#	ASPP(out_channels,out_channels*2, atrous_rates=(6,12,18), separable=True),
		#	#SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
		#	nn.Conv2d(out_channels*2, out_channels*2, 3, padding=1, bias=False),
		#	nn.BatchNorm2d(out_channels*2),
		#	nn.ReLU(),
		#)
		#self.ssh = SSH(out_channels, out_channels)
		#self.ssh = SSH(out_channels*2,out_channels*2)
		#self.se1=SEModule(out_channels)
		#self.se2=SEModule(out_channels)
		#self.se3=SEModule(out_channels)
		#self.psphaed = PSPHead(
		#	in_channels=out_channels*2,
		#	use_batchnorm=True,
		#	out_channels=out_channels*2,
		#	dropout=None,
		#)
		#self.jpu = JPU([256,256,256],out_channels)
			
		#self.dropout = nn.Dropout2d(p=0.2)
		self.cls1 = nn.Conv2d(768,classes, kernel_size=1, stride=1, padding=0, bias=True)	

	def _make_seg_head(self,encoder,size=None,scale_factor=4,muti_head_up=False):
		
		if muti_head_up:
			# Upsampling
			x0_h, x0_w = size[0],size[1]
			x1 = F.interpolate(encoder[0], size=(x0_h, x0_w), mode='bilinear',align_corners=True)
			x2 = F.interpolate(encoder[1], size=(x0_h, x0_w), mode='bilinear',align_corners=True)
			x3 = F.interpolate(encoder[2], size=(x0_h, x0_w), mode='bilinear',align_corners=True)
			x = torch.cat([x1, x2, x3], 1)
			x = self.cls2(x)
		else:
			x = F.interpolate(encoder,scale_factor=scale_factor, mode="bilinear", align_corners=True)
			x = self.cls1(x)

		return x

	def forward(self,inputs):

		out= self.body(inputs)
		
		# FPN
		p5 = self.p5(out[3]) 
		p4 = self.p4(p5, out[2])
		p3 = self.p3(p4, out[1])

		p5_cat = self.p5_up(p5,out[2])
		p4_cat = self.p4_up(p4,out[1])
		p3_cat = self.p3_up(p3,out[0])

		#p5_2 = self.p5_2(p5_cat) 
		#p4_2 = self.p4_2(p5_2, p4_cat)
		#p3_2 = self.p3_2(p4_2, p3_cat)


		#p3_cat = self.cbr(p3_cat)
		p3_2 = self.cbr(p3,p3_cat)
		#p3 = self.cbr(p3_cat)

		#test
		feat = torch.cat((p3_cat,p3_2),dim=1)
		#out = self.ssh(feat)
		#out = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
		out = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat)], dim=1)
		#out = self.dropout(out)
		#out = self.sa(out)*out

		#JPU
		#out = self.jpu([p3,p4,p5])

		# SE
		#feature1 = self.se1(fpn[0])

		# SSH
		#out = self.ssh(p3)

		# PSPHead
		#out = self.psphaed(p3)
		# ASPP
		#out=self.aspp(p3)
		output = self._make_seg_head(out,size=inputs.shape[2:],muti_head_up=False) 

		return output
