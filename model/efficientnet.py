import torch 
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class EfficientNet2(nn.Module):
	def __init__(self):
		super(EfficientNet2,self).__init__()
		model = EfficientNet.from_pretrained('efficientnet-b0')
		del model._avg_pooling
		del model._dropout
		del model._fc
		self.model=model

	def forward(self,x):
		x=self.model._swish(self.model._bn0(self.model._conv_stem(x)))
		feature_maps=[]
		for idx,block in enumerate(self.model._blocks):
			drop_connect_rate=self.model._global_params.drop_connect_rate
			if drop_connect_rate:
				drop_connect_rate*=float(idx)/len(self.model._blocks)
			x=block(x,drop_connect_rate=drop_connect_rate)
			#print(idx,x.shape,drop_connect_rate)
			if idx==2 or idx==4 or idx==10:  #b0
				feature_maps.append(x)
		x=self.model._swish(self.model._bn1(self.model._conv_head(x)))
		feature_maps.append(x)
		return feature_maps[:]


if __name__=="__main__":
	model=EfficientNet2()
	#print(model)
	x = Variable(torch.rand(1,3,224,224)).cuda()
	model.cuda()
	output = model(x)

	#from torchsummary import summary
	#summary(model,(3,224,224))
