import torch 
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

#model = EfficientNet.from_pretrained('efficientnet-b4')

class EfficientNet2(nn.Module):
	def __init__(self):
		super(EfficientNet2,self).__init__()
		model = EfficientNet.from_pretrained('efficientnet-b4')
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
			if idx==9 or idx==21:
				feature_maps.append(x)
		x=self.model._swish(self.model._bn1(self.model._conv_head(x)))
		feature_maps.append(x)
		return feature_maps[:]


if __name__=="__main__":
    model=EfficientNet22()
    x = Variable(torch.rand(1,3,224,224)).cuda()
    model.cuda()
    print (x.shape)     # [8,3,224,224]
    print(model)
    import pdb;pdb.set_trace()
    output = model(x)
    print (output.shape)    # [8,num_classes]
