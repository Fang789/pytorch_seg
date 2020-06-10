import time
import torch
import numpy as np
from model.retina import RetinaSeg

def get_parameter(model):
	num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
	print("model parameters:{:.2f}M".format(num_parameters/(10**6)))

def get_fps(model,x):
	model.cuda()
	model.eval()
	N = 100 #test pics numbers
	with torch.no_grad():
		torch.cuda.synchronize()
		st = time.time()*1000
		for _ in range(N):
			pred = model(x)
			out = pred.argmax(axis=0)
			out = out.cpu().numpy()
			torch.cuda.synchronize()
		end=time.time()*1000
		print(N*1000/(end-st))

if __name__ == "__main__":

	n_classes=12
	backbone="Resnet50"
	model_path="./weights/camvid_best_model.pth"

	model = RetinaSeg(backbone,classes=n_classes)
	model.load_state_dict(torch.load(model_path) )

	x = torch.Tensor(1, 3, 360,480).cuda()
	get_parameter(model)
	get_fps(model,x)
	

