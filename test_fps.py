import time
import torch
import numpy as np
from model.retina import RetinaSeg

if __name__ == "__main__":

	n_classes=12
	backbone="efficient"
	model_path="./weights/camvid_best_model.pth"

	model = RetinaSeg(backbone,classes=n_classes)
	model.load_state_dict(torch.load(model_path) )
	model.cuda()
	model.eval()
	x = torch.Tensor(1, 3, 360,480).cuda()
	N = 101 #test pics numbers
	result = []
	with torch.no_grad():
		for _ in range(N):
			st = time.time()
			out = model(x)
			result.append(time.time()-st)

		print(N/sum(result))
