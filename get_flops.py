from model.retina import RetinaSeg
import torch
from ptflops import get_model_complexity_info
from apex import amp
from apex.parallel import convert_syncbn_model

if __name__=="__main__":
	n_classes=12
	backbone="efficient"
	model_path="./weights/camvid_best_model_dist_73_7.pth"
	checkpoint = torch.load(model_path)

	dist = True
	with torch.cuda.device(0):
		if dist:
			model = convert_syncbn_model(RetinaSeg(backbone,classes=n_classes,aux=False))
		else:
			model = RetinaSeg(backbone,classes=n_classes)

		model.load_state_dict(checkpoint['model'])

		macs, params = get_model_complexity_info(model, (3, 360,480), as_strings=True,
												print_per_layer_stat=False, verbose=True)
		print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
		print('{:<30}  {:<8}'.format('Number of parameters: ', params))
		#FLOPs=2*MACs #1GFLOPs=10**9FLOPs
