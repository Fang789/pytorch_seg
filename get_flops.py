from model.retina import RetinaSeg
import torch
from ptflops import get_model_complexity_info
#from torchstat import stat

if __name__=="__main__":
	n_classes=12
	backbone="efficient"
	model_path="./weights/camvid_best_model.pth"

	with torch.cuda.device(0):
		net = RetinaSeg(backbone,classes=n_classes)
		net.load_state_dict(torch.load(model_path) )
		macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
												print_per_layer_stat=True, verbose=True)
		print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
		print('{:<30}  {:<8}'.format('Number of parameters: ', params))
		#flops=2*macs
