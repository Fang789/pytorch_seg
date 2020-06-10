from model.retina import RetinaSeg
import torch
from torchstat import stat

def get_flops(net):
	stat(net,(3,360,480))

if __name__=="__main__":
	n_classes=12
	backbone="Resnet50"
	model_path="./weights/camvid_best_model.pth"

	net = RetinaSeg(backbone,classes=n_classes)
	net.load_state_dict(torch.load(model_path) )
	get_flops(net)
