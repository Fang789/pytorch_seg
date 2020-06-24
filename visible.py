from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm 
from data.data_loader import Dataset
from model.retina import RetinaSeg
from data.augment import get_validation_augmentation,get_training_augmentation
from utils.label_color import read_labelcolors
from utils.helper import get_confusion_matrix,get_data_class
from loss.losses import CategoricalFocalLoss
from torch.nn import functional as F
#from PIL import Image
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from apex.parallel import convert_syncbn_model

# helper function for data visualization
def visualize(img_name,**images):
	"""PLot images in one row."""
	n = len(images)
	plt.figure(figsize=(16, 5))
	for i, (name, image) in enumerate(images.items()):
		plt.subplot(1, n, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.title(' '.join(name.split('_')).title())
		if name!='image':
			image = read_labelcolors(image,args.data_name)
		plt.imshow(image)
	img_save_path=os.path.join('./pics',img_name)
	plt.savefig(img_save_path) 

def visible(args):

	classes = get_data_class(args.data_name)
	n_classes = 1 if classes ==1 else classes+1
	checkpoint = torch.load(args.model_path)
	# load best saved checkpoint
	if args.train_style =="distribute":
		model = convert_syncbn_model(RetinaSeg(args.backbone,classes=n_classes))
	else:
		model = RetinaSeg(args.backbone,classes=n_classes)

	model.load_state_dict(checkpoint['model'])

	model = model.cuda()
	model.eval()

	#Visualize predictions
	test_dataset = Dataset(
			args.test_txt, 
			n_classes = classes, 
			height = args.height,
			width = args.width,
			resize = True,
	)

	test_dataset_vis = Dataset(
			args.test_txt, 
			n_classes = classes, 
			height = args.height,
			width = args.width,
			preprocessing=False,
			resize = True,
	)

	for i in range(3):
		n = np.random.choice(len(test_dataset))
		img_name = str(i)+'.png'
		
		image_vis = np.array(test_dataset_vis[n][0]).astype('uint8')	
		image, gt_mask = test_dataset[n] #(3,360,480) (12,360,480)
		
		x_tensor = image.cuda().unsqueeze(0)
		with torch.no_grad():	
			pr_mask = model(x_tensor)
			pr_mask = pr_mask.squeeze().cpu().numpy().round()

		visualize(
			img_name=img_name,
			image = image_vis, 
			ground_truth_mask=np.array(gt_mask), 
			predicted_mask=pr_mask.argmax(axis=0)
		)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--backbone', type=str,default='efficient')
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--width', type=int, default=1024)
	parser.add_argument('--test_txt', type=str, default='./txt/city_val.txt')
	parser.add_argument('--model_path', type=str, default='./weights/city_best_model_dist.pth')
	parser.add_argument('--train_style', type=str, default='distribute')
	parser.add_argument('--data_name', type=str, default='city',
						help='Dataset to use',
						choices=['ade20k','city','voc','camvid'])
	parser.add_argument('--gpu_id', type=str, default='1')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

	visible(args)
