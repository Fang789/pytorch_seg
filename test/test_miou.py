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
import cv2
import argparse
from apex.parallel import convert_syncbn_model
from test import SegmentationMetric 

def test(args):

	classes = get_data_class(args.data_name)
	n_classes = 1 if classes == 1 else (classes + 1)  
	ms = False

	checkpoint = torch.load(args.model_path)
	# load best saved checkpoint
	if args.train_style =="distribute":
		model = convert_syncbn_model(RetinaSeg(args.backbone,classes=n_classes,aux=False))
	else:
		model = RetinaSeg(args.backbone,classes=n_classes)
	model.load_state_dict(checkpoint['model'])
	model = model.cuda()
	model.eval()

	metric = SegmentationMetric(n_classes,False)
	metric.reset()
	# create test dataset
	test_dataset = Dataset(
			args.test_txt, 
			n_classes = classes, 
			height = args.height,
			width = args.width,
			val_data = True,
			augmentation = False,
	)
	test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)

	confusion_matrix = np.zeros((n_classes,n_classes))
	if ms:
		scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.data_name == 'city' else \
			        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
	else:
		scales = [1]

	for batch in tqdm(test_dataloader):
		image, mask= batch
		mask = mask.cuda()
		image = image.cuda()
		with torch.no_grad():
			pred = model(image)
		metric.update(pred,mask)
		pixAcc, mIoU = metric.get()

	pixAcc, mIoU, category_iou = metric.get(return_category_iou=True)
	print('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(pixAcc * 100, mIoU * 100))
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--backbone', type=str,default='efficient')
	parser.add_argument('--height', type=int, default=360)
	parser.add_argument('--width', type=int, default=480)
	parser.add_argument('--base_size', type=int, default=480,help="img max edge")
	parser.add_argument('--test_txt', type=str, default='./txt/camvid_ac_val.txt')
	parser.add_argument('--model_path', type=str, default='./weights/camvid_ac_best_model_dist.pth')
	parser.add_argument('--train_style', type=str, default='distribute')
	parser.add_argument('--data_name', type=str, default='camvid_ac',
						help='Dataset to use',
						choices=['ade20k','city','voc','camvid'])
	parser.add_argument('--gpu_id', type=str, default='1')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

	test(args)
