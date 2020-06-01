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
import cv2
import argparse

def train(args):

	if torch.cuda.is_available():
		num_gpus = torch.cuda.device_count()
		torch.cuda.manual_seed(123)
	else:
		torch.manual_seed(123)

	train_txt=os.path.join(args.datadir,args.data_name+'_train.txt') 
	if args.data_name!="city_split":
		val_txt=os.path.join(args.datadir,args.data_name+'_val.txt') 
	else:
		val_txt=os.path.join(args.datadir,'city_val.txt') 

	CLASSES	= get_data_class(args.data_name)
	n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  
	activation = 'sigmoid' if n_classes == 1 else 'softmax'

	# create segmentation model with pretrained encoder
	model = RetinaSeg(args.backbone,classes=n_classes)
	model.train()

	if num_gpus>1:
		parallel_model = nn.DataParallel(model).cuda()
	else:
		parallel_model = model.cuda()


	height,width=args.input_height,args.input_width

	# Dataset for train images
	train_dataset = Dataset(
		train_txt, 
		classes=CLASSES, 
		height=height,
		width=width,
		augmentation=get_training_augmentation(height,width),
	)

	# Dataset for validation images
	valid_dataset = Dataset(
		val_txt, 
		classes=CLASSES, 
		height=height,
		width=width,
		augmentation=get_validation_augmentation(height,width),
	)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size*num_gpus, shuffle=True,num_workers=8,drop_last=True)
	valid_loader = DataLoader(valid_dataset, batch_size=1,shuffle=False,num_workers=4)

	criterion = CategoricalFocalLoss(activation=activation)
	optimizer = torch.optim.Adam([dict(params=parallel_model.parameters(), lr=args.lr)])

	# train model
	best_mIoU = 0
	weight_save_path=os.path.join(args.weights,args.data_name+"_best_model.pth")

	for epoch in range(1, args.epochs+1):
		model.train()
		print('\nEpoch: {}'.format(epoch))	
		pbar = tqdm(train_loader,ncols=60)
		for batch in pbar:
			image,mask = batch
			image = image.cuda()
			mask = mask.cuda()

			optimizer.zero_grad()
			prediction = parallel_model.forward(image)
			loss = criterion(prediction, mask)
			loss.backward()
			optimizer.step()
			
			pbar.set_postfix(loss=loss.item())
			#print("loss:{:.5f}".format(loss.item()),flush=True,end='\r')#loss.cpu().detach().numpy()

		valid_loss, mean_IoU, IoU_array =valid(valid_loader,parallel_model,n_classes,criterion)
		if mean_IoU > best_mIoU:
			best_mIoU = mean_IoU
			if isinstance(parallel_model,nn.DataParallel):
				torch.save(parallel_model.module.state_dict(),weight_save_path)
			else:
				torch.save(parallel_model.state_dict(), weight_save_path)
		msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
					valid_loss, mean_IoU, best_mIoU)
		print(msg)
		print(IoU_array)	
		#if epoch in lr_scheduer:
		#    for param_group in optimizer.param_groups:
		#        param_group["lr"] = lr_scheduer[epoch]
		#    #optimizer.param_groups[0]['lr'] = 1e-5
		#    print('Decrease decoder learning rate to {}!'.format(lr_scheduer[epoch]))

def valid(valid_loader,model,n_classes,criterion):
	model.eval()
	ave_loss=[]
	confusion_matrix = np.zeros((n_classes,n_classes))
	with torch.no_grad():
		for batch in valid_loader:
			image,mask=batch
			image = image.cuda()
			mask = mask.cuda()

			pred = model.forward(image)
			loss = criterion(pred,mask)
			ave_loss.append(loss.item())
			confusion_matrix += get_confusion_matrix(mask,pred,n_classes)	

	pos = confusion_matrix.sum(1)
	res = confusion_matrix.sum(0)
	tp = np.diag(confusion_matrix)
	IoU_array = (tp / np.maximum(1.0, pos + res - tp))
	mean_IoU = IoU_array.mean()
	
	return np.mean(ave_loss),mean_IoU,IoU_array


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--backbone', type=str,default='efficient')
	parser.add_argument('--input_height', type=int, default=360)
	parser.add_argument('--input_width', type=int, default=480)
	parser.add_argument('--classes', type=int, default=2)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=4, help='single gpu batch_size')
	parser.add_argument('--datadir', type=str, default='./txt/')
	parser.add_argument('--weights', type=str, default='./weights/')
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--data_name', type=str, default='camvid',
						help='Dataset to use',
						choices=['ade20k','city','voc','camvid','city_split','camvid_ac'])
	parser.add_argument('--gpu_id', type=str, default='0,1')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

	train(args)
