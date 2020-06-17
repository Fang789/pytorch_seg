from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm 
from data.data_loader import Dataset
from model.retina import RetinaSeg
from data.augment import get_validation_augmentation,get_training_augmentation
from utils.label_color import read_labelcolors,get_class_weight
from utils.helper import get_confusion_matrix,get_data_class
from loss.losses import CrossEntropy,Lovasz_Softmax
from torch.optim import lr_scheduler,Adam,AdamW
import cv2
import argparse
torch.backends.cudnn.deterministic = True

import torch.distributed as dist
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

#os.environ['MASTER_ADDR'] = '127.0.0.1'
#os.environ['MASTER_PORT'] = '29555'


def train(args):

	if torch.cuda.is_available():
		#num_gpus = torch.cuda.device_count()
		torch.cuda.manual_seed(123)
	else:
		torch.manual_seed(123)

	train_txt=os.path.join(args.datadir,args.data_name+'_train.txt') 
	if args.data_name!="city_split":
		val_txt=os.path.join(args.datadir,args.data_name+'_val.txt') 
	else:
		val_txt=os.path.join(args.datadir,'city_val.txt') 

	classes	= get_data_class(args.data_name)
	n_classes = 1 if classes == 1 else classes+1
	#activation = 'sigmoid' if n_classes == 1 else 'softmax'

	torch.cuda.set_device(args.local_rank)
	dist.init_process_group(backend='nccl',init_method="env://")#world_size=2,rank=args.local_rank,world_size=2,rank=0

	# create segmentation model with pretrained encoder
	model = convert_syncbn_model(RetinaSeg(args.backbone,classes=n_classes)).cuda()
	optimizer = AdamW([dict(params=model.parameters(), lr=args.lr)])
	model, optimizer = amp.initialize(model, optimizer, opt_level='O1') #O1表示混合精度
	if args.resume_path is not None:
		checkpoint = torch.load(args.resume_path)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		amp.load_state_dict(checkpoint['amp'])	
		start_epoch = checkpoint['epoch']
		print("recover model from {}".format(args.resume_path))
	else:
		start_epoch = 1
	
	parallel_model = DistributedDataParallel(model,delay_allreduce=True)

	class_weight = get_class_weight(args.data_name)
	criterion = CrossEntropy(weight=class_weight).cuda()#weight=class_weigh
	#lambda1 = lambda epoch: pow((1-((epoch-1)/args.epochs)),0.9)
	#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

	height,width=args.input_height,args.input_width

	# Dataset for train images
	train_dataset = Dataset(
		train_txt, 
		n_classes=classes, 
		height=height,
		width=width,
		augmentation=True,
	)
	train_sampler = DistributedSampler(train_dataset)

	# Dataset for validation images
	valid_dataset = Dataset(
		val_txt, 
		n_classes=classes, 
		height=height,
		width=width,
		resize =True,
		augmentation=False,
	)
	val_sampler = DistributedSampler(valid_dataset)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=8,drop_last=True,pin_memory=True,sampler=train_sampler)#
	valid_loader = DataLoader(valid_dataset, batch_size=1,shuffle=False,num_workers=4,pin_memory=True,sampler=val_sampler)#


	# train model
	best_mIoU = 0
	weight_save_path=os.path.join(args.weights,args.data_name+"_best_model_dist.pth")

	for epoch in range(start_epoch, args.epochs+1):
		model.train()
		print('\nEpoch: {}'.format(epoch))	
		pbar = tqdm(train_loader,ncols=60)
		for batch in pbar:
			image,mask = batch
			image = image.cuda(non_blocking=True)
			mask = mask.cuda(non_blocking=True)

			optimizer.zero_grad()
			prediction = parallel_model.forward(image)
			loss = criterion(prediction, mask)
			with amp.scale_loss(loss, optimizer) as scaled_loss:				
				scaled_loss.backward()
			optimizer.step()
			pbar.set_postfix(loss=loss.item())

		#scheduler.step()	
		valid_loss, mean_IoU, IoU_array =valid(valid_loader,parallel_model,n_classes,criterion)
		if mean_IoU > best_mIoU:
			best_mIoU = mean_IoU
			if args.local_rank == 0:
				checkpoint = {
					'epoch':epoch,
					'model':parallel_model.module.state_dict(),
					'optimizer': optimizer.state_dict(),
					'amp': amp.state_dict()
				}
				torch.save(checkpoint,weight_save_path)

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
			image = image.cuda(non_blocking=True)
			mask = mask.cuda(non_blocking=True)

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
	parser.add_argument('--epochs', type=int, default=500)
	parser.add_argument('--batch_size', type=int, default=4, help='single gpu batch_size')
	parser.add_argument('--datadir', type=str, default='./txt/')
	parser.add_argument('--weights', type=str, default='./weights/')
	parser.add_argument('--resume_path', type=str, default=None)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--data_name', type=str, default='city',
						help='Dataset to use',
						choices=['ade20k','city','voc','camvid','city_split','camvid_ac'])
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument('--gpu_id', type=str, default='0,1')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

	train(args)
