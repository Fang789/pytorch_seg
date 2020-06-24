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

def inference(model, image,flip=False):
	size = image.size()
	pred = model(image)
	pred = F.interpolate(input=pred, 
						size=(size[-2], size[-1]), 
						mode='bilinear',align_corners=True)        
	if flip:
		flip_img = image.cpu().numpy()
		flip_output = model(torch.from_numpy(flip_img.copy()).cuda())
		flip_output = F.interpolate(input=flip_output, 
						size=(size[-2], size[-1]), 
						mode='bilinear',align_corners=True)
		flip_pred = flip_output.cpu().numpy().copy()
		flip_pred = torch.from_numpy(flip_pred.copy()).cuda()
		pred += flip_pred
		pred = pred * 0.5
	return pred.exp()

def multi_scale_aug(image,base_size=None,label=None,rand_scale=1):

	long_size = np.int(base_size * rand_scale + 0.5)
	h, w = image.shape[:2]
	if h > w:
		new_h = long_size
		new_w = np.int(w * long_size / h + 0.5)
	else:
		new_w = long_size
		new_h = np.int(h * long_size / w + 0.5)
	
	image = cv2.resize(image, (new_w, new_h), 
					   interpolation = cv2.INTER_LINEAR)
	if label is not None:
		label = cv2.resize(label, (new_w, new_h), 
					   interpolation = cv2.INTER_NEAREST)
	else:
		return image
	
	return image, label

def multi_scale_test(model,image,base_size,num_classes,scales=[1],flip=False):

	crop_size=(args.height,args.width)
	batch, _, ori_height, ori_width = image.size()
	assert batch == 1, "only supporting batchsize 1."
	image = image.numpy()[0].transpose((1,2,0)).copy()
	stride_h = np.int(crop_size[0] * 1.0)
	stride_w = np.int(crop_size[1] * 1.0)
	final_pred = torch.zeros([1, num_classes,ori_height,ori_width]).cuda()
	for scale in scales:
		new_img = multi_scale_aug(image=image,base_size=base_size,rand_scale=scale)
		height, width = new_img.shape[:-1]
			
		if scale <= 1.0:
			new_img = new_img.transpose((2, 0, 1))
			new_img = np.expand_dims(new_img, axis=0)
			new_img = torch.from_numpy(new_img)
			new_img = new_img.cuda()
			preds = inference(model,new_img, flip)
			preds = preds[:, :, 0:height, 0:width]
		else:
			new_h, new_w = new_img.shape[:-1]
			rows = np.int(np.ceil(1.0 * (new_h - 
							crop_size[0]) / stride_h)) + 1
			cols = np.int(np.ceil(1.0 * (new_w - 
							crop_size[1]) / stride_w)) + 1
			preds = torch.zeros([1, num_classes,
									   new_h,new_w]).cuda()
			count = torch.zeros([1,1, new_h, new_w]).cuda()

			for r in range(rows):
				for c in range(cols):
					h0 = r * stride_h
					w0 = c * stride_w
					h1 = min(h0 +crop_size[0], new_h)
					w1 = min(w0 +crop_size[1], new_w)
					h0 = max(int(h1 -crop_size[0]), 0)
					w0 = max(int(w1 -crop_size[1]), 0)
					crop_img = new_img[h0:h1, w0:w1, :]
					crop_img = crop_img.transpose((2, 0, 1))
					crop_img = np.expand_dims(crop_img, axis=0)
					crop_img = torch.from_numpy(crop_img)
					crop_img = crop_img.cuda()
					pred = inference(model, crop_img, flip)
					preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
					count[:,:,h0:h1,w0:w1] += 1
			preds = preds / count
			preds = preds[:,:,:height,:width]
		preds = F.interpolate(preds, (ori_height, ori_width), 
							   mode='bilinear',align_corners=True)
		final_pred += preds
	return final_pred	

def test(args):

	classes = get_data_class(args.data_name)
	n_classes = 1 if classes == 1 else (classes + 1)  
	ms = False

	checkpoint = torch.load(args.model_path)
	# load best saved checkpoint
	if args.train_style =="distribute":
		model = convert_syncbn_model(RetinaSeg(args.backbone,classes=n_classes))
	else:
		model = RetinaSeg(args.backbone,classes=n_classes)
	model.load_state_dict(checkpoint['model'])
	model = model.cuda()
	model.eval()

	# create test dataset
	test_dataset = Dataset(
			args.test_txt, 
			n_classes = classes, 
			height = args.height,
			width = args.width,
			resize = True,
			augmentation = False,
	)
	test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)

	confusion_matrix = np.zeros((n_classes,n_classes))
	if ms:
		scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.data_name == 'city' else \
			        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
	else:
		scales = [1]
	with torch.no_grad():
		for batch in tqdm(test_dataloader):
			image, mask= batch
			mask = mask.cuda()
			pred = multi_scale_test(model,image,args.base_size,n_classes,scales=scales,flip=False) 
			confusion_matrix += get_confusion_matrix(mask,pred,n_classes)	
	
	pos = confusion_matrix.sum(1)
	res = confusion_matrix.sum(0)
	tp = np.diag(confusion_matrix)
	IoU_array = (tp / np.maximum(1.0, pos + res - tp))
	mean_IoU = IoU_array.mean()
	
	print("mIoU:{},IoU_array:{}".format(mean_IoU,IoU_array))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--backbone', type=str,default='efficient')
	parser.add_argument('--height', type=int, default=512)
	parser.add_argument('--width', type=int, default=1024)
	parser.add_argument('--base_size', type=int, default=1024,help="img max edge")
	parser.add_argument('--test_txt', type=str, default='./txt/city_val.txt')
	parser.add_argument('--model_path', type=str, default='./weights/city_best_model_dist.pth')
	parser.add_argument('--train_style', type=str, default='distribute')
	parser.add_argument('--data_name', type=str, default='city',
						help='Dataset to use',
						choices=['ade20k','city','voc','camvid'])
	parser.add_argument('--gpu_id', type=str, default='1')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

	test(args)
