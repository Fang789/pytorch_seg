import os
import numpy as np
from PIL import Image,ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
import random

# classes for data loading and preprocessing
class Dataset(BaseDataset):
    
	def __init__(
			self,
			txt_dir,
			n_classes=None,
			height=384,
			width=480,
			ignore_label=-1,
			augmentation=False,
			preprocessing=True,
			val_data = False,
			):

		self.width = width
		self.height = height
		self.txt_dir = txt_dir
		self.val_data = val_data
		self.img_filename_list,self.label_filename_list=self.get_filename_list(txt_dir) 
		self.normalize = transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
		self.augmentation = augmentation
		self.preprocessing = preprocessing
		self.crop_size = (height,width)
		self.base_size = 512
		if n_classes == 11:
			self.label_mapping = {k:k for k in range(n_classes)}
		elif n_classes ==19:
			self.label_mapping = {-1: ignore_label, 0: ignore_label, 
							  1: ignore_label, 2: ignore_label, 
							  3: ignore_label, 4: ignore_label, 
							  5: ignore_label, 6: ignore_label, 
							  7: 0, 8: 1, 9: ignore_label, 
							  10: ignore_label, 11: 2, 12: 3, 
							  13: 4, 14: ignore_label, 15: ignore_label, 
							  16: ignore_label, 17: 5, 18: ignore_label, 
							  19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
							  25: 12, 26: 13, 27: 14, 28: 15, 
							  29: ignore_label, 30: ignore_label, 
							  31: 16, 32: 17, 33: 18}

	def __getitem__(self, i):
		
		# read data
		image = Image.open(self.img_filename_list[i]).convert('RGB')
		if 'city_test' in self.txt_dir:
			mask = Image.open(self.label_filename_list[i]).convert('L')
		else:
			mask = Image.open(self.label_filename_list[i])
		assert(mask.mode == "L")

		if self.val_data:
			image,mask = self.val_crop(image,mask)

		# apply augmentations for train dataset
		if self.augmentation:
			image,mask = self.augement(image,mask)
		
		# apply preprocessing
		if self.preprocessing:
			image = self.input_transform(image)
			mask= self.label_transform(mask)
			
		return image, mask
		
	def __len__(self):
		return len(self.img_filename_list)

	def input_transform(self,image):
		image = np.float32(np.array(image)) / 255.
		image = image.transpose((2, 0, 1))
		image = self.normalize(torch.from_numpy(image.copy()))
		return image	

	def label_transform(self,mask):
		mask = np.array(mask)
		mask_copy = mask.copy()
		for k, v in self.label_mapping.items():
			mask_copy[mask == k] = v
		mask = Image.fromarray(mask_copy.astype(np.uint8))	
		return torch.from_numpy(np.array(mask)).long() 

	def get_filename_list(self,txt_dir):
		f=open(txt_dir,'r')
		img_filename_list = []
		label_filename_list = []
		for line in f:
			try:
				line= line[:-1].split(' ')
				img_filename_list.append(line[0])
				label_filename_list.append(line[1])
			except ValueError:
				print('Check that the path is correct.')

		return img_filename_list, label_filename_list

	def augement(self,img,mask):
		# random mirror
		crop_size = self.crop_size
		if random.random() < 0.5:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)
			mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
		# random scale (short edge from 480 to 720)
		short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
		w, h = img.size
		if h > w:
			ow = short_size
			oh = int(1.0 * h * ow / w)
		else:
			oh = short_size
			ow = int(1.0 * w * oh / h)
		img = img.resize((ow, oh), Image.BILINEAR)
		mask = mask.resize((ow, oh), Image.NEAREST)
		# random rotate -10~10, mask using NN rotate
		deg = random.uniform(-10, 10)
		img = img.rotate(deg, resample=Image.BILINEAR)
		mask = mask.rotate(deg, resample=Image.NEAREST)
		# pad crop
		if short_size < min(crop_size):
			padh = crop_size[0] - oh if oh < crop_size[0] else 0
			padw = crop_size[1] - ow if ow < crop_size[1] else 0
			img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
			mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
		# random crop crop_size
		w, h = img.size
		x1 = random.randint(0, w - crop_size[1])
		y1 = random.randint(0, h - crop_size[0])
		img = img.crop((x1, y1, x1+crop_size[1], y1+crop_size[0]))
		mask = mask.crop((x1, y1, x1+crop_size[1], y1+crop_size[0]))
		# gaussian blur as in PSP
		if random.random() < 0.5:
			img = img.filter(ImageFilter.GaussianBlur(
				radius = random.random()))

		return img,mask

	def val_crop(self,img,mask):
		outsize = self.crop_size
		short_size = outsize[0]
		w, h = img.size
		if w > h:
			oh = short_size
			ow = int(1.0 * w * oh / h)
		else:
			ow = short_size
			oh = int(1.0 * h * ow / w)
		img = img.resize((ow, oh), Image.BILINEAR)
		mask = mask.resize((ow, oh), Image.NEAREST)
		# center crop
		w, h = img.size
		x1 = int(round((w - outsize[1]) / 2.))
		y1 = int(round((h - outsize[0]) / 2.))
		img = img.crop((x1, y1, x1+outsize[1], y1+outsize[0]))
		mask = mask.crop((x1, y1, x1+outsize[1],y1+outsize[0]))

		return img,mask	
   
if __name__=="__main__":
	
	train_txt="../txt/city_split_train.txt"
	height,width = 512,1024
	train_dataset = Dataset(train_txt,height=height,width=width,n_classes=19,augmentation=True)
	print(train_dataset[0])
