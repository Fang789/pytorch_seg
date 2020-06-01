import os
import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset

# classes for data loading and preprocessing
class Dataset(BaseDataset):
	"""
	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline 
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing 
			(e.g. noralization, shape manipulation, etc.)

	"""
    
	def __init__(
			self,
			txt_dir,
			classes=None,
			height=384,
			width=480,
			resize=True,
			augmentation=None,
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]):

		self.width=width
		self.height=height
		self.img_filename_list,self.label_filename_list=self.get_filename_list(txt_dir) 
		# convert str names to class values on masks
		self.class_values = [classes.index(cls.lower()) for cls in classes]
		self.mean = mean
		self.std = std	
		self.augmentation = augmentation
		self.resize = resize

	def __getitem__(self, i):
		
		# read data
		image = cv2.imread(self.img_filename_list[i])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.label_filename_list[i],0)

		if self.resize:
			image = cv2.resize(image,(self.width,self.height))        
			mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
		
		# extract certain classes from mask (e.g. cars)
		masks = [(mask == v) for v in self.class_values]
		mask = np.stack(masks, axis=-1).astype('float')
		
		# add background if mask is not binary
		if mask.shape[-1] != 1:
			background = 1 - mask.sum(axis=-1, keepdims=True)
			mask = np.concatenate((mask, background), axis=-1)
		
		# apply augmentations
		if self.augmentation:
			sample = self.augmentation(self.height,self.width)(image=image, mask=mask)
			image, mask = sample['image'], sample['mask']
		
		# apply preprocessing
		image = self.input_transform(image)
		mask= self.label_transform(mask)
			
		return image, mask
		
	def __len__(self):
		return len(self.img_filename_list)

	def input_transform(self, image):
		image = image.astype(np.float32)
		image = image / 255.0
		image -= self.mean
		image /= self.std
		image = image.transpose((2, 0, 1))
		return image	

	def label_transform(self, label):
		return label.transpose(2, 0, 1).astype('float32')

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
    
