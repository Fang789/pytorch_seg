import os
import shutil
import cv2
import numpy as np
from skimage.io import imread, imsave

def split_camvid_data():

	for filename in [("camvid_train.txt","train","trainannot"),("camvid_val.txt","val","valannot"),("camvid_test.txt","test","testannot")]:
		with open("../txt/"+filename[0],"r") as fr:
			for line in fr:
				img_path=line.split(" ")[0]
				img_name=img_path.split("/")[-1]
				anno_name = img_name.split(".")[0]+"_L.png"

				img_folder=os.path.join(dest_path,filename[1])
				anno_folder=os.path.join(dest_path,filename[2])
				if not os.path.exists(img_folder):
					os.makedirs(img_folder)
				if not os.path.exists(anno_folder):
					os.makedirs(anno_folder)

				new_img_path=os.path.join(img_folder,img_name)
				new_anno_path=os.path.join(anno_folder,anno_name)

				old_img_path=os.path.join(src_img_path,img_name)
				old_anno_path=os.path.join(src_anno_path,anno_name)

				shutil.copy(old_img_path,new_img_path)
				shutil.copy(old_anno_path,new_anno_path)

def get_camvid_txt():

	folders=[("train","trainannot"),("val","valannot"),("test","testannot")]
	for folder in folders:
		img_path =os.path.join(dest_path,folder[0])	
		anno_path =os.path.join(dest_path,folder[1])	

		img_list=sorted(os.listdir(img_path))
		anno_list=sorted(os.listdir(anno_path))
		txt_path="/home/fangqin/progect/pytorch_seg/txt/"+"camvid_ac_"+folder[0]+".txt"
		with open(txt_path,"w") as fw:
			for img_name,anno_name in zip(img_list,anno_list):
				img=os.path.join(img_path,img_name)
				anno=os.path.join(anno_path,anno_name)
				line=img+" "+anno+"\n"
				fw.write(line)

def convert_rgb2mask(arr_3d):
	"""
		for camvid
	"""

	Sky = (128, 128, 128)
	Building = (128, 0, 0)
	Pole = (192, 192, 128)
	Road_marking = (255, 69, 0)
	Road = (128, 64, 128)
	Pavement = (60, 40, 222)
	Tree = (128, 128, 0)
	SignSymbol = (192, 128, 128)
	Fence = (64, 64, 128)
	Car = (64, 0, 128)
	Pedestrian = (64, 64, 0)
	Bicyclist = (0, 128, 192)

	label_codes = [Sky,Building,Pole,Road_marking,Road,Pavement,Tree,SignSymbol,Fence,Car,Pedestrian,Bicyclist]
	palette = {k:v for k,v in enumerate(label_codes)}
	arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

	for i,c in palette.items():
		m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
		arr_2d[m] = i

	return arr_2d

def predeal_label():
	anno_list = os.listdir(src_anno_path)
	for anno_name in anno_list:
		anno_path = os.path.join(src_anno_path,anno_name)
		anno = imread(anno_path)
		if (len(anno.shape) > 2):
			anno = convert_rgb2mask(anno)
			save_path = os.path.join("/mnt/Camvid/mask_all",anno_name)
			imsave(save_path,anno)

if __name__=="__main__":

	src_img_path='/mnt/Camvid/camvid_all_img'
	src_anno_path='/mnt/Camvid/camvid_all_anno'
	dest_path='/mnt/Camvid/'

	#predeal_label()
	split_camvid_data()
	get_camvid_txt()

