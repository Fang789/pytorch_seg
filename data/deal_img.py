import os
import shutil
import cv2
import numpy as np
from PIL import Image
import numpy as np

def convert_camvid_to11(anno_convered_path):
	anno_folder = "/mnt/Camvid/labels/"
	annos_list = os.listdir(anno_folder)
	label_mapping={21:0,4:1,8:2,17:3,19:4,26:5,20:6,9:7,5:8,16:9,2:10,31:1,1:1,3:1,12:1,28:3,18:3,15:4,10:4,
			11:4,23:6,24:6,29:7,22:8,6:8,25:8,27:8,7:9,0:9,14:10,13:10,30:11}
	for i,anno_name in enumerate(annos_list):
		mask_path = os.path.join(anno_folder,anno_name)
		mask = Image.open(mask_path)
		mask = np.array(mask)
		mask_copy = mask.copy()
		for k, v in label_mapping.items():
			mask_copy[mask == k] = v
		mask = Image.fromarray(mask_copy.astype(np.uint8))
		save_path = os.path.join(anno_convered_path,anno_name)
		mask.save(save_path)
		print("have save {}/{}".format(i,len(annos_list)),flush=True,end="\r")

def split_camvid_data():

	for filename in [("camvid_train.txt","train","trainannot"),("camvid_val.txt","val","valannot"),("camvid_test.txt","test","testannot")]:
		with open("../txt/"+filename[0],"r") as fr:
			for line in fr:
				img_path=line.split(" ")[0]
				img_name=img_path.split("/")[-1]
				anno_name = img_name.split(".")[0]+"_P.png"

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

def get_ade20k_txt():

	img_folder="/mnt/ADEChallengeData2016/images/"
	anno_folder="/mnt/ADEChallengeData2016/annotations/"
	save_folder="/home/fangqin/progect/pytorch_seg/txt"
	for data in ("training","validation"):
		img_path=os.path.join(img_folder,data)
		anno_path=os.path.join(anno_folder,data)
		img_list=sorted(os.listdir(img_path))
		anno_list=sorted(os.listdir(anno_path))
		if data=="training":
			save_path=os.path.join(save_folder,"ade20k_train.txt")
		elif data=="validation":
			save_path =os.path.join(save_folder,"ade20k_val.txt")
		with open(save_path,"w") as fw:
			for img,anno in zip(img_list,anno_list):
				img_full_path=os.path.join(img_path,img)
				anno_full_path=os.path.join(anno_path,anno)
				line = img_full_path+" "+anno_full_path+"\n"
				fw.write(line)

if __name__=="__main__":
	
	anno_convered_path = "/mnt/Camvid/annos"
	if not os.path.exists(anno_convered_path):
		os.mkdir(anno_convered_path)
	convert_camvid_to11(anno_convered_path)
	src_img_path='/mnt/Camvid/images'
	src_anno_path="/mnt/Camvid/annos"
	dest_path='/mnt/Camvid/'

	split_camvid_data()
	get_camvid_txt()
	#get_ade20k_txt()

