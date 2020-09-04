import os
import sys
import tqdm
import random
import pickle
import numpy as np
from os.path import join as pjoin
from PIL import Image, ImageDraw, ImageFont
import cv2

from base_transform import Transform
from curve import Curve
from folded import Folded
from rotation import Rotation
from perspective import Perspective

class Pipeline:

	def __init__(self, width=960, height=1280, res_path="imgs"):
		self.res_path = res_path

		self.img_postfix = ".jpg"
		self.y_postfix = ".npy"
		self.bbox_postfix = ".pickle"
		self.annot_postfix = '.xml'

	def load_np(self, input_path, label_path, group_path):
		img = cv2.imread(input_path, cv2.IMREAD_COLOR)
		label_y = np.load(label_path)
		group_y = np.load(group_path)
		return img, label_y, group_y
	
	def make_img_paths(self, img_folder_path):
		img_names = os.listdir(img_folder_path)
		img_paths = [pjoin(img_folder_path, x) for x in img_names]
		img_paths.sort()
		return img_paths
	
	def effect(self, tran, img, label_y, group_y):
		assert isinstance(tran, Transform), "Inherit Transform class"
		img = tran.run(3, img, ipt_format="opencv", opt_format="opencv")
		label_y = tran.run(1, np.expand_dims(label_y, 2), ipt_format="opencv", opt_format="opencv")
		group_y = tran.run(1, np.expand_dims(group_y, 2), ipt_format="opencv", opt_format="opencv")
		return img, label_y, group_y

	def transform(self, img, label_y, group_y):
		height, width = img.shape[0], img.shape[1]
		mode = random.randint(0, 3)
		
		# img check
#		 if not os.path.exists('test_sample'):
#			 os.makedirs('test_sample')
#		 heatmap_img = np.clip(label_y * (255 /9), 0 ,255).astype(np.uint8)
#		 heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
#		 cv2.imwrite("test_sample/prev_label_pers_{}.png".format(mode), heatmap_img)
#		 heatmap_img = np.clip(group_y * (255 /9), 0 ,255).astype(np.uint8)
#		 heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
#		 cv2.imwrite("test_sample/prev_group_pers_{}.png".format(mode), heatmap_img)
		
		if mode == 4:
			tran = Perspective(width, height)
		elif mode % 2 == 0:
			curve_random = random.randint(5, 20)
			curve_mode = "down" if random.randint(0, 1) > 0 else "up"
			tran = Curve(width=width, height=height, spacing=40,
						 flexure=curve_random/100, direction=curve_mode,
						 is_horizon=(mode%4==0))
		else:
			folded_up = random.randint(5, 20)
			folded_down = random.randint(5, 20)
			tran = Folded(width=width, height=height, spacing=40,
						  up_slope=folded_up/100, down_slope=folded_down/100,
						  is_horizon=(mode%4==1))
		
		label_y = cv2.resize(label_y, (width, height))
		group_y = cv2.resize(group_y, (width, height))
		img, label_y, group_y = self.effect(tran, img, label_y, group_y)
		if np.random.randint(8) == 0:
			per = Perspective(width, height)
			img, label_y, group_y = self.effect(per, img, label_y, group_y)

		rot = Rotation(width, height)
		img, label_y, group_y = self.effect(rot, img, label_y, group_y)
		
		
		########################### setting resize ##########################
		r_width, r_height = 480, 640
		label_y = np.squeeze(label_y)
		label_y = cv2.resize(label_y, (r_width, r_height))
		group_y = np.squeeze(label_y)
		group_y = cv2.resize(label_y, (r_width, r_height))
		######################################################################
		
		# transformed img check
#		 cv2.imwrite("test_sample/group_pers_ori_{}.png".format(mode), img)
#		 heatmap_img = np.clip(group_y * (255 /9), 0 ,255).astype(np.uint8)
#		 heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
#		 cv2.imwrite("test_sample/group_pers_{}.png".format(mode), heatmap_img)
#		 heatmap_img = np.clip(label_y * (255 /9), 0 ,255).astype(np.uint8)
#		 heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
#		 cv2.imwrite("test_sample/label_pers_{}.png".format(mode), heatmap_img)
		
		return img, label_y, group_y

	def save(self, img, label_y, group_y, res_path, input_name, label_name, group_name):
		if not os.path.exists(pjoin(res_path, 'transform_input')):
			os.makedirs(pjoin(res_path, 'transform_input'))
		if not os.path.exists(pjoin(res_path, 'transform_label')):
			os.makedirs(pjoin(res_path, 'transform_label'))
		if not os.path.exists(pjoin(res_path, 'transform_group')):
			os.makedirs(pjoin(res_path, 'transform_group'))

		cv2.imwrite(pjoin(res_path, 'transform_input', input_name), img)
		np.save(pjoin(res_path, 'transform_label', label_name),  label_y)
		np.save(pjoin(res_path, 'transform_group', group_name),  group_y)

	def run(self):
		######################### select folder list #########################
		root_path = '/data/AugmentedImage'
		document_list = ['doc_0001', 'doc_0002']
		folder1_list = ['train', 'test']
		folder2_list = ['200', '300']
		folder3_list = ['BW', 'Color', 'Gray']
		######################################################################
		
		for document_name in document_list:
			for folder1 in folder1_list:
				for folder2 in folder2_list:
					for folder3 in folder3_list:
						print('folder :', pjoin(document_name, folder1, folder2, folder3))						  
						res_path = pjoin(root_path, document_name, folder1, folder2, folder3)

						# input 열기
						input_folder_path = pjoin(res_path, 'input')
						input_img_paths = self.make_img_paths(input_folder_path)
						#print(input_img_paths)						

						# label 열기
						label_folder_path = pjoin(res_path, 'label')
						label_img_paths = self.make_img_paths(label_folder_path)
						#print(label_img_paths)
						
						# group 열기
						group_folder_path = pjoin(res_path, 'group')
				   
						group_img_paths = self.make_img_paths(group_folder_path)
						#print(group_img_paths)
						

						for input_path, label_path, group_path in tqdm.tqdm(zip(input_img_paths, label_img_paths, group_img_paths)):
							#print(input_path, label_path)
							input_name = os.path.basename(input_path)
							label_name = os.path.basename(label_path)
							group_name = os.path.basename(group_path)						
							
							if label_name[-3:] != 'npy':
								print('label file name error', label_name)
								continue
							
							img, label_y, group_y = self.load_np(input_path, label_path, group_path)
							img, label_y, group_y = self.transform(img, label_y, group_y)
							self.save(img, label_y, group_y, res_path, input_name, label_name, group_name)
						


if __name__ == "__main__":
	p = Pipeline()
	p.run()


