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

    def load_files(self):
        origin_file_list = os.listdir(pjoin(self.res_path, "origin"))
        label_file_list = os.listdir(pjoin(self.res_path, "origin_label"))
        bbox_file_list = os.listdir(pjoin(self.res_path, "origin_bbox"))
        assert len(origin_file_list) == len(label_file_list)# == len(bbox_file_list)
        origin_file_list.sort()
        return origin_file_list

    def load_np(self, input_path, label_path):
        #idx = str(idx)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        y = np.load(label_path)
        #with open(pjoin(self.res_path, "origin_bbox", idx+self.bbox_postfix), 'rb') as fin:
        #    bbox = pickle.load(fin)
        return img, y
    
    def make_img_paths(self, img_folder_path):
        img_names = os.listdir(img_folder_path)
        img_paths = [pjoin(img_folder_path, x) for x in img_names]
        img_paths.sort()
        return img_paths
    
    def effect(self, tran, img, y):#, bbox):
        assert isinstance(tran, Transform), "Inherit Transform class"
        img = tran.run(3, img, ipt_format="opencv", opt_format="opencv")
        y = tran.run(1, np.expand_dims(y, 2), ipt_format="opencv", opt_format="opencv")
        #bbox = tran.transform_points(bbox)
        return img, y

    def transform(self, img, y):#, bbox):
        #img = self.noise_generate(img.copy())
#         points, labels = [], []
#         for idx, x1, y1, x2, y2, x3, y3, x4, y4 in bbox:
#             points.append((x1, y1))
#             points.append((x2, y2))
#             points.append((x3, y3))
#             points.append((x4, y4))
#             labels.append(idx)
#         bbox = points
        height, width = img.shape[0], img.shape[1]
        mode = random.randint(0, 3)
        
        print(mode)
        print(y.shape)
        
        heatmap_img = np.clip(y * (255 /9), 0 ,255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        cv2.imwrite("test_sample/prev_label_pers_{}.png".format(mode), heatmap_img)
        
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
        
        y = cv2.resize(y, (width, height))
        img, y = self.effect(tran, img, y)
        if np.random.randint(8) == 0:
            per = Perspective(width, height)
            img, y = self.effect(per, img, y)

        rot = Rotation(width, height)
        img, y = self.effect(rot, img, y)
        
        y = np.squeeze(y)
        y = cv2.resize(y, (width // 2, height // 2))
        #y = y.transpose(1, 0).astype(np.uint8)
        
        print(y.shape)
        
        heatmap_img = np.clip(y * (255 /9), 0 ,255).astype(np.uint8)
        #heatmap_img = np.clip(y.transpose(1, 0) * (255 /9), 0 ,255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        cv2.imwrite("test_sample/label_pers_ori_{}.png".format(mode), img)
        cv2.imwrite("test_sample/label_pers_{}.png".format(mode), heatmap_img)
        return img, y

    def save(self, img, y, res_path, input_name, label_name):
        if not os.path.exists(pjoin(res_path, 'transform_input')):
            os.makedirs(pjoin(res_path, 'transform_input'))
        if not os.path.exists(pjoin(res_path, 'transform_label')):
            os.makedirs(pjoin(res_path, 'transform_label'))

        cv2.imwrite(pjoin(res_path, 'transform_input', input_name), img)
        np.save(pjoin(res_path, 'transform_label', label_name),  y)

    def run(self):
        # load
        root_path = 'DenoiseImageSet_20200806'
        document_list = ['doc_0001']
        folder1_list = ['train']#, 'test']
        folder2_list = ['200']#, '300']
        folder3_list = ['BW']#, 'Color', 'Gray']
        
        for document_name in document_list:
            for folder1 in folder1_list:
                for folder2 in folder2_list:
                    for folder3 in folder3_list:
                        print('folder :', pjoin(folder1, folder2, folder3))
                        # 해당 위치에 transform_input, transform_label folder 제작
                        res_path = pjoin(root_path, document_name, folder1, folder2, folder3) 
                        
                        # 읽어야할 folder name

                        # input 열기
                        input_folder_path = pjoin(res_path, 'input')
                        input_img_paths = self.make_img_paths(input_folder_path)
                        print(input_img_paths)
                    
                        # label 열기
                        label_folder_path = pjoin(res_path, 'label')
                        label_img_paths = self.make_img_paths(label_folder_path)
                        print(label_img_paths)
                        
                        for input_path, label_path in zip(input_img_paths, label_img_paths):
                            #print(input_path, label_path)
                            input_name = os.path.basename(input_path)
                            label_name = os.path.basename(label_path)
                            print(input_name)
                            print(label_name)
                            
                            if label_name[-3:] != 'npy':
                                print(label_name)
                                continue
                            
                            img, y = self.load_np(input_path, label_path)
                            img, y = self.transform(img, y)
                            self.save(img, y, res_path, input_name, label_name)
                        


if __name__ == "__main__":
    p = Pipeline()
    p.run()


