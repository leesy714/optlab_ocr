import os
import sys
import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.append("../")
from curve import Curve
from folded import Folded

class Pipeline:

    def __init__(self, width=960, height=1280):
        self.width = width
        self.height = height

    def load_files(self):
        origin_file_list = os.listdir("./data/origin")
        label_file_list = os.listdir("./data/origin_label")
        assert len(origin_file_list) == len(label_file_list)
        return origin_file_list

    def load_np(self, idx):
        imgs = np.fromstring(open(os.path.join("./data/origin", idx), "rb").read(), dtype=np.uint8)
        ys = np.fromstring(open(os.path.join("./data/origin_label", idx), "rb").read(), dtype=np.uint8)
        return imgs.reshape(-1, self.height, self.width, 3), ys.reshape(-1, self.height // 2, self.width // 2)
    
    def bwGlobal(self, imgGray):
        maxval = 255
        thresholdType = cv2.THRESH_BINARY

        # Global Binarization
        noStep = 3
        i = np.random.randint(noStep)
        
        if i == 0:
            thresh = 100
        elif i == 1:
            thresh = 128
        ################# 너무 심한거 같은디...?! #############
        else:
            thresh = 156
        
        ret, imgBW = cv2.threshold(imgGray, thresh, maxval, thresholdType)
        # img 확인
        # cv2.imwrite(os.path.join('result-bwGlobal-{}.jpg'.format(thresh)), imgBW)
        return imgBW
        
    
    def bwAdaptive(self, imgGray, bw_type='Gaussian'):
        '''
        img Gray에 대해 적용
        bw_type = 'Gaussian' or 'Mean'
        '''
        
        maxval = 255
        thresholdType = cv2.THRESH_BINARY
        
        # Adaptive thresholding
        blockSize = 11

        noStep = 4
        i = np.random.randint(noStep)
        if i == 0:
            C = 3
        elif i == 1:
            C = 5
        elif i == 2:
            C = 7
        else:
            C = 10
        
        if bw_type == 'Gaussian':
            imgBW = cv2.adaptiveThreshold(imgGray, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType, blockSize, C)
            # img 확인
            # cv2.imwrite(os.path.join('result-bwAdtMean-{}-{}.jpg'.format(blockSize, C)), imgBW)
        elif bw_type == 'Mean':
            imgBW = cv2.adaptiveThreshold(imgGray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType, blockSize, C)
            # img 확인
            # cv2.imwrite(os.path.join('result-bwAdtGaussian-{}-{}.jpg'.format(blockSize, C)), imgBW)
        return cv2.cvtColor(imgBW, cv2.COLOR_GRAY2RGB)
    
    def saturation(self, img):
        
        imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        (orgH, orgS, orgV) = cv2.split(imgHSV)
        orgS = orgS.astype("float32")

        noStep = 7
        i = np.random.randint(noStep)
        
        if i == 0:
            factor = 0.25
        elif i == 1:
            factor = 0.5
        elif i == 2:
            factor = 0.75
        elif i == 3:
            factor = 1.0
        elif i == 4:
            factor = 1.25
        elif i == 5:
            factor = 1.5
        else:
            factor = 1.75

        s = np.clip(orgS * factor, 0, 255).astype(np.uint8)
        imgHSV = cv2.merge([orgH, s, orgV])
        imgSaturated = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB).astype(np.uint8)
        # img 확인
        # cv2.imwrite(os.path.join('result-Saturation-{}.jpg'.format(factor)), cv2.cvtColor(imgSaturated, cv2.COLOR_RGB2BGR))
        return imgSaturated
    
    def contrast_brightness(self, img):
        const_start = 0.25
        const_end = 2.0
        contrastStep = 0.25
        brightnessStep = 25

        contrastRange = np.arange(const_start, const_end + contrastStep, contrastStep)
        # print(contrastRange)

        c = contrastRange[np.random.randint(len(contrastRange))]

        if c == 0.25:
            brightness_start = 0
            brightness_end   = 100
        elif c == 0.5:
            brightness_start = -25
            brightness_end   = 100
        elif c == 0.75:
            brightness_start = -50
            brightness_end   = 100
        elif c == 1.0:
            brightness_start = -75
            brightness_end   = 100
        elif c == 1.25:
            brightness_start = -100
            brightness_end   = 75
        elif c == 1.5:
            brightness_start = -100
            brightness_end   = 25
        elif c == 1.75:
            brightness_start = -100
            brightness_end   = 0
        else:
            brightness_start = -100
            brightness_end   = -50

        brightnessRange = np.arange(brightness_start, brightness_end + brightnessStep, brightnessStep)
        #     print(brightnessRange)
        b = brightnessRange[np.random.randint(len(brightnessRange))]
        #         print("contrast ", c, "- brightness ", b)
        imgAdjusted = np.clip(img.astype(np.float32) * c + b, 0, 255).astype(np.uint8)
        # img 확인
        # cv2.imwrite(os.path.join('result-Contrast-{}-Brightness-{}.jpg'.format(c, b)), cv2.cvtColor(imgAdjusted, cv2.COLOR_RGB2BGR))
        
        return imgAdjusted
    
    def blur(self, img):
        
        noStep = 4        
        sigmaX = np.random.randint(2,2+noStep)
        
        npImgGray = np.array(img)
        npImgBlur = cv2.GaussianBlur(npImgGray, (0, 0), sigmaX)
        # img 확인
        # cv2.imwrite(os.path.join('result-GaussianBlur-{}.jpg'.format(sigmaX)), cv2.cvtColor(npImgBlur, cv2.COLOR_RGB2BGR))
        return npImgBlur
        
        
    def noise_generate(self, img):
        # ys는 변경 안함
        '''
        noise type에 있는 noise들을
        random한 종류와 순서로 배치하여 
        noise 생성
        
        현재 4가지 noise만 반영시킴
        '''
        # define noise_types
        noise_types = ['BW', 'Saturation', 'ContrastBrightness', 'Blur']
        
        # noise_types => shuffle
        np.random.shuffle(noise_types)
        
        # length 정하기
        type_length = np.random.randint(1, len(noise_types))
        noise_types = noise_types[:type_length]
        # print(noise_types)
        
        # 순서에 맞게 처리!!
        for noise_type in noise_types:
            # BW
            if noise_type == 'BW': 
                # Color to Gray
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # Gray to BW
                bw_type_list = ['Gaussian', 'Mean']
                img = self.bwAdaptive(img, bw_type=bw_type_list[np.random.randint(len(bw_type_list))])
    
            # Saturation
            if noise_type == 'Saturation':
                img = self.saturation(img)

            # contrast_brightness
            if noise_type == 'ContrastBrightness':
                img = self.contrast_brightness(img)
            
            # blur
            if noise_type == 'Blur':
                img = self.blur(img)
        
        # random noise img 결과 확인        
        # cv2.imwrite('result-random-noise-sample.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))        
        return img
    
    
    def transform(self, imgs, ys):
        assert imgs.shape[0] == ys.shape[0]
        batch = imgs.shape[0]
        
        for b in tqdm.tqdm(range(batch)):
            ################### random noise generate ###################
            imgs[b] = self.noise_generate(imgs[b])
            #############################################################
            
            if b % 2 == 0:
                curve_random = random.randint(5, 20)
                curve_mode = "down" if curve_random > 10 else "up"
                tran = Curve(width=self.width, height=self.height, spacing=100,
                          flexure=curve_random/100, direction=curve_mode)
            else:
                folded_up = random.randint(5, 20)
                folded_down = random.randint(5, 20)
                tran = Folded(width=self.width, height=self.height, spacing=100,
                              up_slope=folded_up/100, down_slope=folded_down/100)
            
            imgs[b] = tran.run(3, imgs[b], ipt_format="opencv", opt_format="opencv")
            y = tran.run(1, cv2.resize(ys[b], (self.width, self.height)).reshape(self.height, self.width, 1),
                         ipt_format="opencv", opt_format="opencv")
            ys[b] = cv2.resize(y, (self.width // 2, self.height // 2))

        #heatmap_img = cv2.applyColorMap(ys[0], cv2.COLORMAP_JET)
        #cv2.imwrite("label_curved.png", heatmap_img)
        #cv2.iimwrite("origin_curved.png", imgs[0])
        return imgs, ys

    def save(self, imgs, ys, idx=0):
        origin_path = "./data/origin_noise"
        label_path = "./data/origin_noise_label"
        if not os.path.exists(origin_path):
            os.makedirs(origin_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        with open(os.path.join(origin_path, str(idx)), "wb") as fout:
            fout.write(imgs.tostring())
        with open(os.path.join(label_path, str(idx)), "wb") as fout:
            fout.write(ys.tostring())

    def run(self):
        for batch in self.load_files():
            imgs, ys = self.load_np(batch)
            print(batch, imgs.shape, ys.shape)
            imgs, ys = self.transform(imgs, ys)
            self.save(imgs, ys, batch)
            
            
if __name__ == '__main__':
    p = Pipeline()
    p.run()
