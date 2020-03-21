import os
import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

class RandomText:

    def __init__(self, format_path, text_type_list, location_list, 
                 output_num=1, batch_size=256, width=960, height=1280):

        self.text_type_list = text_type_list
        self.batch_size = batch_size
        self.output_num = output_num
        self.img = Image.open(format_path).convert("RGB")
        print("img size: ", self.img.size)
        self.location_list = self.location_scaler(location_list, width, height)        
        # self.fontsize_list = [int(f*width/self.img.size[0]) for f in fontsize_list]
        self.img = self.img.resize((width, height))
        self.width = width
        self.height = height
    
    def location_scaler(self, location_list, width, height):
        scaled_list = []
        for x1y1, x2y2 in location_list:
            scaled_list.append([[int(x1y1[0]*width/self.img.size[0]), int(x1y1[1]*height/self.img.size[1])],
                                [int(x2y2[0]*width/self.img.size[0]), int(x2y2[1]*height/self.img.size[1])]])
        return scaled_list
    
    def str_generate(self):
        text_list = []
        type_style_list = ['-', '/', ',', ' ', '~', ':', ';', '.', '_', '=']

        for text_type in self.text_type_list:
            text_script_list = text_type[0]
            text_script = random.choice(text_script_list)

            fixed_len = text_type[1]
            text_length = len(text_script) if fixed_len else np.random.randint(len(text_script))

            text = ""
            for i in range(text_length):
                x = text_script[i]
                if x == '0':
                    text += str(np.random.randint(10))

                elif x == 'a':
                    text += string.ascii_letters[np.random.randint(len(string.ascii_letters))]
                elif x == 'h':
                    a = np.random.randint(0, 19)
                    b = np.random.randint(0, 21)
                    c = np.random.randint(0, 28)
                    text += chr(0xAC00 + a*0x24C + b*0x1C + c)

                elif x in type_style_list:
                    text += x

            text_list.append(text)
        return text_list


    def draw_text(self, idx, img, y, font, text, xy):
        
        # make font size list 10 - 70
        font_size_sample_list = [int(70 - i*2) for i in range(0,35)]
        fontsize = 25
        for fs in font_size_sample_list:
            font_ = ImageFont.truetype(font, fs)
            font_size = font_.getsize(text)
            if abs(xy[1][0] - xy[0][0]) >= font_size[0]:
                if abs(xy[1][1] - xy[0][1]) >= font_size[1]:
                    fontsize = fs
                    break
        print('text', text, 'fontsize :', fontsize)            
        font = ImageFont.truetype(font, fontsize)
        
        y[xy[0][0]: xy[1][0], xy[0][1]: xy[1][1]] = idx
        
        draw = ImageDraw.Draw(img)
        draw.text(xy[0], text, font=font, fill="black")
        return img, y

    def save(self, imgs, ys, idx=0):
        origin_path = "data/origin"
        label_path = "data/origin_label"
        if not os.path.exists(origin_path):
            os.makedirs(origin_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        with open(os.path.join(origin_path, str(idx)), "wb") as fout:
            fout.write(imgs.tostring())
        with open(os.path.join(label_path, str(idx)), "wb") as fout:
            fout.write(ys.tostring())

    def run(self):
        font = 'fonts/NanumGothic.ttf'

        batches = self.output_num // self.batch_size + 1
        img_size = self.img.size
        for batch in tqdm.tqdm(range(batches)):
            data_len = min(self.output_num - self.batch_size * batch, self.batch_size)
            if data_len == 0:
                continue
            imgs = np.zeros((data_len, img_size[1], img_size[0], 3), dtype=np.uint8)
            ys = np.empty((data_len, img_size[1] // 2, img_size[0] // 2), dtype=np.uint8)

            for _ in range(data_len):
                img = self.img.copy()
                size = img.size
                y = np.zeros(img.size)
                text_list = self.str_generate()
                for idx, (text, xy) in enumerate(zip(text_list, self.location_list)):
                    img, y = self.draw_text(idx, img, y, font, text, xy)
                
                img.save("res.png")
                map_img = np.clip(y.transpose() * (255 / len(self.location_list)), 0, 255).astype(np.uint8)
                heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
                cv2.imwrite("label.png", heatmap_img)

                imgs[_] = np.asarray(img)
                ys[_] = cv2.resize(y.transpose(), (self.width // 2, self.height //2))
            
            self.save(imgs, ys, batch)
            
            
if __name__ == '__main__':
    #***** 특정 field를 미리 정해놓고 box별로 field를 입력해주는 경우 *****

    #*********************** UI를 통해 받아야할 자료 ********************
    # 성명
    name_type = (['hh', 'hhh', 'hhhh'], True)

    # 생년월일
    day_type = (['0000.00.00', '0000/00/00', '0000-00-00'], True)

    # 신청 기간
    period_type = (['0000.00.00 ~ 0000.00.00', '0000/00/00 ~ 0000/00/00', '0000-00-00 ~ 0000-00-00',
                   '0000.00.00 - 0000.00.00', '0000/00/00 - 0000/00/00', '0000-00-00 - 0000-00-00'], True)

    # 긴급 연락처
    phonenum_type = (['00000000000', '000-0000-0000'], True)

    # 연 월 일
    num2_type = (['00'], True)


    # list 형태로 받을 것이라고 가정
    # 성명 / 생년월일 / 편입일자 / 신청기간 / 긴급연락처 / 연 / 월 / 일/ 이름
    text_type_list = [name_type, day_type, day_type, period_type, phonenum_type, num2_type, num2_type, num2_type, name_type]

    location_list = [[[660, 425],[800, 470]],  [[1240, 430], [1406, 459]], [[660, 620], [822, 649]],
                     [[780, 840], [1148, 870]], [[880, 940],[1078, 969]], [[730, 1515], [766, 1545]],
                     [[800, 1515], [836, 1544]], [[880, 1515], [916, 1544]], [[1010, 1700], [1124, 1743]]]

    # fontsize_list = [30, 30, 30, 30, 30, 30, 30, 30, 30]

    #******************************************************************

    # location_list 에 두점을 넣으면 box크기에 맞는 글씨를 만들도록 수정
    random_text = RandomText("./data/format.png", text_type_list, location_list,
                             output_num=1, batch_size=1)
    img = random_text.run()