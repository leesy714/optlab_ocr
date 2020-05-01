import os
import tqdm
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import string

class RandomText:

    def __init__(self, format_path, text_type_lists, location_lists, fontsize_list,
                 output_num=1, batch_size=256, width=960, height=1280,
                 res_path="imgs"):

        self.batch_size = batch_size
        self.output_num = output_num
        self.img = Image.open(format_path).convert("RGB")
        print("img size: ", self.img.size)

        self.text_type_lists = text_type_lists
        self.location_lists = location_lists
        self.fontsize_lists = fontsize_lists

        self.org_img = self.img
        self.img = self.img.resize((width, height))
        self.width = width
        self.height = height

        self.res_path = res_path
        if not os.path.exists(res_path):
            os.makedirs(res_path)

    def str_generate(self, text_type_list):
        text_list = []
        type_style_list = ['-', '/', ',', ' ', '~', ':', ';', '.', '_', '=']

        for text_type in text_type_list:
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


    def draw_text(self, idx, img, y, font, text, xy, fontsize=40):
        font = ImageFont.truetype(font, fontsize)
        #print(text, font.getsize(text))
        font_size = font.getsize(text)
        to_xy = (xy[0] + font_size[0], xy[1] + font_size[1])
        y[xy[0]: to_xy[0], xy[1]: to_xy[1]] = idx
        draw = ImageDraw.Draw(img)
        draw.text(xy, text, font=font, fill="black")
        return img, y

    def save(self, imgs, ys, idx=0):
        origin_path = os.path.join(self.res_path, "origin")
        label_path = os.path.join(self.res_path, "origin_label")
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

                ############### 생성할 때마다 line 수 변경 #############
                gen_num = np.random.randint(len(self.text_type_lists))
                text_type_list = self.text_type_lists[gen_num]
                location_list = [(int(x*self.width/self.org_img.size[0]), int(y*self.height/self.org_img.size[1])) for x, y in self.location_lists[gen_num]]
                fontsize_list = [int(f*self.width/self.org_img.size[0]) for f in self.fontsize_lists[gen_num]]
                ######################################################

                img = self.img.copy()
                size = img.size
                y = np.zeros(img.size)
                text_list = self.str_generate(text_type_list)
                for idx, (text, xy, fontsize) in enumerate(zip(text_list, location_list, fontsize_list)):
                    idx = idx + 1
                    if idx > 8 + gen_num * 5:
                        idx = 9
                    elif idx > 8:
                        idx = 4 + (idx-4) % 5
                    img, y = self.draw_text(idx, img, y, font, text, xy, fontsize)

                # 예시 별로 이미지 출력
                #img.save("res_"+str(gen_num)+".png")

                #map_img = np.clip(y.transpose() * (255 / len(location_list)), 0, 255).astype(np.uint8)
                #map_img = np.clip(y.transpose() * (255 / 9), 0, 255).astype(np.uint8)
                #heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
                #cv2.imwrite("label.png", heatmap_img)

                imgs[_] = np.asarray(img)
                ys[_] = cv2.resize(y.transpose(), (self.width // 2, self.height //2))
            self.save(imgs, ys, batch)



if __name__=='__main__':

    # 건강보험자격득실서 text 생성

    #***************** box 별로 text type을 입력받는 경우 ***************

    #*********************** UI를 통해 받아야할 자료 ********************
    # 발급번호
    text_type_0 = (['a000000000000000'], True)
    # 주민 등록번호
    text_type_1 = (['000000-0000000'], True)
    # 한글 이름
    text_type_2 = (['hh', 'hhh', 'hhhh'], True)
    # 가입자 구분
    text_type_3 = (['hhhhhhhhh'], False)

    '''
    목록 list
    boxing을 모든 목록 칸에 해줬다고 가정하면
    여기부터 여기까지 목록이라는 표시 필요
    해당 목록 list를 몇 줄까지 Gen할 지를 random하게 정하도록 할 수 있음
    '''
    # 현재는 그냥 3줄 작성하도록 지시
    # NO
    text_type_4 = (['0'], True)
    # 사업자 명칭
    text_type_5 = (['hhhhhhhhhhhhhhhh'], False)
    # 자격취득일
    text_type_6 = (['0000.00.00', '0000/00/00', '0000-00-00'], True)
    # 자격상실일
    text_type_7 = (['0000.00.00', '0000/00/00', '0000-00-00'], True)


    # 날짜
    text_type_8 = (['0000.00.00', '0000/00/00', '0000-00-00'], True)

    # sample
    # list 형태로 받을 것이라고 가정
    text_type_list_0 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_0 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (750, 1720)]

    fontsize_list_0 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25]

    text_type_list_1 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_1 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (130, 830), (200, 830), (450, 830), (1100, 830), (1370, 830),
                       (750, 1720)]

    fontsize_list_1 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25]

    text_type_list_2 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_2 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (130, 830), (200, 830), (450, 830), (1100, 830), (1370, 830),
                       (130, 910), (200, 910), (450, 910), (1100, 910), (1370, 910),
                       (750, 1720)]

    fontsize_list_2 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25]

    text_type_list_3 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_3 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (130, 835), (200, 835), (450, 835), (1100, 835), (1370, 835),
                       (130, 920), (200, 920), (450, 920), (1100, 920), (1370, 920),
                       (130, 1005), (200, 1005), (450, 1005), (1100, 1005), (1370, 1005),
                       (750, 1720)]


    fontsize_list_3 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25]

    text_type_list_4 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_4 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (130, 835), (200, 835), (450, 835), (1100, 835), (1370, 835),
                       (130, 920), (200, 920), (450, 920), (1100, 920), (1370, 920),
                       (130, 1005), (200, 1005), (450, 1005), (1100, 1005), (1370, 1005),
                       (130, 1090), (200, 1090), (450, 1090), (1100, 1090), (1370, 1090),
                       (750, 1720)]


    fontsize_list_4 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25]

    text_type_list_5 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_5 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (130, 835), (200, 835), (450, 835), (1100, 835), (1370, 835),
                       (130, 920), (200, 920), (450, 920), (1100, 920), (1370, 920),
                       (130, 1005), (200, 1005), (450, 1005), (1100, 1005), (1370, 1005),
                       (130, 1090), (200, 1090), (450, 1090), (1100, 1090), (1370, 1090),
                       (130, 1175), (200, 1175), (450, 1175), (1100, 1175), (1370, 1175),
                       (750, 1720)]

    fontsize_list_5 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25]

    text_type_list_6 = [text_type_0, text_type_1, text_type_2,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_4, text_type_3, text_type_5, text_type_6, text_type_7,
                        text_type_8]

    location_list_6 = [(330, 220), (1050, 480), (500, 480),
                       (130, 750), (200, 750), (450, 750), (1100, 750), (1370, 750),
                       (130, 835), (200, 835), (450, 835), (1100, 835), (1370, 835),
                       (130, 920), (200, 920), (450, 920), (1100, 920), (1370, 920),
                       (130, 1005), (200, 1005), (450, 1005), (1100, 1005), (1370, 1005),
                       (130, 1090), (200, 1090), (450, 1090), (1100, 1090), (1370, 1090),
                       (130, 1175), (200, 1175), (450, 1175), (1100, 1175), (1370, 1175),
                       (130, 1260), (200, 1260), (450, 1260), (1100, 1260), (1370, 1260),
                       (750, 1720)]

    fontsize_list_6 = [30, 30, 30,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25, 25, 25, 25, 25,
                       25]



    text_type_lists = []
    location_lists = []
    fontsize_lists = []

    text_type_lists.append(text_type_list_0)
    text_type_lists.append(text_type_list_1)
    text_type_lists.append(text_type_list_2)
    text_type_lists.append(text_type_list_3)
    text_type_lists.append(text_type_list_4)
    text_type_lists.append(text_type_list_5)
    text_type_lists.append(text_type_list_6)

    location_lists.append(location_list_0)
    location_lists.append(location_list_1)
    location_lists.append(location_list_2)
    location_lists.append(location_list_3)
    location_lists.append(location_list_4)
    location_lists.append(location_list_5)
    location_lists.append(location_list_6)

    fontsize_lists.append(fontsize_list_0)
    fontsize_lists.append(fontsize_list_1)
    fontsize_lists.append(fontsize_list_2)
    fontsize_lists.append(fontsize_list_3)
    fontsize_lists.append(fontsize_list_4)
    fontsize_lists.append(fontsize_list_5)
    fontsize_lists.append(fontsize_list_6)


    #******************************************************************

    #################### 생성할 data에 따라 format.png 수정 ######################

    random_text = RandomText("sample1.png", text_type_lists, location_lists,
                             fontsize_lists, output_num=4000, batch_size=4)
    img = random_text.run()



