import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
# import cv2
import string


def strGen(text_type_list):
    text_list = []
    type_style_list = ['-', '/', ',', ' ', '~', ':', ';', '.', '_', '='] # 등

    for text_type in text_type_list:
        text = ''
        text_script_list = text_type[0]
        fixed_len = text_type[1]
        
        # text script 설정
        text_script = text_script_list[np.random.randint(len(text_script_list))]
        
        # length 결정
        if fixed_len == True:
            text_length = len(text_script)
        else:
            text_length = np.random.randint(len(text_script))

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

            else:
                print("잘못된 문자 형식이 입력되었습니다.")
        text_list.append(text)
        
    return text_list

def draw_text(template, font, text, xy, fontsize=40):
    x,y = xy
    font = ImageFont.truetype(font, fontsize)
    if isinstance(template,str):
        img = Image.open(template)
    else:
        img = template
    b,g,r,a = 255,255,255,0
    #fontpath = "fonts/gulim.ttc"
    #font = ImageFont.truetype(fontpath, 20)
    img_pil = img
    draw = ImageDraw.Draw(img_pil)
    # fill ???
    draw.text((x, y),  text, font=font, fill=100)
    return img_pil

def imgGen(img, text_list, location_list, fontsize_list):
    '''
    현재 font size로 크기를 조절하지만
    추후에 boxing size를 통해 크기를 조절하도록 수정해야함
    '''
    
    # 현재 동일한 font 모양으로 고정
    font_type = 'fonts/batang.ttc'
    
    for i in range(len(text_list)):
        img = draw_text(img, font_type, text=text_list[i], xy=location_list[i], fontsize=fontsize_list[i])
        
    return img


if __name__=='__main__':
    
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
    text_type_list = [text_type_0, text_type_1, text_type_2, text_type_3, text_type_4,
                      text_type_5, text_type_6, text_type_7, text_type_8]
    location_list = [(330, 220), (1050, 480), (500, 480), (200, 750), (130, 750),
                     (450, 750), (1100, 750), (1370, 750), (750, 1720)]
    fontsize_list = [30, 30, 30, 25, 25, 25, 25, 25, 25]
    
    #******************************************************************
    
    text_list = strGen(text_type_list)
    img = Image.open('template/건강보험자격득실확인서 포맷.png')
    img = imgGen(img, text_list, location_list, fontsize_list)
    img.save('gen_sample/건강보험자격득실확인서_1_gen.png')