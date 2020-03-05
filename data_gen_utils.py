import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2





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
    draw.text((x, y),  text, font=font,fill=1)
    return img_pil


def resize(img, w_new):

    w,h = img.size
    wph = w/h
    p_new = int(w_new / wph)
    return img.resize((w_new,p_new))


def salt_n_pepper(img, s_vs_p = 0.5, amount = 0.003):
    img = img.convert('RGB')
    img_arr = np.array(img)
    row,col,ch = img_arr.shape
    out = np.copy(img_arr)
    # Salt mode
    num_salt = np.ceil(amount * img_arr.size * s_vs_p)

    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in img_arr.shape[:2]]

    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* img_arr.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img_arr.shape[:2]]
    out[coords] = 0
    return Image.fromarray(out).convert('L')


def rotate(img, deg):
    img = img.convert('RGBA')
    rot = img.rotate(deg)


    fff = Image.new('RGBA', rot.size, (255,)*4)
# create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot)

# save your work (converting back to mode='1' or whatever..)
    return out


if __name__=='__main__':
    x,y=680,420
    img = Image.open('template/template-1.png')
    img = draw_text(img, 'ttf/batang.ttf', text='홍길동',xy=(x,y))
    img = draw_text(img, 'ttf/batang.ttf', text='123asb',xy=(315,260))
    img=resize(img, 600)
    img = salt_n_pepper(img)
    img = rotate(img,-5)
    img.save('out.png')





