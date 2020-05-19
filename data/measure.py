import numpy as np
import matplotlib.pyplot as plt
import cv2


def denoise_measure(org_img, noise_img, output_img):
    
    diff_img = org_img - noise_img
    noise_ps_dict = {}   # noise 위치 dict
    denoise_score = 0    # score 높을 수록 denoise 된 수치 높음
    damaged_score = 0    # score 높을 수록 damage 입은 수치 높음
    
    # noise 위치 정보
    for i in range(org_img.shape[0]):
        for j in range(org_img.shape[1]):
            for k in range(org_img.shape[2]):
                if org_img[i][j][k] != diff_img[i][j][k]:
                    noise_ps_dict[(i,j,k)] = True
                else:
                    noise_ps_dict[(i,j,k)] = False
                    
    org_img, noise_img, output_img = org_img.astype(float), noise_img.astype(float), output_img.astype(float)
    
    # 현재 차이 생기면 무조건 +효과
    # denoise score의 경우 값이 커지고 작아지는 것에 대한 고려 필요
    # damaged score는 변화에 대한 조건이므로 괜춘
    
    # 이 measure를 denoising model 자체에 붙이는 것도 괜찮을 듯 ????
    
    # denoise_score
    for i in range(noise_img.shape[0]):
        for j in range(noise_img.shape[1]):
            for k in range(noise_img.shape[2]):
                if noise_ps_dict[(i,j,k)] == True:
                    denoise_score += abs(output_img[i][j][k] - noise_img[i][j][k])
                else:
                    damaged_score += abs(output_img[i][j][k] - org_img[i][j][k])
    
    return denoise_score, damaged_score       


if __name__ == "__main__":
    org_img = cv2.imread('imgs/origin/0.jpg')
    noise_img = cv2.imread('imgs/origin_noise/0.jpg')
    denoise_score, damaged_score = denoise_measure(org_img, noise_img, org_img)
    print('denoise score :',denoise_score)   # noise 하나도 못없앰
    print('damaged score :',damaged_score)   # 피해하나도 없음