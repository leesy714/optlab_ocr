from base_transform import Transform
import numpy as np
import cv2

class cornerFolded(Transform):
    def __init__(self, width, height, sub_width=500, sub_height=500, shift_length=150):
        super().__init__(width, height)        
        ################### 접히는 길이 조절 ###################
        self.sub_width = sub_width
        self.sub_height = sub_height
        
        ################### 접히는 정도 조절 ###################
        self.shift_length = shift_length
        self.deg_ratio = 1/3
        #####################################################
        
    def transform_points(self, points):
        return points
    
    def transform_image(self, ipt, option):
        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        option, degree = option.split('-')[0], option.split('-')[1]
        #print(option, degree)
        deg = 1
        
        start_width = 0
        start_height = 0
        mul = 1
        
        if option == 'lt':
            start_width = 0
            start_height = 0
            mul = 1
        if option == 'rt':
            start_width = self.width-self.sub_width
            start_height = 0
            mul = 1
        if option == 'lb':
            start_width = 0
            start_height = self.height-self.sub_height
            mul = -1
        if option == 'rb':
            start_width = self.width-self.sub_width
            start_height = self.height-self.sub_height
            mul = -1        
        
        temp = origin[start_height:start_height+self.sub_height, start_width : start_width+self.sub_width, :].copy()
        
        if degree == 'w':
            deg = self.deg_ratio
        
        if option == 'lt' or option == 'rb':
            pts1 = np.array([[0, 0],
                             [self.sub_width, 0],
                             [self.sub_width, self.sub_height],
                             [0, self.sub_height]], dtype = "float32")
            pts2 = np.array([[self.shift_length*deg, self.shift_length*deg],
                             [self.sub_width, 0],
                             [self.sub_width-self.shift_length*deg, self.sub_height-self.shift_length*deg],
                             [0, self.sub_height]], dtype = "float32")
        
        if option == 'rt' or option == 'lb':
            pts1 = np.array([[0, 0],
                             [self.sub_width, 0],
                             [self.sub_width, self.sub_height],
                             [0, self.sub_height]], dtype = "float32")
            pts2 = np.array([[0, 0],
                             [self.sub_width-self.shift_length*deg, self.shift_length*deg],
                             [self.sub_width, self.sub_height],
                             [self.shift_length*deg, self.sub_height-self.shift_length*deg]], dtype = "float32")
        
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        temp = cv2.warpPerspective(temp, M, (self.sub_width, self.sub_height))
        
        # 합성
        out = origin.copy()
        
        if option == 'lt' or option == 'rb':
            for i in range(0, self.sub_height):
                for j in range(0, self.sub_width):
                    if mul*(i+j) <= mul*(self.sub_height+self.sub_width)/2:
                        out[start_height+i:start_height+i+1,start_width+j:start_width+j+1,:] = temp[i:i+1,j:j+1,:]
        if option == 'rt' or option == 'lb':
            for i in range(0, self.sub_height):
                for j in range(0, self.sub_width):
                    if mul*(j-i) >= 0:
                        out[start_height+i:start_height+i+1,start_width+j:start_width+j+1,:] = temp[i:i+1,j:j+1,:]
        
        return out

class cornerCurved(Transform):
    
    def __init__(self, width, height, sub_width=200, sub_height=200, shift_length=10, window_raise=50):
        super().__init__(width, height)
        
        ################## 휘는 정도 조절을 위한 hyperparameter #################
        
        self.sub_width = sub_width
        self.sub_height = sub_height
        self.shift_length = shift_length
        self.window_raise = window_raise
        
        self.deg_ratio = 1
        self.iter_folded = 10
        
        #####################################################################
        
    def transform_points(self, points):
        return points
    
    def transform_image(self, ipt, option):
        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        all_option = option
        option, degree = option.split('-')[0], option.split('-')[1]
        
        deg = 1
        if degree == 'w':
            deg = self.deg_ratio
            
        for i in range(0,int(self.iter_folded*deg)):
            tran = cornerFolded(self.width, self.height, sub_width=int(self.sub_width), sub_height=int(self.sub_height), shift_length=int(self.shift_length))
            origin = tran.transform_image(ipt=origin, option=all_option)
            #cv2.imwrite('cornerCurved_sample_{}.png'.format(i), origin)
            self.sub_width += self.window_raise
            self.sub_height += self.window_raise
            self.shift_length += self.window_raise/10
            
        return origin
    
if __name__ == "__main__":
    
    # 접기
    imgPers = cv2.imread('sample1.png')
    height, width = imgPers.shape[:2]
    for option in ['lt-s', 'lt-w', 'rt-s', 'rt-w', 'lb-s', 'lb-w', 'rb-s', 'rb-w']:
        tran = cornerFolded(width, height)
        image = tran.transform_image(ipt=imgPers, option=option)
        cv2.imwrite('cornerFolded_sample_{}.png'.format(option), image)
        
    # 말림
    imgPers = cv2.imread('sample1.png')
    height, width = imgPers.shape[:2]
    for option in ['lt-s', 'lt-w', 'rt-s', 'rt-w', 'lb-s', 'lb-w', 'rb-s', 'rb-w']:
        tran = cornerCurved(width, height)
        image = tran.transform_image(ipt=imgPers, option=option)
        cv2.imwrite('cornerCurved_sample_{}.png'.format(option), image)
        
       