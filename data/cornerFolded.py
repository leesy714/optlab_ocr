from base_transform import Transform
import numpy as np
import cv2


###################### 여러번 접는 것은 tran.transform_image 여러번 하는 것으로 진행 #######################

class cornerFolded(Transform):
    def __init__(self, width, height, sub_width=500, sub_height=500, shift_length=150):
        super().__init__(width, height)        
        ################### 접히는 길이 조절 ###################
        self.sub_width = sub_width
        self.sub_height = sub_height
        
        ################### 접히는 정도 조절 ###################
        self.shift_length = shift_length
        self.deg = 1/3
        #####################################################
        
    def transform_points(self, points):
        return points
    
    def transform_image(self, ipt, option):
        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        option, degree = option.split('-')[0], option.split('-')[1]
        print(option, degree)
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
            deg = self.deg
        
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
    
    
if __name__ == "__main__":
    
    imgPers = cv2.imread('sample1.png')
    height, width = imgPers.shape[:2]
    tran = cornerFolded(width, height)
    for option in ['lt-s', 'lt-w', 'rt-s', 'rt-w', 'lb-s', 'lb-w', 'rb-s', 'rb-w']:
        image = tran.transform_image(ipt=imgPers, option=option)
        cv2.imwrite('cornerFolded_sample_{}.png'.format(option), image)
    
    
    