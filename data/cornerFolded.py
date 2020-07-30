from base_transform import Transform
import numpy as np
import cv2


###################### 여러번 접는 것은 tran.transform_image 여러번 하는 것으로 진행 #######################

class cornerFolded(Transform):
    def __init__(self, width, height):
        super().__init__(width, height)        
        
    def transform_points(self, points):
        return points
    
    def transform_image(self, ipt, option):
        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        option, degree = option.split('-')[0], option.split('-')[1]
        print(option, degree)
        deg = 1
        
        sub_width, sub_height = 500, 500
        shift_length = 100
        temp = origin[0:sub_height, 0:sub_width, :].copy()
        
        if degree == 'w':
            deg = 1/2
        
        if option == 'lt':
            pts1 = np.array([[1, 0], [sub_width, 0], [0, sub_height], [0, 1]], dtype = "float32")
            pts2 = np.array([[np.ceil((shift_length+1)*deg), np.ceil(shift_length*deg)], [sub_width, 0],
                             [0, sub_height], [ np.ceil(shift_length*deg),  np.ceil((shift_length+1)*deg)]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        temp = cv2.warpPerspective(temp, M, (sub_width, sub_height))
        
        # 합성
        out = origin.copy()
        ##################### oprtion에 따라 간단하게 처리되도록 만들 필요 있음 ##################
        if option == 'lt':
            for i in range(0, sub_height):
                for j in range(0, sub_width):
                    if i+j <= (sub_height+sub_width)/2:
                        out[i:i+1,j:j+1,:] = temp[i:i+1,j:j+1,:]                     
        
        return out
    
    
    
if __name__ == "__main__":
    
    imgPers = cv2.imread('sample1.png')
    height, width = imgPers.shape[:2]
    tran = cornerFolded(width, height)
    image = tran.transform_image(ipt=imgPers, option='lt-s')
    
    cv2.imwrite('cornerFoldet_sample.png', image)
    

    