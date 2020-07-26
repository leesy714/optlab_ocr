from base_transform import Transform
import numpy as np
import cv2

class Perspective(Transform):

    def __init__(self, width, height, height_ratio=3/4, width_ratio=3/4, height_space_ratio=1/30):
        '''
        :param height_ratio : 기존 문서 높이 축소 비율
        :param width_ratio : 기존 문서 너비 축소 비율
        :param height_space_ratio : 높이 여백 비율
        :param width_space_ratio : 너비 여백 비율 (높이, 너비 비율이 맞게 설정)
        :param x_ : 좌(0), 우(1)
        :param y_ : 위(0), 중간(1), 아래(2)
        '''
        super().__init__(width, height)
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.height_space_ratio = height_space_ratio
        self.width_space_ratio = height_space_ratio * width/height
        self.x_ = np.random.randint(2)
        self.y_ = np.random.randint(3)

    def transform_points(self, points):
        t_points = []
        for point in points:
            
            # 축소
            x = point[0] * self.width_ratio
            y = point[1] * self.height_ratio

            # 위치 이동
            x, y = x+int(self.x_*(1-self.width_ratio)*self.width), y+int(self.y_*(1-self.height_ratio)*self.height/2)

            # 여백 남기기
            if self.x_ == 0:
                x += int(self.width * self.width_space_ratio)
            else:
                x -= int(self.width * self.width_space_ratio)

            if self.y_ == 0:
                y += int(self.height * self.height_space_ratio)
            elif self.y_ == 2:
                y -= int(self.height * self.height_space_ratio)

            t_points.append((int(x), int(y)))

        return t_points

    def run(self, channel, ipt, ipt_format="file", opt="res_pers.png", opt_format="file", base="Arles-15t.jpg", base_format="file"):


        self.load_ipt(ipt, ipt_format)
        if channel > 1:
            self.origin = cv2.cvtColor(self.origin, cv2.COLOR_BGR2BGRA)
        self.origin = cv2.resize(self.origin, dsize=(self.width, self.height))

        # test code #
#         img = np.clip(self.origin * (255 /9), 0 ,255)
#         heatmap_img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)
#         cv2.imwrite("origin_label.png", heatmap_img)
#         print("origin shape", self.origin.shape)
#         print(self.origin.sum())

        # Base img 받을 변수 필요 (self.base)
        # TODO: temporal black base image
        shape = (self.height, self.width, 3) if channel > 1 else (self.height, self.width)
        imgBase = np.zeros(shape, np.uint8)
        if channel > 1:
            imgBase = cv2.imread(base)
            imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2BGRA)
            imgBase = cv2.resize(imgBase, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
        #else:
        #    imgBase = np.zeros((self.height, self.width), np.uint8)
        
        # Distort perspective
        pts1 = np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])

        # 가운데로 축소 이후 변형시키기
        x = int(self.width*(1-self.width_ratio)/2)
        y = int(self.height*(1-self.height_ratio)/2)
        width = self.width_ratio*self.width
        height = self.height_ratio*self.height
        
        ################################## option 설정 ####################################
        option = "rm"
        degree = "w"
        ##################################################################################
        
        if degree == "s":
            deg = 1/16 
        elif degree == "w":
            deg = 1/8
        else:
            raise
        
        if option == "lt":
            pts2 = np.float32([[x, y], [x + width*(1-deg), y + height*deg/2],
                               [x + width*deg/2, y + height*(1-deg)], [x + width, y + height]])
        elif option == "ct":
            pts2 = np.float32([[x + width*deg, y], [x + width*(1-deg), y],
                               [x, y + height], [x + width, y + height]])
        elif option == "rt":
            pts2 = np.float32([[x + width*deg, y + height*deg/2], [x+width, y],
                               [x, y + height], [x + width*(1-deg/2), y + height*(1-deg)]])
        elif option == "rm":
            pts2 = np.float32([[x, y], [x + width, y + height*deg],
                               [x, y + height], [x + width, y + height*(1-deg)]])
        elif option == "rb":
            pts2 = np.float32([[x, y], [x + width*(1-deg/2), y + height*deg],
                               [x + width*deg, y + height*(1-deg/2)], [x + width, y + height]])
        elif option == "cb":
            pts2 = np.float32([[x, y], [x + width, y],
                               [x + width*deg, y + height], [x + width*(1-deg), y + height]])
        elif option == "lb":
            pts2 = np.float32([[x + width*deg/2, y + height*deg], [x+width, y],
                               [x, y+height], [x + width*(1-deg), y + height*(1-deg/2)]])
        elif option == "lm":
            pts2 = np.float32([[x, y + height*deg], [x+width, y],
                               [x, y + height*(1-deg)], [x+width, y + height]])
        else:
            raise
        
        ################### rm lm ct cb의 경우 folded 적용하기 ##################
        
        # Folded
        
        ######################################################################
        
        
        M_redistort = cv2.getPerspectiveTransform(pts1,pts2)

        self.origin = cv2.warpPerspective(self.origin, M_redistort, (self.width,self.height), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])

        x1, x2 = 0, self.width
        y1, y2 = 0, self.height
        
        if channel > 1:
            alpha_p = self.origin[:, :, 3] / 255.0
        else:
            alpha_p = self.origin[:, :] / 255.0
        alpha_b = 1.0 - alpha_p

        if channel > 1:
            for c in range(0, 3):
                imgBase[y1:y2, x1:x2, c] = (alpha_p[y1:y2, x1:x2] * self.origin[y1:y2, x1:x2, c] + alpha_b[y1:y2, x1:x2] * imgBase[y1:y2, x1:x2, c])
        else:
            imgBase[y1:y2, x1:x2] = self.origin[y1:y2, x1:x2] + imgBase[y1:y2, x1:x2]
        
        #cv2.imwrite(opt, imgBase)
        return self.save(imgBase, opt, opt_format)

        # 이미지 확인
        # plt.figure()
        # plt.imshow(cv2.cvtColor(imgBase, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    # 예시 이미지
    imgPers = cv2.imread('sample1.png')
    height, width = imgPers.shape[:2]
    tran = Perspective(width, height)
    tran.run(3, ipt="sample1.png")
    # cv2.imwrite("perspective_test.png", y)
#     tran.transform_points([(0,0), (width,0), (0, height), (width, height)])

    width, height = 960, 1280
    y = np.load("imgs/origin_label/000000.npy")
    print("origin label shape", y.shape)
    heatmap_img = np.clip(y.transpose(1, 0) * (255 /9), 0 ,255).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    cv2.imwrite("origin_label.png", heatmap_img)

    y = cv2.resize(y.transpose(1, 0), (width, height))
    tran = Perspective(width, height)
    y = tran.run(1, np.expand_dims(y, 2), ipt_format="opencv", opt_format="opencv")
    #cv2.imwrite("perspective_test.png", y)
    # y = cv2.resize(y, (width // 2, height // 2))
    # test code #
    y = np.clip(y * (255 /9), 0 ,255).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(y, cv2.COLORMAP_JET)
    cv2.imwrite("label_pers.png", heatmap_img)


