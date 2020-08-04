import abc
import cv2
import numpy as np


class Transform(metaclass=abc.ABCMeta):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    @abc.abstractmethod
    def transform_image(self, image):
        pass

    @abc.abstractmethod
    def transform_points(self, points):
        pass

class Shift(Transform):

    def __init__(self, width, height, spacing=20, is_horizon=True):
        """
        :param spacing: length of each square pixel
        """
        super().__init__(width, height)
        self.spacing = spacing
        self.is_horizon = is_horizon
        if not self.is_horizon:
            self.width = height
            self.height = width

    def slide(self, h):
        raise NotImplemented

    def get_transform(self):
        widths = [[w for h in range(0, self.height, self.spacing)]
                  for w in range(0, self.width, self.spacing)]
        heights = [[h for h in range(0, self.height, self.spacing)]
                   for w in range(0, self.width, self.spacing)]
        return widths, heights

    def transform_points(self, points):
        if points is None:
            return None
        points = points.reshape(-1, 2)
        points = [(x[0], x[1]) for x in points]
        if not self.is_horizon:
            points = [(y, x) for x, y in points]
        t_points = []
        widths, heights = self.get_transform()
        for point in points:
            w = point[0] // self.spacing
            h = point[1] // self.spacing
            width = widths[w][h]
            height = heights[w][h]
            src = np.array([[width, height],
                            [width + self.spacing, height],
                            [width + self.spacing, height + self.spacing],
                            [width, height + self.spacing]], np.float32)
            dst = np.array([[width + self.slide(h), height],
                            [width + self.spacing + self.slide(h), height],
                            [width + self.spacing + self.slide(h+1), +height + self.spacing],
                            [width + self.slide(h+1), height + self.spacing]], np.float32)
            mat = cv2.getPerspectiveTransform(src, dst)
            
            point = np.array((*point, 1))
            t_point = np.dot(mat, point)
            t_points.append((int(t_point[0]), int(t_point[1])))
        if not self.is_horizon:
            t_points = [(y, x) for x, y in t_points]
        t_points = np.array(t_points)
        t_points = t_points.reshape(-1, 4, 2)
        return t_points

    def transform_image(self, ipt, option, base="Arles-15t.jpg", channel=3):

        if self.is_horizon:
            origin = cv2.resize(ipt, dsize=(self.width, self.height))
        else:
            origin = cv2.resize(ipt, dsize=(self.height, self.width))
            origin = origin.transpose(1, 0, 2) if channel > 1 else origin.transpose(1, 0)

        if channel > 1:
            output = np.zeros((self.height, self.width, channel), np.uint8)
        else:
            output = np.zeros((self.height, self.width), np.uint8)

        widths, heights = self.get_transform()
        for w in range(int(self.width / self.spacing)):
            for h in range(int(self.height / self.spacing)):
                width = widths[w][h]
                height = heights[w][h]
                src = np.array([[width, height],
                                [width + self.spacing, height],
                                [width + self.spacing, height + self.spacing],
                                [width, height + self.spacing]], np.float32)
                dst = np.array([[width + self.slide(h), height],
                                [width + self.spacing + self.slide(h), height],
                                [width + self.spacing + self.slide(h+1), +height + self.spacing],
                                [width + self.slide(h+1), height + self.spacing]], np.float32)

                mat = cv2.getPerspectiveTransform(src, dst)
                warp = cv2.warpPerspective(origin, mat, (self.width, self.height))
                output[height: height + self.spacing, width:width + self.spacing] = warp[height: height + self.spacing, width:width+self.spacing]

        if not self.is_horizon:
            output = output.transpose(1, 0, 2) if channel > 1 else output.transpose(1, 0)
        return output

class Folded(Shift):

    def __init__(self, up_slope=0.1, down_slope=0.2, **kwargs):
        super().__init__(**kwargs)

        self.up_slope = up_slope
        self.down_slope = down_slope

    def slide(self, h):
        max_num = self.height / self.spacing
        x = abs(1 - 2 * h / max_num)
        if h < max_num / 2:
            return x * self.width * self.up_slope
        else:
            return x * self.width * self.down_slope

class Curve(Shift):

    def __init__(self, flexure=0.1, direction="down", **kwargs):
        super().__init__(**kwargs)

        self.flexure = flexure * self.width
        self.direction = direction

    def slide(self, h):
        if self.direction == "up":
            return (1 - h / float(self.height / self.spacing)) ** 2 * self.flexure
        elif self.direction == "down":
            return (h / float(self.height / self.spacing)) ** 2 * self.flexure
        else:
            raise KeyError


class Shadow(Transform):

    def __init__(self, width, height, spacing=20, focus=100, sources=[(0, 0)]):
        """
        :param width: width of the document
        :param height: height of the document
        :param spacing: length of each square pixel
        :param sources: light sources in list format
            e.g. [(0, 0), (0, 1280)]
        """
        super().__init__(width, height)
        self.spacing = spacing
        self.focus = focus
        assert len(sources) > 0, "at least one light source is required"
        self.sources = [(x * self.width, y * self.height) for x, y in sources]

    def get_regularizer(self):
        max_norm = 0.0
        for width, height in [(0, 0), (self.width, 0),
                              (0, self.height), (self.width, self.height)]:
            norms = [(width-x)**2 + (height-y)**2 for x, y in self.sources]
            norm = sum(norms)
            max_norm = max(max_norm, norm)
        return 10.0 / max_norm

    def transform_points(self, points):
        return points

    def transform_image(self, ipt, option, base="Arles-15t.jpg"):

        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        origin = origin.astype(np.float32)
        regularizer = self.get_regularizer()
        for w in range(0, self.width, self.spacing):
            for h in range(0, self.height, self.spacing):
                widths = slice(w, min(w+self.spacing, self.width))
                heights = slice(h, min(h+self.spacing, self.height))
                norms = [(w-x)**2 + (h-y)**2 for x, y in self.sources]
                norm = sum(norms)
                origin[heights, widths, :] -= int(self.focus / (norm * regularizer + 1))
        origin = origin.clip(min=0, max=255).astype(np.uint8)
        return origin

class Perspective(Transform):

    def __init__(self, width, height, options, height_ratio=3/4, width_ratio=3/4, height_space_ratio=1/30):
        '''
        :param height_ratio : 기존 문서 높이 축소 비율
        :param width_ratio : 기존 문서 너비 축소 비율
        :param height_space_ratio : 높이 여백 비율
        :param width_space_ratio : 너비 여백 비율 (높이, 너비 비율이 맞게 설정)
        '''
        super().__init__(width, height)
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio
        self.height_space_ratio = height_space_ratio
        self.width_space_ratio = height_space_ratio * width/height
        
        self.mat = self.make_matrix(options)
        
    def make_matrix(self, options):
        
        # Distort perspective
        pts1 = np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])
        
        option, degree = options.split('-')[0], options.split('-')[1]
        
        x = int(self.width*(1-self.width_ratio)/2)
        y = int(self.height*(1-self.height_ratio)/2)
        width = self.width_ratio*self.width
        height = self.height_ratio*self.height
        
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
        return cv2.getPerspectiveTransform(pts1, pts2)
        

    def transform_points(self, points):
        points = points.reshape(-1, 2)
        points = [(x[0], x[1]) for x in points]
        t_points = []
        for point in points:
            point = np.array((*point, 1))
            t_point = np.dot(self.mat, point)
            t_points.append((int(t_point[0]), int(t_point[1])))
        t_points = np.array(t_points)
        t_points = t_points.reshape(-1, 4, 2)
        return t_points
    
    # transform_image
    def transform_image(self, ipt, option, base="Arles-15t.jpg", channel=3):
        
        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        if channel > 1:
            origin = cv2.cvtColor(origin, cv2.COLOR_BGR2BGRA)
        origin = cv2.resize(origin, dsize=(self.width, self.height))

        # 배경 이미지 load
        # TODO: temporal black base image
        shape = (self.height, self.width, 3) if channel > 1 else (self.height, self.width)
        imgBase = np.zeros(shape, np.uint8)
        if channel > 1:
            imgBase = cv2.imread(base)
            imgBase = cv2.cvtColor(imgBase, cv2.COLOR_BGR2BGRA)
            imgBase = cv2.resize(imgBase, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
        #else:
        #    imgBase = np.zeros((self.height, self.width), np.uint8)
        
        # 이미지 변형
        
        origin = cv2.warpPerspective(origin, self.mat, (self.width,self.height), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
        
        # 배경과 합치기
        x1, x2 = 0, self.width
        y1, y2 = 0, self.height
        
        if channel > 1:
            alpha_p = origin[:, :, 3] / 255.0
        else:
            alpha_p = origin[:, :] / 255.0
        alpha_b = 1.0 - alpha_p

        if channel > 1:
            for c in range(0, 3):
                imgBase[y1:y2, x1:x2, c] = (alpha_p[y1:y2, x1:x2] * origin[y1:y2, x1:x2, c] + alpha_b[y1:y2, x1:x2] * imgBase[y1:y2, x1:x2, c])
        else:
            imgBase[y1:y2, x1:x2] = origin[y1:y2, x1:x2] + imgBase[y1:y2, x1:x2]
        
        #cv2.imwrite(opt, imgBase)
        return imgBase
        #return self.save(imgBase, opt, opt_format)

class cornerFolded(Transform):
    def __init__(self, width, height, sub_width=500, sub_height=500, shift_length=150):
        super().__init__(width, height)        
        ################### 접히는 길이 조절 ###################
        # 2337 1653 기준 hyper parmaeter (500 500 150)
        self.sub_width = sub_width
        self.sub_height = sub_height
        
        ################### 접히는 정도 조절 ###################
        self.shift_length = shift_length
        self.deg_ratio = 1/3
        #####################################################
        
    def transform_points(self, points):
        return points
    
    def transform_image(self, ipt, option, base="Arles-15t.jpg"):
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
        # 2337 1653 기준 hyper parmaeter (200 200 10 50)
        self.sub_width = sub_width
        self.sub_height = sub_height
        self.shift_length = shift_length
        self.window_raise = window_raise
        
        self.deg_ratio = 1
        self.iter_folded = 10
        
        #####################################################################
        
    def transform_points(self, points):
        return points
    
    def transform_image(self, ipt, option, base="Arles-15t.jpg"):
        origin = cv2.resize(ipt, dsize=(self.width, self.height))
        all_option = option
        option, degree = option.split('-')[0], option.split('-')[1]
        
        deg = 1
        if degree == 'w':
            deg = self.deg_ratio
        
        sub_width = self.sub_width
        sub_height = self.sub_height
        shift_length = self.shift_length
        
        for i in range(0,int(self.iter_folded*deg)):
            tran = cornerFolded(self.width, self.height, sub_width=int(sub_width), sub_height=int(sub_height), shift_length=int(shift_length))
            origin = tran.transform_image(ipt=origin, option=all_option)
            #cv2.imwrite('cornerCurved_sample_{}.png'.format(i), origin)
            sub_width += self.window_raise
            sub_height += self.window_raise
            shift_length += self.window_raise/10
            
        return origin
    
def option_reader(transformType, option):
    kwargs = {}

    if transformType is "1A":
        pass

    elif transformType is "1B":
        pass

    elif transformType is "1C":
        kwargs["is_horizon"] = True if "long" in option else False
        kwargs["flexure"] = 0.2 if option[-1] is "s" else 0.1

    elif transformType is "1D":
        kwargs["is_horizon"] = True if "long" in option else False
        kwargs["up_slope"] = 0.2 if option[-1] is "s" else 0.1
        kwargs["down_slope"] = 0.2 if option[-1] is "s" else 0.1

    elif transformType is "2A":
        pass                

    elif transformType is "2B":
        pos, strength = option.split("-")
        kwargs["focus"] = 100 if strength is "s" else 50
        if pos is "l":
            kwargs["sources"] = [(0.0, 0.5)]
        elif pos is "lt":
            kwargs["sources"] = [(0.0, 0.0)]
        elif pos is "t":
            kwargs["sources"] = [(0.5, 0.0)]
        elif pos is "rt":
            kwargs["sources"] = [(1.0, 0.0)]
        elif pos is "r":
            kwargs["sources"] = [(1.0, 0.5)]
        elif pos is "rb":
            kwargs["sources"] = [(1.0, 1.0)]
        elif pos is "b":
            kwargs["sources"] = [(0.5, 1.0)]
        elif pos is "lb":
            kwargs["sources"] = [(0.0, 1.0)]

    return kwargs

def transformImage(image, coords, transformType, option, base="Arles-15t.jpg"):

    height, width, channel = image.shape
    tran = None

    kwargs = option_reader(transformType, option)
    if transformType is "1A":
        tran = cornerFolded(width, height, sub_width=int(min(width, height)/3), sub_height=int(min(width, height)/3),
                            shift_length=int(min(width, height)/9))

    elif transformType is "1B":
        tran = cornerCurved(width, height, sub_width=int(min(width, height)/8), sub_height=int(min(width, height)/8),
                            shift_length=int(min(width, height)/160) , window_raise=int(min(width, height)/32))

    elif transformType is "1C":
        tran = Curve(width=width, height=height, spacing=40, **kwargs)

    elif transformType is "1D":
        tran = Folded(width=width, height=height, spacing=40, **kwargs)

    elif transformType is "2A":
        tran = Perspective(width=width, height=height, options=option)        

    elif transformType is "2B":
        tran = Shadow(width=width, height=height, spacing=40, **kwargs)

    assert isinstance(tran, Transform)
    transImage = tran.transform_image(image, option, base=base)
    transCoords = tran.transform_points(coords)
    
    return transImage, transCoords

def test():
    import os
    import tqdm
    if not os.path.exists("test_api"):
        os.makedirs("test_api")
    origin = cv2.imread("sample1.png")
    origin = cv2.resize(origin, dsize=(480, 640))
    coords = np.array([[[0, 0], [1, 0], [1, 1] ,[0, 1]]])

    # 1A
    print("START TRANSFORM 1A")
    options = ['lt-s', 'lt-w', 'rt-s', 'rt-w', 'lb-s', 'lb-w', 'rb-s', 'rb-w',
               'lt-s;rt-s;lt-w', 'lt-w;rb-w']
    for option in tqdm.tqdm(options):
        option_split = option.split(';')
        trans_image, trans_coords = origin.copy(), coords.copy()
        for opt in option_split:
            trans_image, trans_coords = transformImage(trans_image, trans_coords, "1A", opt)
        assert trans_coords.shape == (1, 4, 2)
        cv2.imwrite("test_api/1A_{}.png".format(option), trans_image)
    # 1B
    print("START TRANSFORM 1B")
    options = ['lt-s', 'lt-w', 'rt-s', 'rt-w', 'lb-s', 'lb-w', 'rb-s', 'rb-w',
               'lt-s;rt-s;lt-w', 'lt-w;rb-w']
    for option in tqdm.tqdm(options):
        option_split = option.split(';')
        trans_image, trans_coords = origin.copy(), coords.copy()
        for opt in option_split:
            trans_image, trans_coords = transformImage(trans_image, trans_coords, "1B", opt)
        assert trans_coords.shape == (1, 4, 2)
        cv2.imwrite("test_api/1B_{}.png".format(option), trans_image)
    # 1C
    print("START TRANSFORM 1C")
    options = ["long-s", "long-w", "short-s", "short-w"]
    for option in tqdm.tqdm(options):
        trans_image, trans_coords = transformImage(origin, coords, "1C", option)
        assert trans_coords.shape == (1, 4, 2)
        cv2.imwrite("test_api/1C_{}.png".format(option), trans_image)
    # 1D
    print("START TRANSFORM 1D")
    options = ["long-s", "long-w", "short-s", "short-w"]
    for option in tqdm.tqdm(options):
        trans_image, trans_coords = transformImage(origin, coords, "1D", option)
        assert trans_coords.shape == (1, 4, 2)
        cv2.imwrite("test_api/1D_{}.png".format(option), trans_image)
    # 2A
    print("START TRANSFORM 2A")
    options = ["lt-s", "lt-w", "ct-s", "ct-w", "rt-s", "rt-w", "rm-s", "rm-w",
               "rb-s", "rb-w", "cb-s", "cb-w", "lb-s", "lb-w", "lm-s", "lm-w"]
    for option in tqdm.tqdm(options):
        ############################### 배경 선택 ################################
        base="Arles-15t.jpg"
        ########################################################################
        trans_image, trans_coords = transformImage(origin, coords, "2A", option, base="Arles-15t.jpg")
        assert trans_coords.shape == (1, 4, 2)
        cv2.imwrite("test_api/2A_{}.png".format(option), trans_image)
        
    # 2B
    print("START TRANSFORM 2B")
    options = ["l-s", "l-w", "lt-s", "lt-w", "t-s", "t-w", "rt-s", "rt-w",
               "r-s", "r-w", "rb-s", "rb-w", "b-s", "b-w", "lb-s", "lb-w"]
    for option in tqdm.tqdm(options):
        trans_image, trans_coords = transformImage(origin, coords, "2B", option)
        assert trans_coords.shape == (1, 4, 2)
        cv2.imwrite("test_api/2B_{}.png".format(option), trans_image)

if __name__ == "__main__":
    test()
