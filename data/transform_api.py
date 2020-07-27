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

    def transform_image(self, ipt, channel=3):

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

    def transform_image(self, ipt):

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
        print(option)
        print(degree)
        
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
        
        t_points = []
        for point in points:
            point = np.array((*point, 1))
            t_point = np.dot(self.mat, point)
            t_points.append((int(t_point[0]), int(t_point[1])))

        return t_points
    
    # transform_image
    def transform_image(self, ipt, channel=3, ipt_format="file", opt="res_pers.png", opt_format="file", base="Arles-15t.jpg", base_format="file"):
        
        self.origin = cv2.resize(ipt, dsize=(self.width, self.height))
        if channel > 1:
            self.origin = cv2.cvtColor(self.origin, cv2.COLOR_BGR2BGRA)
        self.origin = cv2.resize(self.origin, dsize=(self.width, self.height))

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
        
        print(self.origin.shape)
        self.origin = cv2.warpPerspective(self.origin, self.mat, (self.width,self.height), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0])
        print(self.origin.shape)
        
        # 배경과 합치기
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
        return imgBase
        #return self.save(imgBase, opt, opt_format)
    
    
def option_reader(transformType, option):
    kwargs = {}

    if transformType is "1A":
        raise NotImplementedError

    elif transformType is "1B":
        raise NotImplementedError

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

def transformImage(image, coords, transformType, option):

    height, width, channel = image.shape
    tran = None

    kwargs = option_reader(transformType, option)
    if transformType is "1A":
        raise NotImplementedError

    elif transformType is "1B":
        raise NotImplementedError

    elif transformType is "1C":
        tran = Curve(width=width, height=height, spacing=40, **kwargs)

    elif transformType is "1D":
        tran = Folded(width=width, height=height, spacing=40, **kwargs)

    elif transformType is "2A":
        tran = Perspective(width=width, height=height, options=option)        

    elif transformType is "2B":
        tran = Shadow(width=width, height=height, spacing=40, **kwargs)

    assert isinstance(tran, Transform)
    transImage = tran.transform_image(image)
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
    # 1B
    print("START TRANSFORM 1B")
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
        trans_image, trans_coords = transformImage(origin, coords, "2A", option)
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
