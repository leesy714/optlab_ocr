import abc
import cv2
import numpy as np


class Transform(metaclass=abc.ABCMeta):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    @abstractmethod
    def transform_image(self, image):
        pass

    @abstractmethod
    def transform_points(self, points):
        pass

class Shift(Transform):

    def __init__(self, width, height, spacing=20, is_horizon=True):
        """
        :param spacing: length of each square pixel
        """
        super().__init__(width, height, verbose)
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

    def __init__(self, width, height, spacing=20, sources=[(0, 0)]):
        """
        :param width: width of the document
        :param height: height of the document
        :param spacing: length of each square pixel
        :param sources: light sources in list format
            e.g. [(0, 0), (0, 1280)]
        """
        super().__init__(width, height)
        self.spacing = spacing
        assert len(sources) > 0, "at least one light source is required"
        self.sources = sources

    def get_regularizer(self):
        max_norm = 0.0
        for width, height in [(0, 0), (self.width, 0),
                              (0, self.height), (self.width, self.height)]:
            norms = [(width-x)**2 + (height-y)**2 for x, y in self.sources]
            norm = sum(norms)
            max_norm = max(max_norm, norm)
        return 150.0 / max_norm

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
                origin[heights, widths, :] -= int(regularizer * norm)
        origin = origin.clip(min=0, max=255).astype(np.uint8)
        return origin

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
        raise NotImplementedError

    elif transformType is "2B":
        pos, strength = option.split("-")
        kwargs["focus"] = 50 if strength is "s" else 25
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
        tran = Curve(width=width, height=height, spacing=40, **kwrags)

    elif transformType is "1D":
        tran = Folded(width=width, height=height, spacing=40, **kwrags)

    elif transformType is "2A":
        raise NotImplementedError

    elif transformType is "2B":
        tran = Shadow(width=width, height=height, spacing=40, **kwrags)

    assert isinstance(tran, Transform)
    transImage = tran.run(image)
    transCoords = tran.transform_points(coords)

    return transImage, transCoords
