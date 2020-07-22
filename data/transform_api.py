import abc
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
    pass

class Folded(Shift):
    pass

class Curve(Shift):
    pass


def transformImage(image, coords, transformType, option):

    height, width, channel = image.shape
    tran = None

    if transformType is "1A":
        raise NotImplementedError

    elif transformType is "1B":
        raise NotImplementedError

    elif transformType is "1C":
        tran = Folded(width=width, height=height, spacing=40)

    elif transformType is "1D":
        raise NotImplementedError

    elif transformType is "2A":
        raise NotImplementedError

    elif transformType is "2B":
        raise NotImplementedError

    assert isinstance(tran, Transform)
    transImage = tran.run(image)
    transCoords = tran.transform_points(coords)

    return transImage, transCoords
