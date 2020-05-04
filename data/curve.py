import fire

from base_transform import Shift


class Curve(Shift):
    """
    문서 말림 noise

    **How to use??**

    1. 가로 말림
    (약)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png" is_horizon=True flexure=0.1
    (강)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png" is_horizon=True flexure=0.2

    2. 세로 말림
    (약)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png" is_horizon=False flexure=0.1
    (강)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png" is_horizon=False flexure=0.2

    """

    def __init__(self, flexure=0.1, direction="down", **kwargs):
        """
        cureve origin paper to curved paper
        :param width: 문서 너비(pixel)
        :param height: 문서 높이(pixel)
        :param spacing: shift 하는 문서 부분 크기, 클수록 빠르게 변환하지만
            정확도가 떨어진다. default: 20
        :param is_horizon: 가로 말림이면 True, 세로 말림이면 False
        :param verbose: 디버깅용이면 True
        :param flexure: maximum shifted pixel within ratio
        :param direction: curve direction, "down" or "up"
        """
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


if __name__ == "__main__":
    curve = Curve(width=960, height=1280, direction="up", is_horizon=False, verbose=True)
    curve.run(3, ipt="report.png")
    #fire.Fire(Curve)
