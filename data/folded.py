import fire

from base_transform import Shift


class Folded(Shift):
    """
    문서 접힘 noise

    **How to use??**

    1. 가로 접힘
    (약)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png"
        --is_horizon=True --up_slope=0.1 --down_slope=0.1
    (강)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png"
        --is_horizon=True --up_slope=0.2 --down_slope=0.2

    2. 세로 접힘
    (약)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png"
        --is_horizon=False --up_slope=0.1 --down_slope=0.1
    (강)
    $> python curve.py --width=960 --height=1280 --channel=3 --ipt="sample.png"
        --is_horizon=False --up_slope=0.2 --down_slope=0.2
    """
    def __init__(self, up_slope=0.1, down_slope=0.2, **kwargs):
        """
        cureve origin paper to curved paper
        :param width: 문서 너비(pixel)
        :param height: 문서 높이(pixel)
        :param spacing: shift 하는 문서 부분 크기, 클수록 빠르게 변환하지만
            정확도가 떨어진다. default: 20
        :param is_horizon: 가로 접힘이면 True, 세로 접힘이면 False
        :param verbose: 디버깅용이면 True
        :param up_slope: upper slide amount within ratio
        :param down_slope: below slide amount within ratio
        """
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


if __name__ == "__main__":
    fire.Fire(Folded)
