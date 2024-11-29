import abc

import numpy.typing as npt

from .actions import Action


class GameAppInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_image_u8(self, img_u8: npt.NDArray):
        """
        Sets the image to be displayed.

        The `image` argument should be a 2D numpy array of uint8 representing the pixels.
        Values range from 0 (off) to 255 (brightest).

        For float images, see function: set_image
        """
        pass

    @abc.abstractmethod
    def run(self):
        pass


class MixerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def play_sound(self, action: Action):
        pass


class GraphicPainterInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute(self, action: Action, board: npt.NDArray, tiles: list[npt.NDArray], factor: float = 1.0) -> tuple[npt.NDArray, npt.NDArray]:
        pass
