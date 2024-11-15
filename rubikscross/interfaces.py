import abc

import numpy.typing as npt

from actions import Action


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


class GraphicsMoveFunctionsMapInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def cross_rot90_right(self, board, factor: float = 1.0):
        pass

    @abc.abstractmethod
    def cross_rot90_left(self, board, factor: float = 1.0):
        pass

    @abc.abstractmethod
    def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
        pass

    @abc.abstractmethod
    def cross_roll_left(self, board: npt.NDArray, factor: float = 1.0):
        pass

    @abc.abstractmethod
    def cross_roll_up(self, board: npt.NDArray, factor: float = 1.0):
        pass

    @abc.abstractmethod
    def cross_roll_down(self, board: npt.NDArray, factor: float = 1.0):
        pass

    @abc.abstractmethod
    def __getitem__(self, action: Action):
        pass

class GraphicsInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def initialize_frame(self, board: npt.NDArray):
        pass

    @abc.abstractmethod
    def initialize_hint_frame(self, board: npt.NDArray):
        pass

    @abc.abstractmethod
    def generate_frame(self, board: npt.NDArray) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def update_animation(self, action: Action, board: npt.NDArray, frame_count: int | None = None):
        pass

    @abc.abstractmethod
    def get_next_frame(self, height: int | None = None) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def get_hint_frame(self) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def get_tile_size(self) -> int:
        pass
