import cv2
import numpy as np
import numpy.typing as npt

from actions import Action
from interfaces import GraphicsMoveFunctionsMapInterface
from rubikscross import RubiksCross


class RollFuncMap(GraphicsMoveFunctionsMapInterface):
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def cross_rot90_right(self, board, factor: float = 1.0):
        return RubiksCross.cross_rot90_right(board, factor)

    def cross_rot90_left(self, board, factor: float = 1.0):
        return RubiksCross.cross_rot90_left(board, factor)

    def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_right(board, self.tile_size, factor)

    def cross_roll_left(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_left(board, self.tile_size, factor)

    def cross_roll_up(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_up(board, self.tile_size, factor)

    def cross_roll_down(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_down(board, self.tile_size, factor)

    def __getitem__(self, action):
        return {
            Action.UP: self.cross_roll_up,
            Action.DOWN: self.cross_roll_down,
            Action.LEFT: self.cross_roll_left,
            Action.RIGHT: self.cross_roll_right,
            Action.ROT_LEFT: self.cross_rot90_left,
            Action.ROT_RIGHT: self.cross_rot90_right,
        }[action]


class FadeFuncMap(GraphicsMoveFunctionsMapInterface):
    def __init__(self, tile_size):
        self.tile_size = tile_size

    @staticmethod
    def _blend_u8(img0, img1, t):
        return cv2.addWeighted(img0, 1 - t, img1, t, 0)

    def cross_rot90_right(self, board, factor: float = 1.0):
        new_board = RubiksCross.cross_rot90_right(board, factor)
        return self._blend_u8(board, new_board, factor)

    def cross_rot90_left(self, board, factor: float = 1.0):
        new_board = RubiksCross.cross_rot90_left(board, factor)
        return self._blend_u8(board, new_board, factor)

    def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
        new_board = RubiksCross.cross_roll_right(board, self.tile_size, 1)
        return self._blend_u8(board, new_board, factor)

    def cross_roll_left(self, board: npt.NDArray, factor: float = 1.0):
        new_board = RubiksCross.cross_roll_left(board, self.tile_size, 1)
        return self._blend_u8(board, new_board, factor)

    def cross_roll_up(self, board: npt.NDArray, factor: float = 1.0):
        new_board = RubiksCross.cross_roll_up(board, self.tile_size, 1)
        return self._blend_u8(board, new_board, factor)

    def cross_roll_down(self, board: npt.NDArray, factor: float = 1.0):
        new_board = RubiksCross.cross_roll_down(board, self.tile_size, 1)
        return self._blend_u8(board, new_board, factor)

    def __getitem__(self, action):
        return {
            Action.UP: self.cross_roll_up,
            Action.DOWN: self.cross_roll_down,
            Action.LEFT: self.cross_roll_left,
            Action.RIGHT: self.cross_roll_right,
            Action.ROT_LEFT: self.cross_rot90_left,
            Action.ROT_RIGHT: self.cross_rot90_right,
        }[action]
