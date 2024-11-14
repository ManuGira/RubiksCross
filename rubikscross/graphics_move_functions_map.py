import numpy.typing as npt

from actions import Action
from rubikscross import RubiksCross


class GraphicsMoveFunctionsMap:
    def __init__(self, tile_size):
        self.tile_size = tile_size

    @staticmethod
    def cross_rot90_right(board, factor: float = 1.0):
        return RubiksCross.cross_rot90_right(board, factor)

    @staticmethod
    def cross_rot90_left(board, factor: float = 1.0):
        return RubiksCross.cross_rot90_left(board, factor)

    def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_right(board, self.tile_size, factor)

    def cross_roll_left(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_left(board, self.tile_size, factor)

    def cross_roll_up(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_up(board, self.tile_size, factor)

    def cross_roll_down(self, board: npt.NDArray, factor: float = 1.0):
        return RubiksCross.cross_roll_down(board, self.tile_size, factor)

    def __getitem__(self, item):
        return {
            Action.UP: self.cross_roll_up,
            Action.DOWN: self.cross_roll_down,
            Action.LEFT: self.cross_roll_left,
            Action.RIGHT: self.cross_roll_right,
            Action.ROT_LEFT: self.cross_rot90_left,
            Action.ROT_RIGHT: self.cross_rot90_right,
        }[item]
