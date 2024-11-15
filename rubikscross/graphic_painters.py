import functools

import cv2
import numpy as np
import numpy.typing as npt

from actions import Action
from interfaces import GraphicPainterInterface
from rubikscross import RubiksCross


@functools.lru_cache(maxsize=1)
def make_square_mask(grid_size, dst_size):
    square_size = dst_size // grid_size
    rad_ratio = 0.8
    topleft = int(round((1 - rad_ratio) * square_size / 2))
    botright = int(round((1 + rad_ratio) * square_size / 2))

    square = np.zeros((square_size, square_size), dtype=np.uint8)
    square[topleft:botright, topleft:botright] = 255

    square_mask = cv2.repeat(square, grid_size, grid_size)

    if square_mask.shape[0] != dst_size:
        square_mask = cv2.resize(square_mask, (dst_size, dst_size), interpolation=cv2.INTER_NEAREST)
    return square_mask


def insert_tile(image, tile, coord_ij):
    th, tw = tile.shape[:2]
    i, j = coord_ij
    image[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tile


def generate_frame(board: npt.NDArray, tiles: list[npt.NDArray]) -> npt.NDArray:
    """
    This will create an image made of tiles, placed according to the board.

    Each value in the board is used as an index. This index is used to select a tile in the tile list

    Parameters
    ----------
    board: (m0, n0) array

    Returns
    -------
    An array of dimension (m0*m1, n0*n1) where (m1, n1) is the dimension of the tiles
    """
    bsize = board.shape[0]
    tsize = tiles[0].shape[0]
    frame_size = bsize * tsize

    res = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    for i in range(bsize):
        for j in range(bsize):
            ind = int(board[i, j])
            insert_tile(res, tiles[ind], (i, j))
    return res


class RollFuncMap(GraphicPainterInterface):
    def __init__(self, tile_size, dst_size):
        self.tile_size = tile_size
        self.dst_size = dst_size

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

    def compute(self, action: Action, board: npt.NDArray, tiles: list[npt.NDArray], factor: float = 1.0) -> tuple[npt.NDArray, npt.NDArray]:
        move_funcs = {
            Action.UP: self.cross_roll_up,
            Action.DOWN: self.cross_roll_down,
            Action.LEFT: self.cross_roll_left,
            Action.RIGHT: self.cross_roll_right,
            Action.ROT_LEFT: self.cross_rot90_left,
            Action.ROT_RIGHT: self.cross_rot90_right,
        }

        tiled_board = generate_frame(board, tiles)

        img_u8 = move_funcs[action](tiled_board, factor)

        alpha = make_square_mask(img_u8.shape[0], self.dst_size).copy()
        img_u8 = cv2.resize(img_u8, dsize=(self.dst_size, self.dst_size), interpolation=cv2.INTER_NEAREST)

        alpha *= (cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)
        return img_u8, alpha


class FadeFuncMap(GraphicPainterInterface):
    def __init__(self, tile_size, dst_size):
        self.tile_size = tile_size
        self.dst_size = dst_size

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

    def compute(self, action: Action, board: npt.NDArray, tiles: list[npt.NDArray], factor: float = 1.0) -> tuple[npt.NDArray, npt.NDArray]:
        move_funcs = {
            Action.UP: self.cross_roll_up,
            Action.DOWN: self.cross_roll_down,
            Action.LEFT: self.cross_roll_left,
            Action.RIGHT: self.cross_roll_right,
            Action.ROT_LEFT: self.cross_rot90_left,
            Action.ROT_RIGHT: self.cross_rot90_right,
        }

        tiled_board = generate_frame(board, tiles)

        img_u8 = move_funcs[action](tiled_board, factor)

        alpha = make_square_mask(img_u8.shape[0], self.dst_size).copy()
        img_u8 = cv2.resize(img_u8, dsize=(self.dst_size, self.dst_size), interpolation=cv2.INTER_NEAREST)

        alpha *= (cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)
        return img_u8, alpha

# class CoolFuncMap(GraphicPainterInterface):
#     def __init__(self, dst_size, tile_size, difficulty):
#         self.tile_size = tile_size
#         self.board_size = 3 * tile_size * difficulty
#         self.dst_size = dst_size
#
#         mx0, my0 = np.meshgrid(range(self.board_size), range(self.board_size))
#
#         self.frame_count = 100
#         self.mesh_frames: list[tuple[npt.NDArray, npt.NDArray]] = []
#         for k in range(self.frame_count):
#             factor = (k + 1) / self.frame_count
#
#             mx1 = RubiksCross.cross_roll_right(mx0, self.tile_size, -factor)
#             my1 = RubiksCross.cross_roll_right(my0, self.tile_size, -factor)
#
#             factors = np.clip(mx0 / self.board_size - 1 + factor * 2, a_min=0, a_max=1)
#             mx2 = mx0 + (mx1 - mx0) * factors
#             my2 = my0 + (my1 - my0) * factors
#
#             mfx = np.zeros_like(mx2, dtype=np.float32)
#             mfy = np.zeros_like(my2, dtype=np.float32)
#
#             # invert maps
#             for i in range(self.board_size):
#                 for j in range(self.board_size):
#                     x, y = int(mx2[i, j]), int(my2[i, j])
#                     # x, y = int(round(mx0[i, j])), int(round(my0[i, j]))
#                     # x = min(x+k, 47)
#                     mfx[y, x] = j
#                     mfy[y, x] = i
#             self.mesh_frames.append((mfx, mfy))
#
#             # cv2.remap(mx0, mx2, my2, interpolation=cv2.INTER_NEAREST, )
#             # cv2.remap(my0, mx2, my2, interpolation=cv2.INTER_NEAREST)
#
#     @staticmethod
#     def _blend_u8(img0, img1, t):
#         return cv2.addWeighted(img0, 1 - t, img1, t, 0)
#
#     def cross_rot90_right(self, board, factor: float = 1.0):
#         new_board = RubiksCross.cross_rot90_right(board, factor)
#         return self._blend_u8(board, new_board, factor)
#
#     def cross_rot90_left(self, board, factor: float = 1.0):
#         new_board = RubiksCross.cross_rot90_left(board, factor)
#         return self._blend_u8(board, new_board, factor)
#
#     def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
#         i = int(round(factor * self.frame_count)) - 1
#         print(i, factor)
#         mx, my = self.mesh_frames[i]
#         new_board = cv2.remap(board, mx, my, interpolation=cv2.INTER_LINEAR)
#         return new_board
#
#     def cross_roll_left(self, board: npt.NDArray, factor: float = 1.0):
#         new_board = RubiksCross.cross_roll_left(board, self.tile_size, 1)
#         return self._blend_u8(board, new_board, factor)
#
#     def cross_roll_up(self, board: npt.NDArray, factor: float = 1.0):
#         new_board = RubiksCross.cross_roll_up(board, self.tile_size, 1)
#         return self._blend_u8(board, new_board, factor)
#
#     def cross_roll_down(self, board: npt.NDArray, factor: float = 1.0):
#         new_board = RubiksCross.cross_roll_down(board, self.tile_size, 1)
#         return self._blend_u8(board, new_board, factor)
#
#     def __getitem__(self, action):
#         return {
#             Action.UP: self.cross_roll_up,
#             Action.DOWN: self.cross_roll_down,
#             Action.LEFT: self.cross_roll_left,
#             Action.RIGHT: self.cross_roll_right,
#             Action.ROT_LEFT: self.cross_rot90_left,
#             Action.ROT_RIGHT: self.cross_rot90_right,
#         }[action]
