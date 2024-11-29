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


class CoolFuncMap(GraphicPainterInterface):
    def __init__(self, tile_size, dst_size, difficulty):
        self.tile_size = tile_size
        self.board_size = 3 * tile_size * difficulty
        self.dst_size = dst_size

        # shift of the cross roll. Should be 'tile_size' if we were in board_size, but here we are in dst_size We must multiply it by dst_size/board_size
        tile_frame_size = int(round(self.tile_size * self.dst_size / self.board_size))

        self.mx0, self.my0 = np.meshgrid(np.arange(self.dst_size, dtype=np.float32), np.arange(self.dst_size, dtype=np.float32))
        zeros_frame = np.zeros((self.dst_size, self.dst_size), dtype=np.float32)

        # meshgrid of a single tile
        mx0_tile, my0_tile = np.meshgrid(np.arange(tile_frame_size, dtype=np.float32), np.arange(tile_frame_size, dtype=np.float32))
        # tile coordinates normalized in [-1, 1]
        mx0_unit_tile = mx0_tile * 2 / (tile_frame_size - 1) - 1
        my0_unit_tile = my0_tile * 2 / (tile_frame_size - 1) - 1
        zeros_tile = np.zeros_like(mx0_unit_tile)

        self.frame_count = 10
        self.delta_mesh_xy: list[tuple[npt.NDArray, npt.NDArray]] = []
        self.frame_pop_mesh_xya: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray]] = []

        # (y, x) coordinates on the board of the tiles that will pop in and pop out of the board
        pop_in_tiles_yx = [(k, difficulty) for k in range(difficulty)]
        pop_in_tiles_yx += [(k + difficulty, 0) for k in range(difficulty)]
        pop_in_tiles_yx += [(k + 2 * difficulty, difficulty) for k in range(difficulty)]

        pop_out_tiles_yx = [(k, 2 * difficulty - 1) for k in range(difficulty)]
        pop_out_tiles_yx += [(k + difficulty, 3 * difficulty - 1) for k in range(difficulty)]
        pop_out_tiles_yx += [(k + 2 * difficulty, 2 * difficulty - 1) for k in range(difficulty)]

        for k in range(self.frame_count):
            print(f"Compute mesh {k + 1}/{self.frame_count}")
            factor = k / (self.frame_count - 1)

            mdx = zeros_frame + factor * tile_frame_size
            mdy = zeros_frame.copy()
            mda = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.float32).repeat(dst_size, axis=0).repeat(dst_size, axis=1)
            self.delta_mesh_xy.append((mdx, mdy))

            # factor: 0 -> 0.5 -> 1
            alpha_in = max(0.0, factor - 0.5) * 2  # alpha_in: 0.0 -> 0.0 -> 1.0
            alpha_out = max(0.0, -factor + 0.5) * 2  # alpha_out: 1.0 -> 0.0 -> 0.0
            radius_in = max(1e-6, alpha_in)  # radius_in: 1e-6 -> 1e-6 -> 1.0
            radius_out = max(1e-6, alpha_out)  # r_in: 1.0 -> 1e-6 -> 1e-6

            zoom_in = 1 / radius_in
            zoom_out = 1 / radius_out
            mx_tile_in = mx0_unit_tile * zoom_in
            my_tile_in = my0_unit_tile * zoom_in
            mx_tile_out = mx0_unit_tile * zoom_out
            my_tile_out = my0_unit_tile * zoom_out
            bound_in_mask = (abs(mx_tile_in) > 1) + (abs(my_tile_in) > 1)
            bound_out_mask = (abs(mx_tile_out) > 1) + (abs(my_tile_out) > 1)
            ma_tile_in = zeros_tile.copy() + alpha_in
            ma_tile_out = zeros_tile.copy() + alpha_out
            ma_tile_in[bound_in_mask] = 0
            ma_tile_out[bound_out_mask] = 0
            mpx = zeros_frame.copy()
            mpy = zeros_frame.copy()
            mpa = zeros_frame.copy()
            for tile_yx in pop_in_tiles_yx:
                y0 = tile_yx[0] * tile_frame_size
                x0 = tile_yx[1] * tile_frame_size
                mpx[y0:y0 + tile_frame_size, x0:x0 + tile_frame_size] = x0 + mx0_tile + mx_tile_in
                mpy[y0:y0 + tile_frame_size, x0:x0 + tile_frame_size] = y0 + my0_tile + my_tile_in
                mpa[y0:y0 + tile_frame_size, x0:x0 + tile_frame_size] = ma_tile_in

            for tile_yx in pop_out_tiles_yx:
                y0 = tile_yx[0] * tile_frame_size
                x0 = tile_yx[1] * tile_frame_size
                mpx[y0:y0 + tile_frame_size, x0:x0 + tile_frame_size] = x0 + mx0_tile + mx_tile_out
                mpy[y0:y0 + tile_frame_size, x0:x0 + tile_frame_size] = y0 + my0_tile + my_tile_out
                mpa[y0:y0 + tile_frame_size, x0:x0 + tile_frame_size] = ma_tile_out

            self.frame_pop_mesh_xya.append((mpx, mpy, mpa))

    @staticmethod
    def _blend_u8(img0, img1, t: float):
        return cv2.addWeighted(img0, 1 - t, img1, t, 0)

    @staticmethod
    def _mask_blend(img0, img1, alpha):
        f = np.float16
        img0_f = img0.astype(f)
        img1_f = img1.astype(f)
        alpha_f = alpha.astype(f)
        if alpha_f.ndim == 2 and img0_f.ndim == 3:
            alpha_f.shape += (1,)
        res_f = img0_f + alpha_f * (img1_f - img0_f)
        return res_f.astype(img0.dtype)

    def cross_rot90_right(self, board, factor: float = 1.0):
        new_board = RubiksCross.cross_rot90_right(board, factor)
        return self._blend_u8(board, new_board, factor)

    def cross_rot90_left(self, board, factor: float = 1.0):
        new_board = RubiksCross.cross_rot90_left(board, factor)
        return self._blend_u8(board, new_board, factor)

    def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
        i = int(round(factor * (self.frame_count - 1)))
        mdx, mdy = self.delta_mesh_xy[i]
        new_board = cv2.remap(board, self.mx0 - mdx, self.my0 - mdy, interpolation=cv2.INTER_LINEAR)

        mpx, mpy, mpa = self.frame_pop_mesh_xya[i]
        new_board_2 = cv2.remap(board, mpx, mpy, interpolation=cv2.INTER_LINEAR)
        # new_board_2 = self._mask_blend(new_board, new_board_2, mpa)
        return new_board_2

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

        gird_size = board.shape[0] * tiles[0].shape[0]

        tiled_board = generate_frame(board, tiles)
        tiled_board = cv2.resize(tiled_board, dsize=(self.dst_size, self.dst_size), interpolation=cv2.INTER_NEAREST)

        alpha = make_square_mask(gird_size, self.dst_size).copy()
        alpha.shape += (1,)

        tiled_board *= alpha // 255

        img_u8 = move_funcs[action](tiled_board, factor)

        # alpha *= (cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)

        alpha = np.zeros_like(img_u8[:, :, 0]) + 255
        return img_u8, alpha
