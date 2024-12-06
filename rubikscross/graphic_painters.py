import dataclasses
import functools

import cv2
import numpy as np
import numpy.typing as npt

from .actions import Action
from .interfaces import GraphicPainterInterface
from .rubikscross import RubiksCross


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
    Creates an image made of tiles, placed according to the board.

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


@dataclasses.dataclass
class RollingTile:
    coord_ij_out: tuple[int, int]  # coordinate of rolling out tile (where it disappears)
    coord_ij_in: tuple[int, int]  # coordinate of rolling in tile (where it reappears)
    tile_id: int


class Roll3DFuncMap(GraphicPainterInterface):
    def __init__(self, tiles, dst_size, difficulty):
        self.tile_size = tiles[0].shape[0]
        self.dst_size = dst_size
        board_size = difficulty * 3

        # todo: precompute
        big_tiles_size = self.dst_size // board_size
        tile_size = tiles[0].shape[0]
        border = int(round(big_tiles_size * 0.5 / tile_size))
        border_mask = np.zeros((big_tiles_size, big_tiles_size, 3), dtype=np.uint8)
        border_mask[border:-border, border:-border] = 1
        self.big_tiles = [cv2.resize(tile, dsize=(big_tiles_size, big_tiles_size), interpolation=cv2.INTER_NEAREST) * border_mask for tile in tiles]
        self.big_tiles = [tile * border_mask for tile in self.big_tiles]


    @staticmethod
    def insert_subimg(img, subimg, coord_yx):
        """
        Insert subimg in image
        Parameters
        ----------
        img: background image
        subimg: small image to be insterted
        coord_yx: y, x coordinates (in image space) where the top-left corner of the subimg must be inserted

        Returns
        -------
        No return, this function modifies img inplace
        """
        H, W = img.shape[:2]
        h, w = subimg.shape[:2]
        y, x = coord_yx

        # img coordinates
        y0 = max(0, y)
        x0 = max(0, x)
        y1 = min(H, y + h)
        x1 = min(W, x + w)

        # subimg coordinates
        i0 = max(0, -y)  # crop top of subimg if (x < 0),
        j0 = max(0, -x)  # crop left of subimg if (x < 0),
        i1 = i0 + y1 - y0
        j1 = j0 + x1 - x0

        img[y0:y1, x0:x1] = subimg[i0:i1, j0:j1]

    @classmethod
    def draw_roll_out_tiles(cls, img, big_tiles, rolling_tiles_list, factor):

        def zoom_out(quad_xy, factor):
            quad_xy = quad_xy.copy()
            quad_xy = (quad_xy - 0.5) * (1 - factor) + 0.5 + np.array([0.5, 0]) * factor
            return quad_xy

        def rot3d(quad_xy, factor):
            quad_xy = quad_xy.copy()
            theta = np.pi * factor
            cos = np.cos(theta)
            sin = np.sin(theta)

            # center
            quad_xy -= 0.5

            quad_xy[:, 0] *= cos
            # quad_xy[2:4, 1] *= (cos*0.5+0.5)

            # uncenter
            quad_xy += 0.5

            # slide x
            quad_xy[:, 0] += 0.5 * factor
            return quad_xy

        tiles_size = big_tiles[0].shape[0]
        img_size = img.shape[0]
        board_size = img_size / tiles_size

        # shift = big_tiles_size*factor
        square_xy = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0],
        ], dtype=float)
        quad_out_xy = rot3d(square_xy, factor)
        hmat_out, _ = cv2.findHomography(square_xy * tiles_size, quad_out_xy * tiles_size)

        v10 = np.array([1.0, 0.0])
        vm11 = np.array([-1.0, 1.0])
        # square_inv_xy = v10 - square_xy
        quad_in_xy = rot3d(square_xy, 1 - factor)
        hmat_in, _ = cv2.findHomography(
            (v10 + vm11 * square_xy) * tiles_size,
            (v10 + vm11 * quad_in_xy) * tiles_size)

        # quad_in_xy = square_xy.copy()
        # quad_in_xy[:2, 0] = factor
        # hmat_in, _ = cv2.findHomography(square_xy * tiles_size, quad_out_xy * tiles_size)

        for rolling_tile in rolling_tiles_list:
            tile = big_tiles[rolling_tile.tile_id]
            if hmat_out is not None:
                i_out, j_out = rolling_tile.coord_ij_out
                y_out = int(round(i_out * img_size / board_size))
                x_out = int(round(j_out * img_size / board_size))
                alpha = max(0, (1 - factor) * 2 - 1)
                tile_out = cv2.warpPerspective(tile, hmat_out, dsize=(tiles_size, tiles_size)) * alpha
                cls.insert_subimg(img, tile_out, (y_out, x_out))

            if hmat_in is not None:
                i_in, j_in = rolling_tile.coord_ij_in
                y_in = int(round(i_in * img_size / board_size))
                x_in = int(round(j_in * img_size / board_size))
                alpha = max(0, factor * 2 - 1)
                tile_in = cv2.warpPerspective(tile, hmat_in, dsize=(tiles_size, tiles_size)) * alpha
                cls.insert_subimg(img, tile_in.astype(np.uint8), (y_in, x_in))

    @classmethod
    def draw_sliding_tiles(cls, img, tiles, board, factor):
        board_size = board.shape[0]
        tile_size = tiles[0].shape[0]
        img_size = img.shape[0]
        assert board.shape[0] == board.shape[1]
        assert tiles[0].shape[0] == tiles[0].shape[1]
        assert img.shape[0] == img.shape[1]
        assert abs(img_size / tile_size - board_size) <= 1

        shift = factor * tile_size

        for i in range(board_size):
            for j in range(board_size):
                tile_id = board[i, j]
                if tile_id == 0:
                    continue
                tile = tiles[tile_id]
                y = int(round(i * img_size / board_size))
                x = int(round(j * img_size / board_size + shift))
                cls.insert_subimg(img, tile, (y, x))

    def cross_roll_right(self, board: npt.NDArray, factor: float = 1.0):
        h, w = board.shape[:2]
        assert h % 3 == 0
        assert h == w
        n = h // 3
        board = board.copy()

        rolling_tiles_list: list[RollingTile] = []
        # We want (i, j0) to point to the last tile of the row of the rubikscross (disapearing tile coordinates)
        # We want (i, j1) to point to the first tile of the row of the rubikscross (appearing tile coordinates)
        for i in range(h):
            # The cross can be splited in 5 square. 'n' is the size of a square
            # These squares can be placed on a 3x3 grid.
            # k is the x coordinate of the square on the 3x3 grid
            k0 = [1, 2, 1][3 * i // h]
            j0 = n * (k0 + 1) - 1
            k1 = [1, 0, 1][3 * i // h]
            j1 = n * k1
            id = int(board[i, j0])
            rolling_tile = RollingTile((i, j0), (i, j1), id)
            rolling_tiles_list.append(rolling_tile)

        # replace tiles ids by 0 in the board (0 is id of empty tile)
        for rolling_tile in rolling_tiles_list:
            i, j = rolling_tile.coord_ij_out
            board[i, j] = 0

        img = np.zeros((self.dst_size, self.dst_size, 3), dtype=np.uint8)

        self.draw_roll_out_tiles(img, self.big_tiles, rolling_tiles_list, factor)
        self.draw_sliding_tiles(img, self.big_tiles, board, factor)

        return img

    def cross_roll_left(self, board: npt.NDArray, factor: float = 1.0):
        board = cv2.rotate(board, cv2.ROTATE_180)
        board = self.cross_roll_right(board, factor)
        return cv2.rotate(board, cv2.ROTATE_180)

    def cross_roll_up(self, board: npt.NDArray, factor: float = 1.0):
        board = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)
        board = self.cross_roll_right(board, factor)
        return cv2.rotate(board, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def cross_roll_down(self, board: npt.NDArray, factor: float = 1.0):
        board = cv2.rotate(board, cv2.ROTATE_90_COUNTERCLOCKWISE)
        board = self.cross_roll_right(board, factor)
        return cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)

    def cross_rot90_right(self, board, factor: float = 1.0):
        h, w = board.shape[:2]
        assert h % 3 == 0
        assert h == w

        img = np.zeros((self.dst_size, self.dst_size, 3), dtype=np.uint8)
        self.draw_sliding_tiles(img, self.big_tiles, board, 0)

        return RubiksCross.cross_rot90_right(img, factor)

    def cross_rot90_left(self, board, factor: float = 1.0):
        return self.cross_rot90_right(board, -factor)

    def compute(self, action: Action, board: npt.NDArray, tiles: list[npt.NDArray], factor: float = 1.0) -> tuple[npt.NDArray, npt.NDArray]:
        move_funcs = {
            Action.UP: self.cross_roll_up,
            Action.DOWN: self.cross_roll_down,
            Action.LEFT: self.cross_roll_left,
            Action.RIGHT: self.cross_roll_right,
            Action.ROT_LEFT: self.cross_rot90_left,
            Action.ROT_RIGHT: self.cross_rot90_right,
        }

        img_u8 = move_funcs[action](board, factor)
        alpha = (cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8) * 255
        return img_u8, alpha


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
