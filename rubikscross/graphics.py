import numpy as np
import numpy.typing as npt

from actions import Action
from graphics_move_functions_map import GraphicsMoveFunctionsMap
from interfaces import GraphicsInterface


class CroixPharmaGraphics(GraphicsInterface):
    def __init__(self, tiles: list[npt.NDArray], colors: npt.NDArray, move_func_map: GraphicsMoveFunctionsMap, animation_max_length: int):
        self.colors = colors.copy()

        # colorize tiles
        self.tiles = []
        for tile, color in zip(tiles, colors):
            tile = tile.reshape((*tile.shape, 1))
            color = color.reshape((1, 1, 3))
            self.tiles.append((1 - tile) * color)

        self.tile_size = self.tiles[0].shape[0]
        self.animation_max_length: int = animation_max_length
        self.frame_config_list: list[tuple[Callable, npt.NDArray]] = []
        self.frame_continuous_index: int = 0
        self.frame: npt.NDArray
        self.hint_frame: npt.NDArray
        self.move_func_map = move_func_map

    def initialize_frame(self, board):
        self.frame = self.generate_frame(board)

    def initialize_hint_frame(self, initial_board):
        self.hint_frame = self.generate_frame(initial_board.copy())

    @staticmethod
    def insert_tile(image, tile, coord_ij):
        th, tw = tile.shape[:2]
        i, j = coord_ij
        image[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tile

    def generate_frame(self, board: npt.NDArray) -> npt.NDArray:
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
        tsize = self.tiles[0].shape[0]
        frame_size = bsize * tsize

        res = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

        for i in range(bsize):
            for j in range(bsize):
                ind = int(board[i, j])
                CroixPharmaGraphics.insert_tile(res, self.tiles[ind], (i, j))
        return res

    def update_animation(self, action: Action, board: npt.NDArray, frame_count: int | None = None):
        self.frame_config_list.append((action, board.copy()))

    def get_next_frame(self, height: int | None = None) -> npt.NDArray:
        fci_max = len(self.frame_config_list)
        remaining_length = fci_max - self.frame_continuous_index

        # update index and avoid out of bounds
        min_step = 1 / 20
        step = max(remaining_length / 5, min_step)
        self.frame_continuous_index = min(self.frame_continuous_index + step, fci_max)

        #  frame_continuous_index is N+t, where N is an integer and t a float in interval [0, 1[
        N = int(self.frame_continuous_index)
        t = self.frame_continuous_index % 1.0

        if N == len(self.frame_config_list) and N > 0:
            N -= 1
            t += 1

        # drop the N first configs
        self.frame_config_list = self.frame_config_list[N:]
        need_new_frame = self.frame_continuous_index > min_step / 2 and len(self.frame_config_list) > 0
        if need_new_frame:
            # generate the frame given its config
            action, board = self.frame_config_list[0]
            move_func = self.move_func_map[action]
            tiled_board = self.generate_frame(board)
            self.frame = move_func(board=tiled_board, factor=t)

        res = self.frame
        if height is not None:
            # resize it to given height
            h = res.shape[0]
            f = height / h
            res = cv2.resize(res, dsize=None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)

        self.frame_continuous_index = t
        return res

    def get_hint_frame(self) -> npt.NDArray:
        return self.hint_frame

    def get_tile_size(self) -> int:
        return self.tile_size
